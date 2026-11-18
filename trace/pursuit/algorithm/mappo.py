"""Faithful, dependency-light port of the upstream PSTO MAPPO update.

The tensor formulas and module dimensions mirror upstream/mappo.py and nav_utils.py:
centralized 84-D critic, Beta actor, PPO clipping, GAE, ValueNorm, four independent
optimizers/schedulers, and the supervised LSTM auxiliary loss.
"""
from dataclasses import dataclass
from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F
from ..models import FullModel, STATE_DIM
from ..trajectory_prediction import TrajectoryPredictor
from ... import __name__ as _trace_name

class ValueNorm(nn.Module):
 def __init__(self,input_shape=1,beta=.995,epsilon=1e-5):
  super().__init__(); self.input_shape=torch.Size((input_shape,)) if isinstance(input_shape,int) else torch.Size(input_shape); self.beta=beta; self.epsilon=epsilon
  self.register_buffer('running_mean',torch.zeros(self.input_shape)); self.register_buffer('running_mean_sq',torch.zeros(self.input_shape)); self.register_buffer('debiasing_term',torch.tensor(0.)); self.reset_parameters()
 def reset_parameters(self): self.running_mean.zero_(); self.running_mean_sq.zero_(); self.debiasing_term.zero_()
 def _check(self,x):
  if x.shape[-len(self.input_shape):]!=self.input_shape: raise ValueError(f'Expected trailing shape {self.input_shape}, got {x.shape}')
 def running_mean_var(self):
  d=self.debiasing_term.clamp(min=self.epsilon); m=self.running_mean/d; ms=self.running_mean_sq/d; return m,(ms-m**2).clamp(min=.01)
 @torch.no_grad()
 def update(self,x):
  self._check(x); dims=tuple(range(x.dim()-len(self.input_shape))); m=x.mean(dims); ms=(x*x).mean(dims); self.running_mean.mul_(self.beta).add_(m*(1-self.beta)); self.running_mean_sq.mul_(self.beta).add_(ms*(1-self.beta)); self.debiasing_term.mul_(self.beta).add_(1-self.beta)
 def normalize(self,x): self._check(x); m,v=self.running_mean_var(); return (x-m)/torch.sqrt(v)
 def denormalize(self,x): self._check(x); m,v=self.running_mean_var(); return x*torch.sqrt(v)+m

class IndependentBeta(torch.distributions.Independent):
 def __init__(self,alpha,beta,validate_args=None): super().__init__(torch.distributions.Beta(alpha,beta),1,validate_args=validate_args)

class BetaActor(nn.Module):
 def __init__(self,action_dim=2,input_feature_dim=48):
  super().__init__(); self.hidden_layer=nn.Sequential(nn.Linear(input_feature_dim,72),nn.LayerNorm(72),nn.LeakyReLU(.01),nn.Dropout(.05),nn.Linear(72,36),nn.LayerNorm(36),nn.LeakyReLU(.01),nn.Dropout(.05)); self.alpha_transition=nn.Sequential(nn.Linear(36,18),nn.Tanh()); self.beta_transition=nn.Sequential(nn.Linear(36,18),nn.Tanh()); self.alpha_layer=nn.Linear(18,action_dim); self.beta_layer=nn.Linear(18,action_dim); self.alpha_softplus=nn.Softplus(); self.beta_softplus=nn.Softplus()
 def forward(self,x):
  h=self.hidden_layer(x); return 1+self.alpha_softplus(self.alpha_layer(self.alpha_transition(h)))+1e-6,1+self.beta_softplus(self.beta_layer(self.beta_transition(h)))+1e-6

class GAE:
 def __init__(self,gamma=.995,lmbda=.95): self.gamma=gamma; self.lmbda=lmbda
 def __call__(self,reward,terminated,value,next_value):
  # reward/value: [B,T,1], next_value [B,T,1]
  adv=torch.zeros_like(reward); gae=torch.zeros_like(reward[:,0]); alive=1-terminated.float()
  for t in reversed(range(reward.shape[1])):
   delta=reward[:,t]+self.gamma*next_value[:,t]*alive[:,t]-value[:,t]; gae=delta+self.gamma*self.lmbda*alive[:,t]*gae; adv[:,t]=gae
  return adv,adv+value

def _orthogonal(m):
 if isinstance(m,nn.Linear): nn.init.orthogonal_(m.weight,nn.init.calculate_gain('relu')); nn.init.constant_(m.bias,0.)

@dataclass
class MAPPOConfig:
 gamma:float=.995; gae_lambda:float=.95; clip_ratio:float=.1; value_clip_ratio:float=.1; entropy_coef:float=.01; actor_lr:float=6e-5; critic_lr:float=6e-5; encoder_lr:float=6e-5; sl_lr:float=8e-5; max_grad_norm:float=1.; lstm_max_grad_norm:float=1.; training_epoch_num:int=4; num_minibatches:int=1; sl_future_len:int=10; action_limit:float=.8; scheduler_steps:int=600

class RolloutStorage:
 """Explicit on-policy storage used by train_psto and unit tests."""
 def __init__(self): self.data=[]
 def add(self,**transition): self.data.append(transition)
 def clear(self): self.data.clear()
 def __len__(self): return len(self.data)

class PSTOMAPPO(nn.Module):
 def __init__(self,state_dim=13,central_dim=84,action_dim=2,config=None,device='cpu'):
  super().__init__(); self.device=torch.device(device); c=config or MAPPOConfig(); self.cfg=c; self.num_agents=2; self.action_dim=action_dim; self.gamma=c.gamma; self.gae=GAE(c.gamma,c.gae_lambda); self.clip_ratio=c.clip_ratio; self.value_clip_ratio=c.value_clip_ratio; self.entropy_coef=c.entropy_coef
  self.encoder=FullModel(state_dim=state_dim).to(self.device); self.actor=BetaActor(action_dim,48).to(self.device)
  self.critic_encoder=nn.Sequential(nn.Linear(central_dim,256),nn.LayerNorm(256),nn.ReLU(),nn.Linear(256,128),nn.LayerNorm(128),nn.ReLU()).to(self.device)
  self.critic=nn.Sequential(nn.Linear(128,64),nn.LayerNorm(64),nn.ReLU(),nn.Linear(64,1)).to(self.device)
  self.encoder.apply(_orthogonal); self.actor.apply(_orthogonal); self.critic_encoder.apply(_orthogonal); self.critic.apply(_orthogonal)
  self.lstm=TrajectoryPredictor(4,64,c.sl_future_len*3).to(self.device); self.value_norm=ValueNorm(1).to(self.device)
  self.encoder_opt=torch.optim.AdamW(self.encoder.parameters(),lr=c.encoder_lr,weight_decay=2e-4); self.actor_opt=torch.optim.AdamW(self.actor.parameters(),lr=c.actor_lr,weight_decay=2e-4); self.critic_encoder_opt=torch.optim.AdamW(self.critic_encoder.parameters(),lr=c.critic_lr,weight_decay=1e-4); self.critic_opt=torch.optim.AdamW(self.critic.parameters(),lr=c.critic_lr,weight_decay=1e-4); self.lstm_opt=torch.optim.Adam(self.lstm.parameters(),lr=c.sl_lr)
  self.encoder_scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(self.encoder_opt,max(c.scheduler_steps,1),eta_min=3e-6); self.actor_scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(self.actor_opt,max(c.scheduler_steps,1),eta_min=3e-6); self.critic_encoder_scheduler=torch.optim.lr_scheduler.ConstantLR(self.critic_encoder_opt,1.0); self.critic_scheduler=torch.optim.lr_scheduler.ConstantLR(self.critic_opt,1.0); self.lstm_scheduler=torch.optim.lr_scheduler.StepLR(self.lstm_opt,1000,.95)
 def _dist(self,image,state): return IndependentBeta(*self.actor(self.encoder(image,state)))
 def act(self,image,state,central=None,deterministic=False):
  dist=self._dist(image,state); a=dist.base_dist.mean if deterministic else dist.rsample(); lp=dist.log_prob(a); v=self.critic(self.critic_encoder(central)) if central is not None else None; return a,lp,v
 @staticmethod
 def action_to_body_velocity(action,max_speed=.8):
  angle=torch.pi*(1-2*action[...,0]); length=.9+.1*action[...,1]
  return torch.stack((length*torch.cos(angle)*max_speed,length*torch.sin(angle)*max_speed,torch.zeros_like(length)),-1)
 @staticmethod
 def action_to_velocity(action,quat,max_speed=.8):
  body=PSTOMAPPO.action_to_body_velocity(action,max_speed); q=quat/torch.linalg.vector_norm(quat,dim=-1,keepdim=True).clamp_min(1e-9); w,x,y,z=q.unbind(-1); yaw=torch.atan2(2*(w*z+x*y),1-2*(y*y+z*z)); c=torch.cos(yaw); s=torch.sin(yaw); return torch.stack((c*body[...,0]-s*body[...,1],s*body[...,0]+c*body[...,1],body[...,2]),-1)
 def update(self,rollout):
  """Run the complete on-policy PPO update.

  Rollout tensors use ``[T, B, A, ...]`` (or ``[T, B, ...]`` for the
  centralized quantities).  Returns and advantages are computed once from
  the old policy, then every epoch shuffles the time/environment samples and
  performs the configured number of minibatch updates.  This mirrors the
  upstream MAPPO train loop while retaining a plain PyTorch interface.
  """
  image=rollout['image'].to(self.device); state=rollout['state'].to(self.device)
  central=rollout['central'].to(self.device); action=rollout['action'].to(self.device)
  old_lp=rollout['log_prob'].to(self.device); reward=rollout['reward'].to(self.device)
  done=rollout['terminated'].to(self.device); old_v=rollout['value'].to(self.device)
  next_v=rollout.get('next_value',torch.zeros_like(old_v)).to(self.device)
  T,B=image.shape[:2]; N=T*B
  # GAE is evaluated in [B,T,1] order, then restored to rollout order.
  adv_bt,ret_bt=self.gae(
   reward.permute(1,0).unsqueeze(-1), done.permute(1,0).unsqueeze(-1),
   self.value_norm.denormalize(old_v.permute(1,0,2)),
   self.value_norm.denormalize(next_v.permute(1,0,2)))
  adv=adv_bt.permute(1,0,2); ret=ret_bt.permute(1,0,2)
  # ``unbiased=False`` keeps a one-transition smoke rollout finite.
  adv=(adv-adv.mean())/(adv.std(unbiased=False)+1e-8)
  self.value_norm.update(ret.reshape(-1,1)); ret_norm=self.value_norm.normalize(ret)
  flat_img=image.reshape(N,self.num_agents,*image.shape[3:]); flat_state=state.reshape(N,self.num_agents,-1)
  flat_action=action.reshape(N,self.num_agents,-1); flat_old_lp=old_lp.reshape(N,self.num_agents)
  flat_adv=adv.reshape(N,1).expand(N,self.num_agents); flat_ret=ret_norm.reshape(N,1)
  flat_old_v=old_v.reshape(N,1); flat_c=central.reshape(N,-1)
  logs=[]; last_sl=torch.tensor(0.,device=self.device); last_lg=torch.tensor(0.,device=self.device); self._last_lstm_update=False
  num_minibatches=max(int(self.cfg.num_minibatches),1); mb_size=max((N+num_minibatches-1)//num_minibatches,1)
  for _epoch in range(max(int(self.cfg.training_epoch_num),1)):
   order=torch.randperm(N,device=self.device)
   for start in range(0,N,mb_size):
    ids=order[start:start+mb_size]
    # Agent observations are flattened only after selecting the time/env batch.
    fi=flat_img[ids].reshape(-1,*image.shape[3:]); fs=flat_state[ids].reshape(-1,state.shape[-1]); fa=flat_action[ids].reshape(-1,action.shape[-1]); flp=flat_old_lp[ids].reshape(-1); fadv=flat_adv[ids].reshape(-1)
    dist=self._dist(fi,fs); lp=dist.log_prob(fa); ratio=torch.exp(lp-flp)
    policy_loss=-torch.min(fadv*ratio,fadv*ratio.clamp(1-self.clip_ratio,1+self.clip_ratio)).mean(); entropy=dist.entropy().mean()
    newv=self.critic(self.critic_encoder(flat_c[ids])); oldv_mb=flat_old_v[ids]
    oldv_raw=self.value_norm.denormalize(oldv_mb); newv_raw=self.value_norm.denormalize(newv)
    clipped_raw=oldv_raw+(newv_raw-oldv_raw).clamp(-self.value_clip_ratio,self.value_clip_ratio)
    value_loss=torch.max(F.huber_loss(newv,flat_ret[ids],delta=10.),F.huber_loss(self.value_norm.normalize(clipped_raw),flat_ret[ids],delta=10.))
    loss=policy_loss-self.entropy_coef*entropy+value_loss
    for opt in (self.encoder_opt,self.actor_opt,self.critic_encoder_opt,self.critic_opt): opt.zero_grad(set_to_none=True)
    loss.backward()
    eg=nn.utils.clip_grad_norm_(self.encoder.parameters(),self.cfg.max_grad_norm); ag=nn.utils.clip_grad_norm_(self.actor.parameters(),self.cfg.max_grad_norm); ceg=nn.utils.clip_grad_norm_(self.critic_encoder.parameters(),self.cfg.max_grad_norm); cg=nn.utils.clip_grad_norm_(self.critic.parameters(),self.cfg.max_grad_norm)
    self.encoder_opt.step(); self.actor_opt.step(); self.critic_encoder_opt.step(); self.critic_opt.step()
    sl=torch.tensor(0.,device=self.device); lg=torch.tensor(0.,device=self.device)
    if 'history' in rollout and 'future' in rollout:
     h=rollout['history'].to(self.device); f=rollout['future'].to(self.device); mask=rollout.get('history_mask',torch.ones(h.shape[:2],dtype=torch.bool,device=self.device)).to(self.device); dt=rollout.get('agent_dt',torch.full((h.shape[0],1),1/30,device=self.device)).to(self.device); dt=dt.unsqueeze(1) if dt.ndim==2 else dt; dt=dt.expand(-1,h.shape[1],-1); pred=self.lstm(torch.cat((h,dt),-1),mask=mask); sl=F.mse_loss(pred,f.reshape(f.shape[0],-1)); self.lstm_opt.zero_grad(set_to_none=True); sl.backward(); lg=nn.utils.clip_grad_norm_(self.lstm.parameters(),self.cfg.lstm_max_grad_norm); self.lstm_opt.step(); self._last_lstm_update=True; last_sl,last_lg=sl.detach(),lg.detach()
    logs.append((policy_loss.detach(),value_loss.detach(),entropy.detach(),eg,ag,ceg,cg))
  vals=torch.stack([x[0] for x in logs]); vls=torch.stack([x[1] for x in logs]); ents=torch.stack([x[2] for x in logs]); grads=[x[3:] for x in logs]
  return {'policy_loss':float(vals.mean()),'value_loss':float(vls.mean()),'entropy':float(ents.mean()),'sl_loss':float(last_sl),'encoder_grad_norm':float(torch.stack([x[0] for x in grads]).mean()),'actor_grad_norm':float(torch.stack([x[1] for x in grads]).mean()),'critic_grad_norm':float(torch.stack([x[3] for x in grads]).mean()),'lstm_grad_norm':float(last_lg)}
 def step_schedulers(self):
  self.encoder_scheduler.step(); self.actor_scheduler.step(); self.critic_encoder_scheduler.step(); self.critic_scheduler.step()
  if getattr(self,'_last_lstm_update',False): self.lstm_scheduler.step()
 def save_state(self,path):
  Path(path).parent.mkdir(parents=True,exist_ok=True); torch.save({'encoder':self.encoder.state_dict(),'actor':self.actor.state_dict(),'critic_encoder':self.critic_encoder.state_dict(),'critic':self.critic.state_dict(),'lstm':self.lstm.state_dict(),'encoder_opt':self.encoder_opt.state_dict(),'actor_opt':self.actor_opt.state_dict(),'critic_encoder_opt':self.critic_encoder_opt.state_dict(),'critic_opt':self.critic_opt.state_dict(),'lstm_opt':self.lstm_opt.state_dict(),'encoder_scheduler':self.encoder_scheduler.state_dict(),'actor_scheduler':self.actor_scheduler.state_dict(),'critic_encoder_scheduler':self.critic_encoder_scheduler.state_dict(),'critic_scheduler':self.critic_scheduler.state_dict(),'lstm_scheduler':self.lstm_scheduler.state_dict(),'value_norm':self.value_norm.state_dict()},path)
 def load_model_weights(self,path,load_optimizers=False):
  s=torch.load(path,map_location=self.device); self.encoder.load_state_dict(s['encoder']);self.actor.load_state_dict(s['actor']);self.critic_encoder.load_state_dict(s['critic_encoder']);self.critic.load_state_dict(s['critic']);self.lstm.load_state_dict(s['lstm']);self.value_norm.load_state_dict(s['value_norm']);
  if load_optimizers:
   for k,o in [('encoder_opt',self.encoder_opt),('actor_opt',self.actor_opt),('critic_encoder_opt',self.critic_encoder_opt),('critic_opt',self.critic_opt),('lstm_opt',self.lstm_opt)]: o.load_state_dict(s[k])
   for k,o in [('encoder_scheduler',self.encoder_scheduler),('actor_scheduler',self.actor_scheduler),('critic_encoder_scheduler',self.critic_encoder_scheduler),('critic_scheduler',self.critic_scheduler),('lstm_scheduler',self.lstm_scheduler)]:
    if k in s: o.load_state_dict(s[k])

# Explicit alias used by code written against the upstream name.
MAPPO=PSTOMAPPO
