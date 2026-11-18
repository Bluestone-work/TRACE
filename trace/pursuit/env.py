"""Single-World PSTO pursuit environment using TRACE rotor dynamics."""
import numpy as np, torch
import torch.nn.functional as F
from .scene import PSTOPursuitScene
from .lidar import PSTOLidarAdapter
from .observations import world_to_body,heatmap
from .evader import evader_policy
from .visibility import target_visible
from .rewards import official_reward

def build_global_state_84(positions,quaternions,velocities,angular_velocities,prev_actions,downsampled_lidar):
 positions=np.asarray(positions,np.float32); quaternions=np.asarray(quaternions,np.float32); velocities=np.asarray(velocities,np.float32); angular_velocities=np.asarray(angular_velocities,np.float32); prev_actions=np.asarray(prev_actions,np.float32); down=np.asarray(downsampled_lidar,np.float32)
 pooled=[]
 for i in range(2): pooled.append(F.adaptive_max_pool1d(torch.as_tensor(down[i]).amax(0).reshape(1,1,-1),18).reshape(-1).numpy())
 out=np.concatenate([positions.reshape(-1),np.concatenate([quaternions,velocities,angular_velocities],-1).reshape(-1),prev_actions.reshape(-1),np.concatenate(pooled)]).astype(np.float32)
 assert out.shape==(84,),out.shape
 return out
class PSTOPursuitEnv:
 def __init__(self,scene=None,max_episode_steps=280,seed=4348,with_lidar=True):
  self.scene=scene or PSTOPursuitScene(); self.max_episode_steps=max_episode_steps; self.with_lidar=with_lidar; self.lidar_adapter=PSTOLidarAdapter(); self.rng=np.random.default_rng(seed); self.step_count=0; self.prev_actions=np.zeros((3,3),np.float32); self.observed_evader_history=[]; self.observed_evader_times=[]; self.ground_truth_evader_history=[]; self.policy_lstm=None; self.last_lidar=np.zeros((2,24,480),np.float32); self.last_lidar_down=np.zeros((2,6,120),np.float32); self.last_evader_lidar=np.zeros((24,480),np.float32)
 def set_policy_lstm(self,module): self.policy_lstm=module
 def reset(self): self.step_count=0; self.prev_actions.fill(0); self.observed_evader_history=[]; self.observed_evader_times=[]; self.ground_truth_evader_history=[]; self.scene.reset(); return self.observe()
 def _root(self):
  p=np.stack([v.position for v in self.scene.vehicles]); q=np.stack([v.orientation for v in self.scene.vehicles]); v=np.stack([v.velocity for v in self.scene.vehicles]); w=np.stack([v.angular_velocity for v in self.scene.vehicles]); return p,q,v,w
 def _central(self,p,q,v,w,down):
  return build_global_state_84(p,q,v,w,self.prev_actions,down)
 def _visibility(self,p,q,scans): return np.asarray([target_visible(p[i],q[i],p[2],scans[i],max_distance=15.) for i in range(2)],dtype=bool)
 def _history_tensors(self):
  """Pack only real target observations and retain their elapsed intervals."""
  n=min(len(self.observed_evader_history),10); h=np.zeros((10,3),np.float32); mask=np.zeros(10,dtype=bool); dt=np.zeros((10,1),np.float32)
  if n:
   h[:n]=np.asarray(self.observed_evader_history[-n:],np.float32); mask[:n]=True
   times=np.asarray(self.observed_evader_times[-n:],np.float32); intervals=np.empty(n,np.float32)
   if n>1: intervals[:-1]=np.diff(times)
   intervals[-1]=max(float(getattr(self,'step_count',times[-1]))-float(times[-1]),1.)
   dt[:n,0]=intervals/30.
  return torch.as_tensor(h),torch.as_tensor(mask),torch.as_tensor(dt)
 def observe(self):
  p,q,v,w=self._root(); scans=np.stack([drone._raycast_lidar() if self.with_lidar else np.zeros((24,480),np.float32) for drone in self.scene.vehicles[:2]]); ev_scan=self.scene.vehicles[2]._raycast_lidar() if self.with_lidar else np.zeros((24,480),np.float32); self.last_lidar=scans; self.last_evader_lidar=ev_scan; vis=self._visibility(p,q,scans)
  self.ground_truth_evader_history.append(p[2].copy()); self.ground_truth_evader_history=self.ground_truth_evader_history[-20:]
  if vis.any():
   self.observed_evader_history.append(p[2].copy()); self.observed_evader_times.append(float(self.step_count))
   self.observed_evader_history=self.observed_evader_history[-10:]; self.observed_evader_times=self.observed_evader_times[-10:]
  h,mask,h_dt=self._history_tensors()
  if self.policy_lstm is not None and mask.any():
   lstm_device=next(self.policy_lstm.parameters()).device
   with torch.no_grad(): pred=self.policy_lstm(torch.cat((h[None].to(lstm_device),h_dt[None].to(lstm_device)), -1),mask=mask[None].to(lstm_device)).view(1,10,3)[0].cpu().numpy()
  else: pred=np.repeat(h[-1].numpy()[None],10,axis=0)
  down=self.lidar_adapter.downsample(torch.as_tensor(scans)); self.last_lidar_down=down.numpy(); obs_states=[]; heat=[]
  for i in range(2):
   body_v=world_to_body(torch.as_tensor(v[i]),torch.as_tensor(q[i])).numpy(); obs_states.append(np.concatenate([q[i],body_v,w[i],self.prev_actions[i]]))
   teammate=1-i; team_future=(p[teammate][None]+v[teammate][None]*np.arange(1,11,dtype=np.float32)[:,None]/30.)[None]
   hm=heatmap(torch.as_tensor(p[i:i+1]),torch.as_tensor(q[i:i+1]),torch.as_tensor(p[2:3]),torch.as_tensor(pred[None]),torch.as_tensor(p[teammate:teammate+1]),torch.as_tensor(team_future),visibility=torch.as_tensor([vis[i]])); heat.append(hm[0].numpy())
  fused=np.stack([np.stack([down[i].numpy(),heat[i]]) for i in range(2)]).astype(np.float32); central=self._central(p,q,v,w,down)
  assert fused.shape==(2,2,6,120) and np.asarray(obs_states).shape==(2,13)
  # Supervised data is delayed by ten policy steps: the LSTM sees the older
  # ten ground-truth samples and predicts the newer ten.  It is kept in a
  # separate key and never enters the actor observation above.
  if len(self.ground_truth_evader_history)>=20:
   sl_h=np.asarray(self.ground_truth_evader_history[-20:-10],np.float32); sl_f=np.asarray(self.ground_truth_evader_history[-10:],np.float32); sl_m=np.ones(10,dtype=bool)
  else:
   sl_h=np.zeros((10,3),np.float32); sl_f=np.zeros((10,3),np.float32); sl_m=np.zeros(10,dtype=bool)
  sl_dt=np.full((10,1),1/30.,np.float32)
  if len(self.ground_truth_evader_history)>=10:
   tf_h=np.asarray(self.ground_truth_evader_history[-10:],np.float32); tf_m=np.ones(10,dtype=bool)
  else:
   tf_h=np.zeros((10,3),np.float32); tf_m=np.zeros(10,dtype=bool)
  return {'fused_image':fused,'state':np.asarray(obs_states,np.float32),'central_state':central,'visibility':vis.astype(np.float32),'history':h.numpy(),'history_mask':mask.numpy(),'history_dt':h_dt.numpy(),'future_ground_truth':sl_f,'lstm_history':sl_h,'lstm_history_mask':sl_m,'lstm_history_dt':sl_dt,'lstm_target_available':bool(sl_m.any()),'teacher_history':tf_h,'teacher_history_mask':tf_m,'teacher_history_dt':sl_dt}
 def step(self,actions,actions_are_normalized=True):
  actions=np.asarray(actions,np.float32)
  if actions_are_normalized:
   if actions.shape!=(2,2): raise ValueError(f'PSTO actor action must be [2,2], got {actions.shape}')
   from .algorithm.mappo import PSTOMAPPO
   p,q,v,w=self._root(); body=PSTOMAPPO.action_to_body_velocity(torch.as_tensor(actions)).numpy(); world=PSTOMAPPO.action_to_velocity(torch.as_tensor(actions),torch.as_tensor(q[:2])).numpy()
  else:
   if actions.shape!=(2,3): raise ValueError('explicit 3-D commands require actions_are_normalized=False')
   world=actions; body=np.stack([world_to_body(torch.as_tensor(world[i]),torch.as_tensor(q[i])).numpy() for i in range(2)])
  p,q,v,w=self._root(); evader_hits=self.scene.vehicles[2]._raycast_lidar_hit_points() if self.with_lidar else None; ev_target=evader_policy(torch.as_tensor(p[:2]),torch.as_tensor(p[2]),obstacle_hits=torch.as_tensor(evader_hits) if evader_hits is not None else None).numpy(); self.prev_actions[:2]=body; self.prev_actions[2]=world_to_body(torch.as_tensor(ev_target),torch.as_tensor(q[2])).numpy(); self.scene.step(np.vstack([world,ev_target])); self.step_count+=1; p,q,v,w=self._root(); obs=self.observe(); prox=torch.as_tensor(self.last_lidar_down); ev_collision=bool(np.max(self.last_evader_lidar)>.0 and np.max(self.last_evader_lidar)>10.-.32); reward,captured,escaped,done,terms=official_reward(torch.as_tensor(p[:2]),torch.as_tensor(v[:2]),torch.as_tensor(p[2]),prox,step=self.step_count,evader_vel=torch.as_tensor(v[2]),evader_collision=torch.tensor(ev_collision),below_bound_altitude=.5); timeout=self.step_count>=self.max_episode_steps; done=bool(done.item() or timeout); return obs,float(reward.item()),done,{'capture':bool(captured.item()),'escaped':bool(escaped.item()),'collision':bool(terms['pursuer_collision'].item() or ev_collision),'timeout':timeout,'distance':float(torch.linalg.vector_norm(torch.as_tensor(p[:2])-torch.as_tensor(p[2]),dim=-1).min()),'visibility_ratio':float(np.mean(obs['visibility'])),'reward_terms':{k:float(v.item()) for k,v in terms.items() if torch.is_tensor(v) and v.numel()==1}}
 def close(self): self.scene.close()
