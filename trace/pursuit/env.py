"""Single-World PSTO pursuit environment using TRACE rotor dynamics."""
import numpy as np, torch
from .scene import PSTOPursuitScene
from .lidar import PSTOLidarAdapter
from .observations import world_to_body,heatmap
from .evader import evader_policy
from .rewards import official_reward
class PSTOPursuitEnv:
 def __init__(self,scene=None,max_episode_steps=280,seed=4348,with_lidar=True):
  self.scene=scene or PSTOPursuitScene(); self.max_episode_steps=max_episode_steps; self.with_lidar=with_lidar; self.lidar_adapter=PSTOLidarAdapter(); self.rng=np.random.default_rng(seed); self.step_count=0; self.prev_actions=np.zeros((2,3),np.float32); self.evader_history=[]; self.policy_lstm=None
 def set_policy_lstm(self,module): self.policy_lstm=module
 def reset(self): self.step_count=0; self.prev_actions.fill(0); self.evader_history=[]; self.scene.reset(); return self.observe()
 def _root(self):
  p=np.stack([v.position for v in self.scene.vehicles]); q=np.stack([v.orientation for v in self.scene.vehicles]); v=np.stack([v.velocity for v in self.scene.vehicles]); w=np.stack([v.angular_velocity for v in self.scene.vehicles]); return p,q,v,w
 def _central(self,p,q,v,w):
  # Exact 84-D layout: positions(9), q+v+w(30), velocities(9), two 9-D pursuer states.
  extras=np.concatenate([q,v,w],-1).reshape(-1); pursuer=np.concatenate([q[:2],v[:2],w[:2]],-1).reshape(-1); return np.concatenate([p.reshape(-1),extras,v.reshape(-1),pursuer]).astype(np.float32)
 def _visibility(self,p):
  d=np.linalg.norm(p[:2]-p[2],axis=-1); return d<15.
 def observe(self):
  p,q,v,w=self._root(); vis=self._visibility(p); self.evader_history.append(p[2].copy()); self.evader_history=self.evader_history[-10:]; hist=torch.as_tensor(np.asarray(self.evader_history),dtype=torch.float32)
  if self.policy_lstm is not None and len(self.evader_history)>0:
   h=hist[None]; mask=torch.ones(1,h.shape[1],dtype=torch.bool); dt=torch.full((1,h.shape[1],1),1/30.); pred=self.policy_lstm(torch.cat((h,dt),-1),mask=mask).view(1,10,3)[0].detach().numpy()
  else: pred=np.repeat(p[2][None],10,axis=0)
  scans=[]
  for drone in self.scene.vehicles[:2]: scans.append(drone._raycast_lidar() if self.with_lidar else np.zeros((24,480),np.float32))
  scan=torch.as_tensor(np.stack(scans),dtype=torch.float32); down=self.lidar_adapter.downsample(scan); obs_states=[]; heat=[]
  for i in range(2):
   body_v=world_to_body(torch.as_tensor(v[i]),torch.as_tensor(q[i])).numpy(); obs_states.append(np.concatenate([q[i],body_v,w[i],self.prev_actions[i]])); hm=heatmap(torch.as_tensor(p[i:i+1]),torch.as_tensor(q[i:i+1]),torch.as_tensor(p[2:3]),torch.as_tensor(pred[None]),torch.as_tensor(p[1-i:i+2-i]),torch.as_tensor(np.repeat(p[1-i:i+2-i],10,axis=0)[None])); heat.append(hm[0].numpy() if hm.ndim==3 else hm.numpy())
  fused=np.stack([np.stack([down[i].numpy(),heat[i]]) for i in range(2)])
  return {'fused_image':fused.astype(np.float32),'state':np.asarray(obs_states,np.float32),'central_state':self._central(p,q,v,w),'visibility':vis.astype(np.float32),'history':np.asarray(self.evader_history,np.float32)}
 def step(self,actions):
  actions=np.asarray(actions,np.float32).reshape(2,3); self.prev_actions=actions.copy(); p,q,v,w=self._root(); pursuer=torch.as_tensor(p[:2]); ev=torch.as_tensor(p[2]); target=evader_policy(pursuer,ev).numpy(); self.scene.step(np.vstack([actions,target])); self.step_count+=1; p,q,v,w=self._root(); d=torch.linalg.vector_norm(torch.as_tensor(p[:2])-torch.as_tensor(p[2]),dim=-1); clearance=np.asarray([self.scene.vehicles[i]._raycast_lidar().min() for i in range(2)],np.float32); r,c,e=official_reward(torch.as_tensor(p[:2]),torch.as_tensor(v[:2]),torch.as_tensor(p[2]),torch.as_tensor(clearance),self.step_count); collision=bool(np.min(d).item()<.35); done=bool(c.item() or e.item() or collision or self.step_count>=self.max_episode_steps); return self.observe(),float(r.item()),done,{'capture':bool(c.item()),'escaped':bool(e.item()),'collision':collision,'distance':float(d.min()),'visibility_ratio':float(self._visibility(p).mean())}
 def close(self): self.scene.close()
