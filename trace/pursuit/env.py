"""Single-World PSTO pursuit environment using TRACE rotor dynamics."""
import numpy as np, torch
import torch.nn.functional as F
from .scene import PSTOPursuitScene
from .lidar import PSTOLidarAdapter
from .observations import world_to_body,heatmap
from .evader import evader_policy
from .visibility import target_visible
from .rewards import official_reward

def _portable_rng_state(rng):
 def convert(value):
  if isinstance(value, dict): return {k: convert(v) for k, v in value.items()}
  if isinstance(value, np.ndarray): return value.tolist()
  if isinstance(value, np.generic): return value.item()
  return value
 return convert(rng.bit_generator.state)

def build_global_state_84(positions,quaternions,velocities,angular_velocities,prev_actions,downsampled_lidar):
 positions=np.asarray(positions,np.float32); quaternions=np.asarray(quaternions,np.float32); velocities=np.asarray(velocities,np.float32); angular_velocities=np.asarray(angular_velocities,np.float32); prev_actions=np.asarray(prev_actions,np.float32); down=np.asarray(downsampled_lidar,np.float32)
 pooled=[]
 for i in range(2): pooled.append(F.adaptive_max_pool1d(torch.as_tensor(down[i]).amax(0).reshape(1,1,-1),18).reshape(-1).numpy())
 out=np.concatenate([positions.reshape(-1),np.concatenate([quaternions,velocities,angular_velocities],-1).reshape(-1),prev_actions.reshape(-1),np.concatenate(pooled)]).astype(np.float32)
 assert out.shape==(84,),out.shape
 return out
class PSTOPursuitEnv:
 def __init__(self,scene=None,max_episode_steps=280,seed=4348,with_lidar=True,physics_dt=1/240,policy_hz=30.,num_obstacles=18,randomize_resets=False,randomize_obstacles=False):
  self.seed=int(seed); self.randomize_resets=bool(randomize_resets); self.randomize_obstacles=bool(randomize_obstacles); self.scene=scene or PSTOPursuitScene(physics_dt=physics_dt,policy_hz=policy_hz,num_obstacles=num_obstacles,seed=self.seed,randomize_resets=self.randomize_resets,randomize_obstacles=self.randomize_obstacles); self.max_episode_steps=max_episode_steps; self.with_lidar=with_lidar; self.lidar_adapter=PSTOLidarAdapter(); self.rng=np.random.default_rng(self.seed); self.step_count=0; self.episode_index=0; self.prev_actions=np.zeros((3,3),np.float32); self.observed_evader_history=[]; self.observed_evader_times=[]; self.ground_truth_evader_history=[]; self.policy_lstm=None; self.last_lidar=np.zeros((2,24,480),np.float32); self.last_lidar_down=np.zeros((2,6,120),np.float32); self.last_evader_lidar=np.zeros((24,480),np.float32)
 def set_policy_lstm(self,module): self.policy_lstm=module
 def reset(self,seed=None):
  self.step_count=0; self.episode_index+=1; self.prev_actions.fill(0); self.observed_evader_history=[]; self.observed_evader_times=[]; self.ground_truth_evader_history=[]
  if seed is not None:
   self.seed=int(seed); self.rng=np.random.default_rng(self.seed)
  self.scene.reset(randomize=self.randomize_resets,seed=seed,randomize_obstacles=self.randomize_obstacles); return self.observe()
 def _root(self):
  p=np.stack([v.position for v in self.scene.vehicles]); q=np.stack([v.orientation for v in self.scene.vehicles]); v=np.stack([v.velocity for v in self.scene.vehicles]); w=np.stack([v.angular_velocity for v in self.scene.vehicles]); return p,q,v,w
 def _central(self,p,q,v,w,down):
  return build_global_state_84(p,q,v,w,self.prev_actions,down)
 def _collision_flags(self,p):
  """Classify geometric contacts for diagnostics and reward termination.

  The LiDAR proximity threshold alone cannot distinguish a pursuer-pursuer hit
  from a vehicle-obstacle hit.  Use the PhysX poses and the known obstacle AABBs
  as a deterministic contact fallback, while retaining the raw collision classes
  in ``step`` info for auditing.
  """
  p=np.asarray(p,dtype=np.float32)
  pursuer_pair=bool(np.linalg.norm(p[0]-p[1])<.35)
  pursuer_evader=bool(np.min(np.linalg.norm(p[:2]-p[2],axis=-1))<.32)
  obstacle_hits=np.zeros(3,dtype=bool)
  for center,scale in zip(getattr(self.scene,'obstacle_positions',[]),getattr(self.scene,'obstacle_scales',[])):
   half=np.asarray(scale,dtype=np.float32)/2.; delta=np.maximum(np.abs(p-np.asarray(center,dtype=np.float32))-half[None,:],0.)
   obstacle_hits |= np.linalg.norm(delta,axis=-1)<.32
  pursuer_obstacle=bool(obstacle_hits[:2].any()); evader_obstacle=bool(obstacle_hits[2])
  return {'pursuer_pursuer':pursuer_pair,'pursuer_evader':pursuer_evader,
          'pursuer_obstacle':pursuer_obstacle,'evader_obstacle':evader_obstacle,
          'pursuer_collision':pursuer_pair or pursuer_obstacle,
          'evader_collision':pursuer_evader or evader_obstacle,
          'obstacle_collision':pursuer_obstacle or evader_obstacle}
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
  p,q,v,w=self._root(); evader_hits=self.scene.vehicles[2]._raycast_lidar_hit_points() if self.with_lidar else None; ev_target=evader_policy(torch.as_tensor(p[:2]),torch.as_tensor(p[2]),obstacle_hits=torch.as_tensor(evader_hits) if evader_hits is not None else None).numpy(); self.prev_actions[:2]=body; self.prev_actions[2]=world_to_body(torch.as_tensor(ev_target),torch.as_tensor(q[2])).numpy(); self.scene.step(np.vstack([world,ev_target])); self.step_count+=1; p,q,v,w=self._root(); obs=self.observe(); prox=torch.as_tensor(self.last_lidar_down); collisions=self._collision_flags(p); reward,captured,escaped,terminated_tensor,terms=official_reward(torch.as_tensor(p[:2]),torch.as_tensor(v[:2]),torch.as_tensor(p[2]),prox,step=self.step_count,evader_vel=torch.as_tensor(v[2]),pursuer_collision=torch.tensor(collisions['pursuer_collision']),evader_collision=torch.tensor(collisions['evader_collision']),below_bound_altitude=.5); terminated=bool(terminated_tensor.item()); timeout=self.step_count>=self.max_episode_steps; truncated=bool(timeout and not terminated); episode_end=bool(terminated or truncated); return obs,float(reward.item()),episode_end,{'capture':bool(captured.item()),'escaped':bool(escaped.item()),'collision':bool(collisions['pursuer_collision'] or collisions['evader_collision']),'pursuer_collision':collisions['pursuer_collision'],'pursuer_pursuer_collision':collisions['pursuer_pursuer'],'pursuer_evader_collision':collisions['pursuer_evader'],'pursuer_obstacle_collision':collisions['pursuer_obstacle'],'evader_collision':collisions['evader_collision'],'evader_obstacle_collision':collisions['evader_obstacle'],'obstacle_collision':collisions['obstacle_collision'],'timeout':timeout,'terminated':terminated,'truncated':truncated,'episode_end':episode_end,'distance':float(torch.linalg.vector_norm(torch.as_tensor(p[:2])-torch.as_tensor(p[2]),dim=-1).min()),'visibility_ratio':float(np.mean(obs['visibility'])),'reward_terms':{k:float(v.item()) for k,v in terms.items() if torch.is_tensor(v) and v.numel()==1}}
 def close(self): self.scene.close()
 def state_dict(self):
  p,q,v,w=self._root()
  return {'step_count':int(self.step_count),'episode_index':int(self.episode_index),
   'seed':int(self.seed),'prev_actions':self.prev_actions.copy(),
   'positions':p.copy(),'quaternions':q.copy(),'velocities':v.copy(),'angular_velocities':w.copy(),
   'rotor_speeds':[np.asarray(x.actuator.rotor_speeds).copy() for x in self.scene.vehicles],
   'observed_evader_history':[x.copy() for x in self.observed_evader_history],
   'observed_evader_times':list(self.observed_evader_times),
   'ground_truth_evader_history':[x.copy() for x in self.ground_truth_evader_history], 'rng_state':_portable_rng_state(self.rng),
   'obstacle_positions':np.asarray(getattr(self.scene,'obstacle_positions',[]),np.float32).copy(),
   'scene_state':self.scene.state_dict() if hasattr(self.scene,'state_dict') else None}
 def load_state_dict(self,state):
  positions=np.asarray(state['positions'],np.float32); self.scene.reset(positions=positions,randomize=False,randomize_obstacles=False)
  if state.get('obstacle_positions') is not None:
   self.scene.obstacle_positions=np.asarray(state['obstacle_positions'],np.float32).copy()
   for obstacle,op in zip(self.scene.obstacles,self.scene.obstacle_positions): obstacle.set_world_pose(position=op)
  for vehicle,q,v,w,speeds in zip(self.scene.vehicles,state['quaternions'],state['velocities'],state['angular_velocities'],state['rotor_speeds']):
   vehicle.drone.set_world_pose(position=vehicle.position,orientation=np.asarray(q,np.float32)); vehicle.drone.set_linear_velocity(np.asarray(v,np.float32)); vehicle.drone.set_angular_velocity(np.asarray(w,np.float32)); vehicle.drone.set_joint_velocities(np.asarray(speeds,np.float32),joint_indices=vehicle.rotor_joint_indices)
   throttle=np.asarray(speeds,np.float64)/(vehicle.actuator.omega_max*vehicle.actuator.directions)
   vehicle.actuator.throttle=np.clip(throttle,-1.,1.); vehicle._read_physics_state()
  self.step_count=int(state['step_count']); self.episode_index=int(state.get('episode_index',0)); self.seed=int(state.get('seed',self.seed)); self.prev_actions=np.asarray(state['prev_actions'],np.float32).copy(); self.observed_evader_history=[np.asarray(x,np.float32).copy() for x in state.get('observed_evader_history',[])]; self.observed_evader_times=list(state.get('observed_evader_times',[])); self.ground_truth_evader_history=[np.asarray(x,np.float32).copy() for x in state.get('ground_truth_evader_history',[])];
  if state.get('rng_state') is not None: self.rng.bit_generator.state=state['rng_state']
  if state.get('scene_state') is not None and hasattr(self.scene,'load_state_dict'):
   self.scene.load_state_dict(state['scene_state'])
