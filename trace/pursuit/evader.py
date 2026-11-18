"""Official PSTO scripted evader velocity policy (pure Torch)."""
import torch

def normalize_safe(v):
 n=torch.linalg.vector_norm(v,dim=-1,keepdim=True); return torch.where(n>1e-6,v/(n+1e-6),torch.zeros_like(v))

def evader_policy(pursuer_pos,evader_pos,obstacle_hits=None,arena_centers=None,arena_radius=9.,pursuer_force_multiplier=18.,max_speed=1.7,near_boundary_threshold=2.2,obstacle_awareness_distance=2.,obstacle_repulsion_weight=.1,inward_bias_strength=.1,target_altitude=2.,altitude_gain=2.):
 """Matches env.py::_evader_policy_vec; obstacle_hits is [B,N,3] world hit points."""
 if evader_pos.ndim==3: evader_pos=evader_pos.squeeze(1)
 B=evader_pos.shape[0]; device=evader_pos.device
 if arena_centers is None: arena_centers=torch.zeros_like(evader_pos)
 vec_center=arena_centers-evader_pos; vec_center[...,2]=0.; dist_wall=arena_radius-torch.linalg.vector_norm(vec_center[...,:2],dim=-1)
 flee=evader_pos.unsqueeze(1)-pursuer_pos; flee[...,2]=0.; dp=torch.linalg.vector_norm(flee,dim=-1); strength=torch.clamp(pursuer_force_multiplier/(dp.clamp(min=.1)**2+1e-6),max=50.); pursuer_force=(normalize_safe(flee)*strength.unsqueeze(-1)).sum(1)
 obstacle_force=torch.zeros_like(evader_pos)
 if obstacle_hits is not None:
  rel=obstacle_hits-evader_pos.unsqueeze(1); d=torch.linalg.vector_norm(rel,dim=-1); mask=(d<obstacle_awareness_distance)&torch.isfinite(d)
  obstacle_force=(-normalize_safe(rel)*(torch.where(mask,1./d.clamp_min(.01)**2,torch.zeros_like(d)).unsqueeze(-1))).sum(1)
 near=dist_wall<near_boundary_threshold; effective=torch.zeros_like(evader_pos); desired=pursuer_force+obstacle_force*obstacle_repulsion_weight; effective[~near]=desired[~near]
 if near.any():
  out=-normalize_safe(vec_center[near]); dprod=(desired[near]*out).sum(-1,keepdim=True); projected=desired[near]-torch.clamp(dprod,min=0)*out; effective[near]=normalize_safe(normalize_safe(projected)+(-out)*inward_bias_strength)
 speed=torch.linalg.vector_norm(effective,dim=-1,keepdim=True); final=torch.where(near.unsqueeze(-1),torch.full_like(speed,max_speed),speed.clamp(max=max_speed)); target=normalize_safe(effective)*final; target[...,2]+=(target_altitude-evader_pos[...,2])*altitude_gain; return target
