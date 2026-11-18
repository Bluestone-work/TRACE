"""Numerically faithful PSTO reward and termination computation."""
import math
import torch

def official_reward(pursuer_pos,pursuer_vel,evader_pos,obstacle_proximity=None,step=0,evader_vel=None,lidar_range=10.,capture_threshold=1.,capture_bonus=1800.,pursuer_collision_distance=.35,evader_collision_distance=.32,pursuer_collision_penalty=-900.,evader_collision_bonus=100.,escape_min_distance=7.5,escape_max_distance=10.,escape_penalty=-700.,below_bound_altitude=.5,approach_reward_min=-3.,coordination_optimal_distance=4.,coordination_sigma=2.,obstacle_penalty_sensitive_range=2.5,obstacle_penalty_log_power=5.,obstacle_penalty_log_offset=1.6,obstacle_penalty_quantile=.03,time_penalty_start_step=200,time_penalty_value=-.05,pursuit_weight=1.1,pursuit_offset=.1,coordination_weight=.2,obstacle_weight=.5,formation_weight=1.3,pursuer_collision=None,evader_collision=None):
 squeeze=pursuer_pos.ndim==2
 if squeeze:pursuer_pos=pursuer_pos.unsqueeze(0); pursuer_vel=pursuer_vel.unsqueeze(0); evader_pos=evader_pos.unsqueeze(0)
 B,P,_=pursuer_pos.shape; evader_vel=torch.zeros_like(evader_pos) if evader_vel is None else (evader_vel.unsqueeze(0) if evader_vel.ndim==1 else evader_vel)
 rpos=evader_pos.unsqueeze(1)-pursuer_pos; dist=torch.linalg.vector_norm(rpos,dim=-1); min_dist=dist.min(-1).values; max_dist=dist.max(-1).values
 captured=(dist<capture_threshold).any(-1); escaped=(min_dist>escape_min_distance)|(max_dist>escape_max_distance)
 rel_vel=evader_vel.unsqueeze(1)-pursuer_vel; unit=rpos/(torch.linalg.vector_norm(rpos,dim=-1,keepdim=True)+1e-6); approach=(rel_vel*unit).sum(-1); approach_reward=torch.clamp(-approach,min=approach_reward_min); pursuit=approach_reward.mean(-1)+approach_reward.min(-1).values
 pair_dist=torch.linalg.vector_norm(pursuer_pos[:,0]-pursuer_pos[:,1],dim=-1) if P>1 else torch.zeros(B,device=pursuer_pos.device); coordination=torch.exp(-((pair_dist-coordination_optimal_distance)**2)/(2*coordination_sigma**2))
 if obstacle_proximity is None: obstacle_penalty=torch.zeros(B,device=pursuer_pos.device); prox=torch.zeros(B,P,1,device=pursuer_pos.device)
 else:
  prox=obstacle_proximity if obstacle_proximity.ndim==4 else obstacle_proximity.unsqueeze(0); actual=(lidar_range-prox).clamp(min=0.); sensitive=max(obstacle_penalty_sensitive_range,1e-9); scaled=(actual.clamp(max=obstacle_penalty_sensitive_range)/sensitive).clamp(min=1e-6); logs=torch.log(scaled**obstacle_penalty_log_power); logs=torch.where(torch.isposinf(logs),torch.zeros_like(logs),logs); logs=torch.where(torch.isneginf(logs),torch.full_like(logs,-100.),logs); q=torch.quantile(logs.reshape(B,P,-1),obstacle_penalty_quantile,dim=-1); obstacle_penalty=(q+obstacle_penalty_log_offset).mean(-1)
 if pursuer_collision is None: pursuer_collision=(prox.reshape(B,-1).max(-1).values>lidar_range-pursuer_collision_distance)
 else:pursuer_collision=pursuer_collision if pursuer_collision.ndim==1 else pursuer_collision.any(-1)
 evader_collision=torch.zeros(B,dtype=torch.bool,device=pursuer_pos.device) if evader_collision is None else evader_collision.bool(); evader_collision=evader_collision.reshape(-1).expand(B) if evader_collision.ndim==0 or evader_collision.numel()==1 else evader_collision
 closest=dist.argmin(-1); closest_unit=torch.gather(unit,1,closest[:,None,None].expand(-1,1,3)).squeeze(1); angles=torch.acos(torch.clamp((closest_unit[:,None]*unit).sum(-1),-1.,1.)); formation=angles.mean(-1)/math.pi*2.-.5 if P>1 else torch.zeros(B,device=pursuer_pos.device)
 below=(torch.cat((pursuer_pos[...,2],evader_pos[...,2:3]),-1)<below_bound_altitude).any(-1); time=torch.where(torch.as_tensor(step,device=pursuer_pos.device)>time_penalty_start_step,torch.as_tensor(time_penalty_value,device=pursuer_pos.device),torch.zeros(B,device=pursuer_pos.device)); reward=pursuit_weight*(pursuit+pursuit_offset)+coordination_weight*coordination+captured.float()*capture_bonus+obstacle_weight*obstacle_penalty+formation_weight*formation+time; reward[pursuer_collision]+=pursuer_collision_penalty; reward[evader_collision]+=evader_collision_bonus; reward[escaped]+=escape_penalty; done=captured|pursuer_collision|evader_collision|below|escaped
 if squeeze:return reward[0],captured[0],escaped[0],done[0],{'pursuit':pursuit[0],'coordination':coordination[0],'obstacle':obstacle_penalty[0],'formation':formation[0],'pursuer_collision':pursuer_collision[0],'evader_collision':evader_collision[0],'below_bound':below[0]}
 return reward,captured,escaped,done,{'pursuit':pursuit,'coordination':coordination,'obstacle':obstacle_penalty,'formation':formation,'pursuer_collision':pursuer_collision,'evader_collision':evader_collision,'below_bound':below}

def pursuit_reward(*args,**kwargs):
 r,c,e,d,_=official_reward(*args,**kwargs); return r,c,e
