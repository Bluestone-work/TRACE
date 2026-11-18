"""PSTO reward terms and termination conditions."""
import torch

def official_reward(pursuer_pos,pursuer_vel,evader_pos,obstacle_clearance=None,step=0,capture_threshold=1.,capture_bonus=1800.,pursuer_collision_distance=.35,evader_collision_distance=.32,pursuer_collision_penalty=-900.,evader_collision_bonus=100.,escape_min_distance=7.5,escape_max_distance=10.,escape_penalty=-700.,approach_reward_min=-3.,coordination_optimal_distance=4.,coordination_sigma=2.,time_penalty_start_step=200,time_penalty_value=-.05,pursuit_weight=1.1,pursuit_offset=.1,coordination_weight=.2,obstacle_weight=.5,formation_weight=1.3):
 d=torch.linalg.vector_norm(pursuer_pos-evader_pos.unsqueeze(-2),dim=-1); nearest=d.min(-1).values; capture=nearest<capture_threshold; escaped=(nearest>escape_min_distance)&(nearest<escape_max_distance)
 radial=(pursuer_pos-evader_pos.unsqueeze(-2))/(d.unsqueeze(-1).clamp_min(1e-6)); approach=(pursuer_vel*radial).sum(-1); ap=torch.clamp(-approach,min=approach_reward_min); pursuit=ap.mean(-1)+ap.min(-1).values
 coordination=torch.exp(-((d[:,0]-coordination_optimal_distance)**2)/(2*coordination_sigma**2)) if d.shape[1]>1 else torch.zeros_like(nearest)
 obstacle=torch.zeros_like(nearest) if obstacle_clearance is None else torch.log(torch.clamp(obstacle_clearance,min=1e-6)).mean(-1)
 formation=torch.zeros_like(nearest)
 reward=pursuit_weight*(pursuit+pursuit_offset)+coordination_weight*coordination+capture.float()*capture_bonus+obstacle_weight*obstacle+formation_weight*formation
 reward=torch.where(step>=time_penalty_start_step,reward+time_penalty_value,reward); reward=torch.where(escaped,reward+escape_penalty,reward); return reward,capture,escaped

def pursuit_reward(pursuer_pos,evader_pos,step,threshold=1.,capture_bonus=1800.,escape_min=7.5,escape_max=10.,escape_penalty=-700.):
 vel=torch.zeros_like(pursuer_pos); return official_reward(pursuer_pos,vel,evader_pos,step=step,capture_threshold=threshold,capture_bonus=capture_bonus,escape_min_distance=escape_min,escape_max_distance=escape_max,escape_penalty=escape_penalty)
