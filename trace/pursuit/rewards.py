import torch
def pursuit_reward(pursuer_pos,evader_pos,step,threshold=1.,capture_bonus=1800.,escape_min=7.5,escape_max=10.,escape_penalty=-700.):
    d=torch.linalg.vector_norm(pursuer_pos-evader_pos.unsqueeze(-2),dim=-1); nearest=d.min(-1).values
    capture=nearest<threshold; escaped=(nearest>escape_min)&(nearest<escape_max)
    reward=-nearest + capture.float()*capture_bonus + escaped.float()*escape_penalty
    return reward,capture,escaped
