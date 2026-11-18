import torch

def beta_action_to_velocity(action, quaternion=None, max_speed=.8):
    """Official PSTO Beta action [direction, magnitude] -> world velocity."""
    a=torch.as_tensor(action)
    angle=torch.pi*(1-2*a[...,0]); length=.9+.1*a[...,1]
    body=torch.stack((length*torch.cos(angle)*max_speed,length*torch.sin(angle)*max_speed,torch.zeros_like(length)),dim=-1)
    if quaternion is None: return body
    q=quaternion/torch.linalg.vector_norm(quaternion,dim=-1,keepdim=True).clamp_min(1e-9)
    w,x,y,z=q.unbind(-1); c=torch.cos(torch.atan2(2*(w*z+x*y),1-2*(y*y+z*z))); s=torch.sin(torch.atan2(2*(w*z+x*y),1-2*(y*y+z*z)))
    return torch.stack((c*body[...,0]-s*body[...,1],s*body[...,0]+c*body[...,1],body[...,2]),-1)
