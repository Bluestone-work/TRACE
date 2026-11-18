"""Pure Torch coordinate transforms and PSTO intent heatmap."""
import torch

def world_to_body(v,q):
 q=q/torch.linalg.vector_norm(q,dim=-1,keepdim=True).clamp_min(1e-9); w,x,y,z=q.unbind(-1)
 qc=torch.stack((w,-x,-y,-z),-1); t=2*torch.cross(qc[...,1:],v,dim=-1)
 return v+qc[...,0:1]*t+torch.cross(qc[...,1:],t,dim=-1)

def heatmap(observer_pos,observer_quat,evader_pos,evader_future=None,teammate_pos=None,teammate_future=None,shape=(6,120),max_range=10.):
 out=torch.zeros((*observer_pos.shape[:-1],*shape),device=observer_pos.device,dtype=observer_pos.dtype)
 q=observer_quat
 def add(points,value,decay):
  nonlocal out
  if points is None:return
  if points.ndim==observer_pos.ndim: points=points.unsqueeze(-2)
  rel=world_to_body(points-observer_pos.unsqueeze(-2),q.unsqueeze(-2).expand(*q.shape[:-1],points.shape[-2],4))
  dist=torch.linalg.vector_norm(rel,dim=-1); az=torch.atan2(rel[...,1],rel[...,0]); el=torch.atan2(rel[...,2],torch.linalg.vector_norm(rel[...,:2],dim=-1).clamp_min(1e-6))
  x=((az/(2*torch.pi)+.5)*shape[-1]).long().remainder(shape[-1]); y=((el/torch.pi+.5)*shape[-2]).long().clamp(0,shape[-2]-1)
  for i in range(points.shape[-2]):
   val=value*(decay**i) if points.shape[-2]>1 else value
   valid=dist[...,i]<=max_range
   # scatter-add one point per batch element; this is intentionally deterministic.
   flat=out.reshape(-1,shape[-2]*shape[-1]); idx=(y[...,i]*shape[-1]+x[...,i]).reshape(-1); vv=torch.where(valid.reshape(-1),torch.as_tensor(val,device=out.device,dtype=out.dtype),torch.zeros_like(idx,dtype=out.dtype)); flat.scatter_add_(1,idx[:,None],vv[:,None]); out=flat.reshape_as(out)
 add(evader_pos,2.,.95); add(evader_future,1.5,.95); add(teammate_pos,-2.,.9); add(teammate_future,-1.5,.9); return out
