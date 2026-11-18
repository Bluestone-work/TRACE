"""PSTO observation transforms and the official native-beam heatmap projection."""
import torch

def world_to_body(v,q):
 q=q/torch.linalg.vector_norm(q,dim=-1,keepdim=True).clamp_min(1e-9); w,x,y,z=q.unbind(-1); qc=torch.stack((w,-x,-y,-z),-1); t=2*torch.cross(qc[...,1:],v,dim=-1); return v+qc[...,0:1]*t+torch.cross(qc[...,1:],t,dim=-1)

def generate_trajectory_heatmap(observer_pos,observer_quat,trajectories,value_signs,gammas,lidar_vbeams=24,lidar_hres=.75,lidar_vfov=(-10.,20.),downrate=4):
 """Exact upstream projection: world->body, azimuth/elevation->native bin, max/min signed pooling."""
 if trajectories.ndim==3: trajectories=trajectories.unsqueeze(1); value_signs=value_signs.unsqueeze(1); gammas=gammas.unsqueeze(1); observer_pos=observer_pos.unsqueeze(1); observer_quat=observer_quat.unsqueeze(1)
 B,P,T=trajectories.shape[:3]; Vd,Hd=lidar_vbeams//downrate,int(360/lidar_hres)//downrate; obs=observer_pos.unsqueeze(2).expand(-1,-1,T,-1); q=observer_quat.unsqueeze(2).expand(-1,-1,T,-1); rel=world_to_body(trajectories-obs,q); x,y,z=rel.unbind(-1); az=(torch.atan2(y,x)*180/torch.pi)%360.; el=torch.atan2(z,torch.sqrt((x*x+y*y).clamp_min(1e-12)))*180/torch.pi; vmin,vmax=lidar_vfov; vstep=(vmax-vmin)/(lidar_vbeams-1); ih=(az/lidar_hres).floor().long()%int(360/lidar_hres); iv=((el-vmin)/max(vstep,1e-6)).round().long().clamp(0,lidar_vbeams-1); ih=(ih//downrate).clamp(0,Hd-1); iv=(iv//downrate).clamp(0,Vd-1); idx=iv*Hd+ih; dist=torch.linalg.vector_norm(rel,dim=-1).clamp_min(1e-6); values=value_signs*(10./dist)*gammas; values=torch.where(dist<=10.,values,torch.zeros_like(values)); pos=torch.full((B,P,Vd*Hd),-torch.inf,device=rel.device); neg=torch.full_like(pos,torch.inf); pos.scatter_reduce_(2,idx,torch.where(values>0,values,-torch.inf),reduce='amax',include_self=False); neg.scatter_reduce_(2,idx,torch.where(values<0,values,torch.inf),reduce='amin',include_self=False); pos[torch.isinf(pos)]=0; neg[torch.isinf(neg)]=0; out=(pos+neg).view(B,P,Vd,Hd); assert torch.isfinite(out).all(); return out

def heatmap(observer_pos,observer_quat,evader_pos,evader_future=None,teammate_pos=None,teammate_future=None,shape=(6,120),max_range=10.,visibility=None):
 """Convenience builder with official signs/decays; future data is caller-supplied only."""
 B=observer_pos.shape[0]; P=1 if observer_pos.ndim==2 else observer_pos.shape[1]; op=observer_pos if observer_pos.ndim==3 else observer_pos.unsqueeze(1); oq=observer_quat if observer_quat.ndim==3 else observer_quat.unsqueeze(1); entities=[]; signs=[]; gammas=[]
 def add(x,s,g):
  if x is None:return
  if x.ndim==op.ndim-1:x=x.unsqueeze(-2)
  if x.ndim==3:x=x.unsqueeze(1).expand(-1,P,-1,-1)
  entities.append(x); signs.append(torch.full(x.shape[:-1],s,device=x.device,dtype=x.dtype)); gammas.append(g)
 add(evader_pos,2.,1.); add(evader_future,1.5,.95); add(teammate_pos,-2.,1.); add(teammate_future,-1.5,.9)
 if not entities:return torch.zeros((*op.shape[:-1],*shape),device=op.device)
 traj=torch.cat(entities,2); sg=torch.cat(signs,2); gm=[]
 for x,g in zip(entities,gammas): gm.append(torch.full(x.shape[:-1],g,device=x.device,dtype=x.dtype) if x.shape[2]==1 else g**torch.arange(x.shape[2],device=x.device,dtype=x.dtype).view(1,1,-1).expand(B,P,-1))
 out=generate_trajectory_heatmap(op,oq,traj,sg,torch.cat(gm,2));
 if visibility is not None: out=out*(visibility.unsqueeze(-1).unsqueeze(-1).to(out.dtype))
 return out[:,0] if observer_pos.ndim==2 else out
