"""LiDAR visibility test matching PSTO distance/alignment/occlusion logic."""
import math
import numpy as np

def target_visible(observer_pos,observer_quat,target_pos,proximity,range_m=10.,hres=.75,vbeams=24,vfov=(-10.,20.),max_distance=15.,alignment_threshold=.7,obstacle_clearance=.5):
 rel=np.asarray(target_pos,dtype=np.float64)-np.asarray(observer_pos,dtype=np.float64); distance=float(np.linalg.norm(rel))
 if not np.isfinite(distance) or distance>=max_distance:return False
 q=np.asarray(observer_quat,dtype=np.float64); q=q/max(np.linalg.norm(q),1e-9); w,x,y,z=q; qc=np.array([w,-x,-y,-z]); t=2*np.cross(qc[1:],rel); body=rel+qc[0]*t+np.cross(qc[1:],t); target_dir=body/max(np.linalg.norm(body),1e-9)
 scan=np.asarray(proximity,dtype=np.float64); hit_distance=range_m-scan; valid=np.isfinite(hit_distance)&(scan>1e-5)&(hit_distance<range_m-1e-5)
 if not valid.any(): return True
 hi_grid=np.arange(scan.shape[1]); az=np.deg2rad(hi_grid*hres); el=np.deg2rad(np.linspace(vfov[0],vfov[1],vbeams)); gx=np.cos(el)[:,None]*np.cos(az)[None,:]; gy=np.cos(el)[:,None]*np.sin(az)[None,:]; gz=np.sin(el)[:,None]*np.ones((1,scan.shape[1])); dirs=np.stack((gx,gy,gz),axis=-1); aligned=np.sum(dirs*target_dir[None,None,:],axis=-1)>alignment_threshold; relevant=valid&aligned
 if not relevant.any(): return True
 return float(np.min(hit_distance[relevant])) >= distance-obstacle_clearance
