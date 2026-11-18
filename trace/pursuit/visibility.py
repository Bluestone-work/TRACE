"""LiDAR visibility test matching PSTO's distance/alignment/occlusion logic."""
import math
import numpy as np

def target_visible(observer_pos,observer_quat,target_pos,proximity,range_m=10.,hres=.75,vbeams=24,vfov=(-10.,20.),max_distance=15.,alignment_threshold=.7,obstacle_clearance=.5):
 rel=np.asarray(target_pos)-np.asarray(observer_pos); distance=float(np.linalg.norm(rel))
 if not np.isfinite(distance) or distance>=max_distance:return False
 # Body-frame inverse rotation, scalar-first quaternion.
 q=np.asarray(observer_quat,dtype=np.float64); q=q/max(np.linalg.norm(q),1e-9); w,x,y,z=q; qc=np.array([w,-x,-y,-z]); v=rel; t=2*np.cross(qc[1:],v); b=v+qc[0]*t+np.cross(qc[1:],t)
 az=(math.degrees(math.atan2(b[1],b[0]))%360.); el=math.degrees(math.atan2(b[2],max(np.linalg.norm(b[:2]),1e-9))); hi=int(math.floor(az/hres))%int(round(360/hres)); vstep=(vfov[1]-vfov[0])/(vbeams-1); vi=int(round((el-vfov[0])/max(vstep,1e-9)))
 if vi<0 or vi>=vbeams:return False
 scan=np.asarray(proximity); hit_distance=range_m-scan; radius_h=2; radius_v=1; hidx=[(hi+j)%scan.shape[1] for j in range(-radius_h,radius_h+1)]; vidx=range(max(0,vi-radius_v),min(scan.shape[0],vi+radius_v+1)); min_hit=float(np.min(hit_distance[np.ix_(list(vidx),hidx)]))
 return min_hit >= distance-obstacle_clearance
