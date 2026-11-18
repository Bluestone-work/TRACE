#!/usr/bin/env python3
"""P0-6 physical duration tests: 10 s hover and 30 s independent control."""
import argparse,csv,json
from pathlib import Path
import numpy as np

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--out',default='results/psto_long'); ap.add_argument('--headless',action='store_true'); args=ap.parse_args(); out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
 from isaacsim import SimulationApp
 app=SimulationApp({'headless':bool(args.headless),'multi_gpu':False}); scene=None; rows=[]; failures=[]; response=[]
 try:
  from trace.pursuit.scene import PSTOPursuitScene
  scene=PSTOPursuitScene(); scene.reset()
  def run(seconds,name,command):
   steps=int(seconds*30); z=[]; speeds=[]; moved=[]
   for k in range(steps):
    scene.step(command(k)); p=np.stack([v.position for v in scene.vehicles]); vel=np.stack([v.velocity for v in scene.vehicles]);
    if not np.isfinite(np.r_[p,vel]).all(): failures.append(f'nonfinite_{name}_{k}')
    z.extend(p[:,2].tolist()); speeds.extend(np.linalg.norm(vel,axis=1).tolist()); moved.append(p.copy())
    if k%30==0:
     scans=[v._raycast_lidar() for v in scene.vehicles];
     if not np.isfinite(np.concatenate([x.reshape(-1) for x in scans])).all(): failures.append(f'lidar_nonfinite_{name}_{k}')
    rows.append({'test':name,'step':k,'drone0_x':float(p[0,0]),'drone0_y':float(p[0,1]),'drone0_z':float(p[0,2]),'drone1_x':float(p[1,0]),'drone1_y':float(p[1,1]),'drone1_z':float(p[1,2]),'evader_x':float(p[2,0]),'evader_y':float(p[2,1]),'evader_z':float(p[2,2]),'drone0_speed':float(np.linalg.norm(vel[0])),'drone1_speed':float(np.linalg.norm(vel[1]))})
   return np.asarray(z),np.asarray(speeds),np.asarray(moved)
  z1,s1,m1=run(10.,'hover',lambda k:np.zeros((3,3),np.float32));
  if z1.min()<.5 or z1.max()>5.5: failures.append('hover_altitude_unstable')
  scene.reset(); z2,s2,m2=run(30.,'independent',lambda k:np.array([[.5,0,0],[0,0,0],[0,0,0]],np.float32));
  displacement=float(np.linalg.norm(m2[-1,0,:2]-m2[0,0,:2])); stationary=float(np.linalg.norm(m2[-1,1,:2]-m2[0,1,:2])); peak=float(s2.reshape(-1,3)[:,0].max()) if s2.size else 0.
  if displacement<.05: failures.append('position_response_missing')
  if z2.min()<.5 or z2.max()>5.5: failures.append('independent_altitude_unstable')
  if peak<.05: failures.append('velocity_response_missing')
  result={'status':'pass' if not failures else 'fail','hover_seconds':10,'independent_seconds':30,'physics_hz':240,'policy_hz':30,'physics_substeps':8,'position_response_m':displacement,'stationary_drone_displacement_m':stationary,'peak_speed_mps':peak,'failures':failures,'csv':str(out/'long.csv')}
 except Exception as exc:
  result={'status':'blocked','hover_seconds':10,'independent_seconds':30,'failures':[type(exc).__name__+': '+str(exc)],'csv':str(out/'long.csv')}; (out/'exception.txt').write_text(repr(exc))
 with (out/'long.csv').open('w',newline='') as f:
  fields=rows[0].keys() if rows else ['test']; w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
 (out/'long.json').write_text(json.dumps(result,indent=2)); print(json.dumps(result,indent=2),flush=True)
 if scene is not None: scene.close()
 app.close()
 if result['status']!='pass': raise SystemExit(1)
if __name__=='__main__':main()
