#!/usr/bin/env python3
"""Physical Isaac Sim 6 acceptance test for three rotor-driven Hummingbirds."""
import argparse,csv,json
from pathlib import Path
import numpy as np

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--steps',type=int,default=6); ap.add_argument('--out',default='results/psto_scene'); ap.add_argument('--headless',action='store_true'); args=ap.parse_args(); out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
 from isaacsim import SimulationApp
 app=SimulationApp({'headless':bool(args.headless),'multi_gpu':False,'width':1280,'height':720})
 scene=None; rows=[]; failures=[]; screenshot_error=None; screenshot_path=out/'psto_scene.png'; screenshot_path.unlink(missing_ok=True); (out/'screenshot_error.txt').unlink(missing_ok=True)
 try:
  from trace.pursuit.scene import PSTOPursuitScene
  scene=PSTOPursuitScene(); scene.reset()
  for k in range(args.steps):
   pos=scene.step(np.zeros((3,3),np.float32))
   for i,v in enumerate(scene.vehicles):
    p=np.asarray(v.position); vel=np.asarray(v.velocity); q=np.asarray(v.orientation); w=np.asarray(v.angular_velocity); rs=np.asarray(v.actuator.rotor_speeds); lidar=v._raycast_lidar()
    if not np.isfinite(np.r_[p,vel,q,w,rs,lidar.ravel()]).all(): failures.append(f'nonfinite_drone_{i}_step_{k}')
    rows.append({'step':k,'drone':i,'x':float(p[0]),'y':float(p[1]),'z':float(p[2]),'vx':float(vel[0]),'vy':float(vel[1]),'vz':float(vel[2]),'qw':float(q[0]),'qx':float(q[1]),'qy':float(q[2]),'qz':float(q[3]),'wx':float(w[0]),'wy':float(w[1]),'wz':float(w[2]),'rotor0':float(rs[0]),'rotor1':float(rs[1]),'rotor2':float(rs[2]),'rotor3':float(rs[3]),'lidar_min_clearance':float(np.min(lidar)),'lidar_shape':'24x480'})
  before=[v.actuator.rotor_speeds.copy() for v in scene.vehicles]; scene.step(np.array([[.5,0,0],[0,0,0],[0,0,0]],np.float32)); after=[v.actuator.rotor_speeds.copy() for v in scene.vehicles]
  if scene.vehicles[0].actuator is scene.vehicles[1].actuator or np.allclose(after[0],before[0]) or not np.allclose(after[1],before[1]): failures.append('independent_rotor_control_failed')
  try:
   # Use an explicit RTX camera/render product so the screenshot is meaningful in SSH/headless mode.
   import omni.replicator.core as rep
   from pxr import UsdGeom,Gf
   cam=UsdGeom.Camera.Define(scene.world.stage,'/World/PSTOCamera'); cam.CreateFocalLengthAttr(24.); cam.CreateClippingRangeAttr(Gf.Vec2f(.05,100.)); xf=UsdGeom.Xformable(cam).AddTransformOp(); xf.Set(Gf.Matrix4d().SetLookAt(Gf.Vec3d(14.,-14.,10.),Gf.Vec3d(2.,0.,2.),Gf.Vec3d(0.,0.,1.)).GetInverse())
   product=rep.create.render_product('/World/PSTOCamera',(1280,720)); annotator=rep.AnnotatorRegistry.get_annotator('rgb'); annotator.attach([product]); scene.world.render(); rep.orchestrator.step(delta_time=0.,rt_subframes=2,pause_timeline=False); data=annotator.get_data();
   from PIL import Image
   Image.fromarray(np.asarray(data)[...,:3]).save(out/'psto_scene.png'); annotator.detach([product]); product.destroy()
  except Exception as exc:
   screenshot_error=repr(exc); (out/'screenshot_error.txt').write_text(screenshot_error)
  status='pass' if not failures else 'fail'
 except Exception as exc:
  status='blocked'; failures.append(type(exc).__name__+': '+str(exc)); (out/'exception.txt').write_text(repr(exc))

 if not screenshot_path.is_file(): screenshot_status='missing'
 else:
  try:
   from PIL import Image
   screenshot_status='black_or_unavailable' if float(np.asarray(Image.open(screenshot_path))[...,:3].mean())<1.0 else 'file_created'
  except Exception: screenshot_status='file_created_unchecked'
 result={'status':status,'steps':args.steps,'physics_dt':1/240,'policy_dt':1/30,'physics_substeps':8,'num_vehicles':3,'failures':failures,'csv':str(out/'psto_scene.csv'),'screenshot':str(screenshot_path),'screenshot_status':screenshot_status,'screenshot_error':screenshot_error}
 with (out/'psto_scene.csv').open('w',newline='') as f:
  fields=rows[0].keys() if rows else ['step']; w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)
 (out/'psto_scene.json').write_text(json.dumps(result,indent=2));
 try:
  import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
  fig=plt.figure(figsize=(8,6)); ax=fig.add_subplot(111,projection='3d')
  for i,color,label in [(0,'tab:blue','Pursuer 0'),(1,'tab:orange','Pursuer 1'),(2,'tab:red','Evader')]:
   rr=[x for x in rows if x['drone']==i]; ax.plot([x['x'] for x in rr],[x['y'] for x in rr],[x['z'] for x in rr],color=color,label=label)
  ax.set_xlabel('x (m)'); ax.set_ylabel('y (m)'); ax.set_zlabel('z (m)'); ax.legend(); fig.tight_layout(); fig.savefig(out/'psto_trajectory.png',dpi=140); plt.close(fig)
 except Exception as exc: (out/'trajectory_plot_error.txt').write_text(repr(exc))
 print(json.dumps(result,indent=2),flush=True)
 if scene is not None: scene.close()
 app.close()
 if status!='pass': raise SystemExit(1)
if __name__=='__main__': main()
