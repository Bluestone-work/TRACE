#!/usr/bin/env python3
"""P0 integration: Isaac Sim 6 + full PSTO observation/actor/action/rotor loop."""
import argparse,csv,json,time
from pathlib import Path
import numpy as np

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--steps',type=int,default=20); ap.add_argument('--out',default='results/psto_integration'); ap.add_argument('--headless',action='store_true'); args=ap.parse_args(); out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
 from isaacsim import SimulationApp
 app=SimulationApp({'headless':bool(args.headless),'multi_gpu':False,'width':1280,'height':720})
 rows=[]; failures=[]; env=None
 try:
  import torch
  from trace.pursuit.env import PSTOPursuitEnv
  from trace.pursuit.algorithm.mappo import PSTOMAPPO
  env=PSTOPursuitEnv(max_episode_steps=280); policy=PSTOMAPPO(device='cuda' if torch.cuda.is_available() else 'cpu'); obs=env.reset(); assert obs['central_state'].shape==(84,); assert obs['state'].shape==(2,13); assert obs['fused_image'].shape==(2,2,6,120)
  for k in range(args.steps):
   image=torch.as_tensor(obs['fused_image'],device=policy.device); state=torch.as_tensor(obs['state'],device=policy.device); central=torch.as_tensor(obs['central_state'],device=policy.device)
   with torch.no_grad(): action,logp,value=policy.act(image,state,central)
   assert action.shape==(2,2) and torch.isfinite(action).all() and torch.isfinite(value).all()
   # Ground-truth future labels exist only in info; they are intentionally never passed to act().
   next_obs,reward,done,info=env.step(action.cpu().numpy(),actions_are_normalized=True); assert next_obs['central_state'].shape==(84,); assert np.isfinite(next_obs['fused_image']).all(); assert np.isfinite(next_obs['central_state']).all()
   rows.append({'step':k,'reward':reward,'done':done,'visibility_ratio':info['visibility_ratio'],'distance':info['distance'],'action_max':float(action.max()),'value':float(value.mean())}); obs=env.reset() if done else next_obs
  status='pass' if not failures else 'fail'
 except Exception as exc:
  import traceback; traceback.print_exc()
  status='blocked'; failures.append(type(exc).__name__+': '+str(exc)); (out/'exception.txt').write_text(repr(exc))
 result={'status':status,'steps_requested':args.steps,'steps_completed':len(rows),'failures':failures,'no_future_label_to_actor':True,'central_state_dim':84,'actor_action_dim':2,'csv':str(out/'integration.csv')};
 with (out/'integration.csv').open('w',newline='') as f:
  fields=rows[0].keys() if rows else ['step']; w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
 (out/'integration.json').write_text(json.dumps(result,indent=2)); print(json.dumps(result,indent=2),flush=True)
 if env is not None: env.close()
 app.close()
 if status!='pass': raise SystemExit(1)
if __name__=='__main__': main()
