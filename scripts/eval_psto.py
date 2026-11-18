#!/usr/bin/env python3
"""Independent evaluation of a physical PSTO checkpoint."""
import argparse, json, random
from pathlib import Path
import numpy as np

def main():
 p=argparse.ArgumentParser(); p.add_argument('--checkpoint',required=True); p.add_argument('--episodes',type=int,default=5); p.add_argument('--seed',type=int,default=100); p.add_argument('--headless',action='store_true'); p.add_argument('--max_steps',type=int,default=280); p.add_argument('--out',default='results/psto_eval/eval.json'); args=p.parse_args()
 random.seed(args.seed); np.random.seed(args.seed)
 from isaacsim import SimulationApp
 app=SimulationApp({'headless':bool(args.headless),'multi_gpu':False}); env=None
 try:
  import torch
  from trace.pursuit.env import PSTOPursuitEnv
  from trace.pursuit.algorithm.mappo import PSTOMAPPO
  device='cuda' if torch.cuda.is_available() else 'cpu'; policy=PSTOMAPPO(device=device); policy.load_model_weights(args.checkpoint,load_optimizers=False); policy.eval(); env=PSTOPursuitEnv(max_episode_steps=args.max_steps,seed=args.seed)
  rows=[]
  for ep in range(args.episodes):
   obs=env.reset(); total=0.; vis=0.; ade=[]; fde=[]; terminal={}; done=False; lost_at=None; reacq=[]
   for step in range(args.max_steps):
    with torch.no_grad(): action,_,_=policy.act(torch.as_tensor(obs['fused_image'],device=device),torch.as_tensor(obs['state'],device=device),torch.as_tensor(obs['central_state'],device=device),deterministic=True)
    if obs['lstm_target_available']:
     with torch.no_grad(): pred=policy.lstm(torch.cat((torch.as_tensor(obs['lstm_history'],device=device)[None],torch.full((1,10,1),1/30.,device=device)),-1),mask=torch.as_tensor(obs['lstm_history_mask'],device=device)[None]).view(10,3).cpu().numpy()
     gt=np.asarray(obs['future_ground_truth']); ade.append(float(np.linalg.norm(pred-gt,axis=-1).mean())); fde.append(float(np.linalg.norm(pred[-1]-gt[-1])))
    nxt,reward,done,info=env.step(action.cpu().numpy(),actions_are_normalized=True); total+=reward; vis+=info['visibility_ratio']; terminal=info
    visible=info['visibility_ratio']>0.0
    if not visible and lost_at is None: lost_at=step
    if visible and lost_at is not None: reacq.append((step-lost_at)/30.); lost_at=None
    obs=env.reset() if done else nxt
    if done: break
   rows.append({'episode':ep,'steps':step+1,'reward':total,'capture':bool(terminal.get('capture',False)),'escaped':bool(terminal.get('escaped',False)),'collision':bool(terminal.get('collision',False)),'visibility_ratio':vis/(step+1),'reacquisition_time_s':float(np.mean(reacq)) if reacq else None,'lstm_ade':float(np.mean(ade)) if ade else None,'lstm_fde':float(np.mean(fde)) if fde else None})
  result={'status':'pass','episodes':len(rows),'capture_success_rate':float(np.mean([r['capture'] for r in rows])),'average_capture_time_s':float(np.mean([r['steps']/30 for r in rows if r['capture']])) if any(r['capture'] for r in rows) else None,'escape_rate':float(np.mean([r['escaped'] for r in rows])),'collision_rate':float(np.mean([r['collision'] for r in rows])),'target_visibility_ratio':float(np.mean([r['visibility_ratio'] for r in rows])),'reacquisition_time_s':float(np.mean([r['reacquisition_time_s'] for r in rows if r['reacquisition_time_s'] is not None])) if any(r['reacquisition_time_s'] is not None for r in rows) else None,'lstm_ade':float(np.mean([r['lstm_ade'] for r in rows if r['lstm_ade'] is not None])) if any(r['lstm_ade'] is not None for r in rows) else None,'lstm_fde':float(np.mean([r['lstm_fde'] for r in rows if r['lstm_fde'] is not None])) if any(r['lstm_fde'] is not None for r in rows) else None,'episode_reward':float(np.mean([r['reward'] for r in rows])),'episodes_detail':rows}
 except Exception as exc:
  import traceback; traceback.print_exc(); result={'status':'blocked','error':f'{type(exc).__name__}: {exc}'}
 finally:
  out=Path(args.out); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(result,indent=2)); print(json.dumps(result,indent=2),flush=True)
  if env is not None: env.close()
  app.close()
 if result['status']!='pass': raise SystemExit(1)
if __name__=='__main__': main()
