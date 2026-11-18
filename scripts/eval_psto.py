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
  device='cuda' if torch.cuda.is_available() else 'cpu'; policy=PSTOMAPPO(device=device); policy.load_model_weights(args.checkpoint,load_optimizers=False); policy.eval(); env=PSTOPursuitEnv(max_episode_steps=args.max_steps,seed=args.seed); env.set_policy_lstm(policy.lstm)
  rows=[]
  for ep in range(args.episodes):
   obs=env.reset(); total=0.; vis=0.; tf_ade=[]; tf_fde=[]; oo_ade=[]; oo_fde=[]; terminal={}; done=False; lost_at=None; reacq=[]; true_positions=[env._root()[0][2].copy()]; prediction_queue=[]
   for step in range(args.max_steps):
    with torch.no_grad(): action,_,_=policy.act(torch.as_tensor(obs['fused_image'],device=device),torch.as_tensor(obs['state'],device=device),torch.as_tensor(obs['central_state'],device=device),deterministic=True)
    pred_tf=pred_oo=None
    if np.asarray(obs['teacher_history_mask']).any():
     with torch.no_grad():
      th=torch.as_tensor(obs['teacher_history'],device=device)[None]; tm=torch.as_tensor(obs['teacher_history_mask'],device=device)[None]; td=torch.as_tensor(obs['teacher_history_dt'],device=device)[None]; pred_tf=policy.lstm(torch.cat((th,td),-1),mask=tm).view(10,3).cpu().numpy()
    if np.asarray(obs['history_mask']).any():
     with torch.no_grad():
      oh=torch.as_tensor(obs['history'],device=device)[None]; om=torch.as_tensor(obs['history_mask'],device=device)[None]; od=torch.as_tensor(obs['history_dt'],device=device)[None]; pred_oo=policy.lstm(torch.cat((oh,od),-1),mask=om).view(10,3).cpu().numpy()
    prediction_queue.append((len(true_positions)-1,pred_tf,pred_oo))
    nxt,reward,done,info=env.step(action.cpu().numpy(),actions_are_normalized=True); total+=reward; vis+=info['visibility_ratio']; terminal=info; true_positions.append(env._root()[0][2].copy())
    while prediction_queue and len(true_positions)-1-prediction_queue[0][0]>=10:
     start,pred_tf_i,pred_oo_i=prediction_queue.pop(0); gt=np.asarray(true_positions[start+1:start+11],np.float32)
     if pred_tf_i is not None: tf_ade.append(float(np.linalg.norm(pred_tf_i-gt,axis=-1).mean())); tf_fde.append(float(np.linalg.norm(pred_tf_i[-1]-gt[-1])))
     if pred_oo_i is not None: oo_ade.append(float(np.linalg.norm(pred_oo_i-gt,axis=-1).mean())); oo_fde.append(float(np.linalg.norm(pred_oo_i[-1]-gt[-1])))
    visible=info['visibility_ratio']>0.0
    if not visible and lost_at is None: lost_at=step
    if visible and lost_at is not None: reacq.append((step-lost_at)/30.); lost_at=None
    if done:
     obs=env.reset(); true_positions=[env._root()[0][2].copy()]; prediction_queue=[]
    else: obs=nxt
    if done: break
   rows.append({'episode':ep,'steps':step+1,'reward':total,'capture':bool(terminal.get('capture',False)),'escaped':bool(terminal.get('escaped',False)),'collision':bool(terminal.get('collision',False)),'visibility_ratio':vis/(step+1),'reacquisition_time_s':float(np.mean(reacq)) if reacq else None,'teacher_forced_ade':float(np.mean(tf_ade)) if tf_ade else None,'teacher_forced_fde':float(np.mean(tf_fde)) if tf_fde else None,'observation_only_ade':float(np.mean(oo_ade)) if oo_ade else None,'observation_only_fde':float(np.mean(oo_fde)) if oo_fde else None})
  result={'status':'pass','episodes':len(rows),'capture_success_rate':float(np.mean([r['capture'] for r in rows])),'average_capture_time_s':float(np.mean([r['steps']/30 for r in rows if r['capture']])) if any(r['capture'] for r in rows) else None,'escape_rate':float(np.mean([r['escaped'] for r in rows])),'collision_rate':float(np.mean([r['collision'] for r in rows])),'target_visibility_ratio':float(np.mean([r['visibility_ratio'] for r in rows])),'reacquisition_time_s':float(np.mean([r['reacquisition_time_s'] for r in rows if r['reacquisition_time_s'] is not None])) if any(r['reacquisition_time_s'] is not None for r in rows) else None,'teacher_forced_ade':float(np.mean([r['teacher_forced_ade'] for r in rows if r['teacher_forced_ade'] is not None])) if any(r['teacher_forced_ade'] is not None for r in rows) else None,'teacher_forced_fde':float(np.mean([r['teacher_forced_fde'] for r in rows if r['teacher_forced_fde'] is not None])) if any(r['teacher_forced_fde'] is not None for r in rows) else None,'observation_only_ade':float(np.mean([r['observation_only_ade'] for r in rows if r['observation_only_ade'] is not None])) if any(r['observation_only_ade'] is not None for r in rows) else None,'observation_only_fde':float(np.mean([r['observation_only_fde'] for r in rows if r['observation_only_fde'] is not None])) if any(r['observation_only_fde'] is not None for r in rows) else None,'episode_reward':float(np.mean([r['reward'] for r in rows])),'episodes_detail':rows}
 except Exception as exc:
  import traceback; traceback.print_exc(); result={'status':'blocked','error':f'{type(exc).__name__}: {exc}'}
 finally:
  out=Path(args.out); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(result,indent=2)); print(json.dumps(result,indent=2),flush=True)
  if env is not None: env.close()
  app.close()
 if result['status']!='pass': raise SystemExit(1)
if __name__=='__main__': main()
