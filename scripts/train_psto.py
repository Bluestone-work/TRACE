#!/usr/bin/env python3
"""Run a real short PSTO MAPPO update in the Isaac Sim PhysX environment."""
import argparse, json, random
from pathlib import Path
import numpy as np

def main():
 p=argparse.ArgumentParser(); p.add_argument('--seed',type=int,default=0); p.add_argument('--num_envs',type=int,default=1); p.add_argument('--total_frames',type=int,default=32); p.add_argument('--rollout_steps',type=int,default=8); p.add_argument('--checkpoint',default='results/psto_train/checkpoint.pt'); p.add_argument('--resume',action='store_true'); p.add_argument('--headless',action='store_true'); p.add_argument('--training_epoch_num',type=int,default=2); p.add_argument('--num_minibatches',type=int,default=2); args=p.parse_args()
 if args.num_envs!=1: raise SystemExit('The physical TRACE adapter currently supports --num_envs 1 only.')
 random.seed(args.seed); np.random.seed(args.seed)
 from isaacsim import SimulationApp
 app=SimulationApp({'headless':bool(args.headless),'multi_gpu':False}); env=None
 try:
  import torch
  from trace.pursuit.env import PSTOPursuitEnv
  from trace.pursuit.algorithm.mappo import PSTOMAPPO,MAPPOConfig
  torch.manual_seed(args.seed); env=PSTOPursuitEnv(max_episode_steps=280,seed=args.seed); device='cuda' if torch.cuda.is_available() else 'cpu'; policy=PSTOMAPPO(config=MAPPOConfig(training_epoch_num=args.training_epoch_num,num_minibatches=args.num_minibatches),device=device); ckpt=Path(args.checkpoint)
  if args.resume:
   if not ckpt.exists(): raise SystemExit(f'checkpoint not found: {ckpt}')
   policy.load_model_weights(ckpt,load_optimizers=True)
  obs=env.reset(); frames=0; updates=0; logs=[]
  while frames<args.total_frames:
   imgs=[];states=[];centrals=[];acts=[];lps=[];vals=[];rewards=[];dones=[];histories=[];futures=[];masks=[];dts=[];sup_valid=[]
   for _ in range(min(args.rollout_steps,args.total_frames-frames)):
    image=torch.as_tensor(obs['fused_image'],device=policy.device); state=torch.as_tensor(obs['state'],device=policy.device); central=torch.as_tensor(obs['central_state'],device=policy.device)
    with torch.no_grad(): action,logp,value=policy.act(image,state,central)
    next_obs,reward,done,info=env.step(action.cpu().numpy(),actions_are_normalized=True)
    imgs.append(image.cpu());states.append(state.cpu());centrals.append(central.cpu());acts.append(action.cpu());lps.append(logp.cpu());vals.append(value.reshape(1).cpu());rewards.append(float(reward));dones.append(float(done));histories.append(torch.as_tensor(obs['lstm_history'],dtype=torch.float32));futures.append(torch.as_tensor(obs['future_ground_truth'],dtype=torch.float32));masks.append(torch.as_tensor(obs['lstm_history_mask'],dtype=torch.bool));dts.append(torch.tensor([1/30.],dtype=torch.float32));sup_valid.append(bool(obs['lstm_target_available'])); frames+=1; obs=env.reset() if done else next_obs
   rollout={'image':torch.stack(imgs)[:,None],'state':torch.stack(states)[:,None],'central':torch.stack(centrals)[:,None],'action':torch.stack(acts)[:,None],'log_prob':torch.stack(lps)[:,None],'value':torch.stack(vals)[:,None],'reward':torch.tensor(rewards,dtype=torch.float32)[:,None],'terminated':torch.tensor(dones,dtype=torch.float32)[:,None],'next_value':torch.zeros(len(rewards),1,1)}
   if any(sup_valid):
    keep=torch.as_tensor(sup_valid,dtype=torch.bool); rollout.update({'history':torch.stack(histories)[keep],'future':torch.stack(futures)[keep],'history_mask':torch.stack(masks)[keep],'agent_dt':torch.stack(dts)[keep]})
   log=policy.update(rollout); policy.step_schedulers(); updates+=1; logs.append({'frames':frames,'update':updates,**log}); policy.save_state(ckpt); print(json.dumps(logs[-1]),flush=True)
  result={'status':'pass','frames':frames,'updates':updates,'checkpoint':str(ckpt),'logs':logs}
 except Exception as exc:
  import traceback; traceback.print_exc(); result={'status':'blocked','error':f'{type(exc).__name__}: {exc}'}
 finally:
  Path(args.checkpoint).parent.mkdir(parents=True,exist_ok=True); Path(args.checkpoint).with_suffix('.json').write_text(json.dumps(result,indent=2)); print(json.dumps(result,indent=2),flush=True)
  if env is not None: env.close()
  app.close()
 if result['status']!='pass': raise SystemExit(1)
if __name__=='__main__': main()
