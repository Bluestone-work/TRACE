#!/usr/bin/env python3
"""One real PPO + LSTM parameter update and checkpoint roundtrip."""
import argparse,json
from pathlib import Path
import torch
from trace.pursuit.algorithm.mappo import PSTOMAPPO,MAPPOConfig

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--out',default='results/psto_update'); args=ap.parse_args(); out=Path(args.out); out.mkdir(parents=True,exist_ok=True); torch.manual_seed(4348)
 p=PSTOMAPPO(config=MAPPOConfig(training_epoch_num=1)); T,B,A=2,1,2; image=torch.rand(T,B,A,2,6,120); state=torch.rand(T,B,A,13); central=torch.rand(T,B,84); acts=[];lps=[];vals=[]
 for t in range(T):
  a,lp,v=p.act(image[t,0],state[t,0],central[t,0]);acts.append(a);lps.append(lp);vals.append(v)
 before={k:v.detach().clone() for k,v in p.actor.state_dict().items()}; log=p.update({'image':image,'state':state,'central':central,'action':torch.stack(acts)[:,None],'log_prob':torch.stack(lps)[:,None],'value':torch.stack(vals)[:,None].detach(),'next_value':torch.zeros(T,B,1),'reward':torch.ones(T,B),'terminated':torch.zeros(T,B),'history':torch.rand(4,10,3),'future':torch.rand(4,10,3),'history_mask':torch.ones(4,10,dtype=torch.bool),'agent_dt':torch.full((4,1),1/30.)}); ck=out/'checkpoint.pt'; p.save_state(ck); q=PSTOMAPPO(config=MAPPOConfig(training_epoch_num=1)); q.load_model_weights(ck,load_optimizers=True); changed=any(not torch.equal(before[k],v) for k,v in p.actor.state_dict().items()); result={'status':'pass' if changed else 'fail','parameter_changed':changed,'checkpoint':str(ck),'log':log}; (out/'update.json').write_text(json.dumps(result,indent=2)); print(json.dumps(result,indent=2))
if __name__=='__main__':main()
