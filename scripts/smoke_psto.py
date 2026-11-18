#!/usr/bin/env python3
import argparse, json, torch
from trace.pursuit.lidar import PSTOLidarAdapter
from trace.pursuit.algorithm.mappo import PSTOMAPPO

def main():
 p=argparse.ArgumentParser(); p.add_argument('--steps',type=int,default=3); args=p.parse_args(); torch.manual_seed(0)
 lidar=PSTOLidarAdapter(); model=PSTOMAPPO(); result=[]
 for i in range(args.steps):
  image=torch.rand(2,2,*lidar.down_shape); state=torch.rand(2,13); central=torch.rand(2,84)
  with torch.no_grad(): action,logp,value=model.act(image,state,central)
  result.append({'step':i,'action_finite':bool(torch.isfinite(action).all()),'value_finite':bool(torch.isfinite(value).all())})
 print(json.dumps({'status':'pass','lidar_down_shape':lidar.down_shape,'checks':result},indent=2))
if __name__=='__main__': main()
