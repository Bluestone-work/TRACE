import torch
from trace.pursuit.action import beta_action_to_velocity
from trace.pursuit.lidar import PSTOLidarAdapter
from trace.pursuit.models import FullModel
from trace.pursuit.algorithm.mappo import PSTOMAPPO, MAPPOConfig, ValueNorm, GAE
from trace.pursuit.observations import heatmap

def test_action_bounds_and_yaw():
 a=torch.tensor([[0.,0.],[1.,1.]])
 v=beta_action_to_velocity(a,torch.tensor([[1.,0,0,0],[.7071,0,0,.7071]]))
 assert torch.allclose(torch.linalg.vector_norm(v[:,:2],dim=-1),torch.tensor([.72,.8]),atol=1e-3)

def test_lidar_shape():
 x=torch.rand(2,24,480); y=PSTOLidarAdapter().downsample(x); assert y.shape==(2,6,120)

def test_model_and_mappo_forward():
 image=torch.rand(2,2,6,120); state=torch.rand(2,13); central=torch.rand(2,84); p=PSTOMAPPO(); a,lp,v=p.act(image,state,central); assert a.shape==(2,2) and lp.shape==(2,) and v.shape==(2,1)

def test_official_value_norm_and_gae():
 vn=ValueNorm(1); x=torch.tensor([[2.],[4.]]); vn.update(x); assert torch.isfinite(vn.normalize(x)).all(); adv,ret=GAE(.995,.95)(torch.ones(1,3,1),torch.zeros(1,3,1),torch.zeros(1,3,1),torch.zeros(1,3,1)); assert adv.shape==ret.shape==(1,3,1)

def test_psto_heatmap_projection():
 h=heatmap(torch.zeros(2,3),torch.tensor([[1.,0,0,0]]*2),torch.tensor([[1.,0,0],[0,1,0.]])); assert h.shape==(2,6,120) and torch.isfinite(h).all() and h.abs().sum()>0

def test_real_parameter_update_and_checkpoint(tmp_path):
 torch.manual_seed(3); p=PSTOMAPPO(config=MAPPOConfig(training_epoch_num=1)); T,B,A=2,1,2
 image=torch.rand(T,B,A,2,6,120); state=torch.rand(T,B,A,13); central=torch.rand(T,B,84); action,lp,val=p.act(image[0,0],state[0,0],central[0,0]);
 # Use a rollout with actions sampled from the current network, so the PPO ratio is finite.
 acts=[]; lps=[]; vals=[]
 for t in range(T):
  aa,ll,vv=p.act(image[t,0],state[t,0],central[t,0]); acts.append(aa); lps.append(ll); vals.append(vv)
 rollout={'image':image,'state':state,'central':central,'action':torch.stack(acts)[:,None],'log_prob':torch.stack(lps)[:,None],'value':torch.stack(vals)[:,None].detach(),'next_value':torch.zeros(T,B,1),'reward':torch.ones(T,B),'terminated':torch.zeros(T,B),'history':torch.rand(4,10,3),'future':torch.rand(4,10,3),'history_mask':torch.ones(4,10,dtype=torch.bool),'agent_dt':torch.full((4,1),1/30.)}
 before={k:v.detach().clone() for k,v in p.actor.state_dict().items()}; log=p.update(rollout); assert all(torch.isfinite(torch.tensor(v)) for v in log.values()); assert any(not torch.equal(before[k],v) for k,v in p.actor.state_dict().items()); p.step_schedulers(); ck=tmp_path/'p.pt'; p.save_state(ck); q=PSTOMAPPO(config=MAPPOConfig(training_epoch_num=1)); q.load_model_weights(ck,load_optimizers=True)
 for k,v in p.actor.state_dict().items(): assert torch.equal(v,q.actor.state_dict()[k])


def test_heatmap_visibility_gate_and_no_future_actor_input():
 vis=torch.tensor([False]); h=heatmap(torch.zeros(1,3),torch.tensor([[1.,0,0,0.]]),torch.tensor([[1.,0,0.]]),torch.ones(1,10,3),visibility=vis); assert torch.allclose(h,torch.zeros_like(h))
