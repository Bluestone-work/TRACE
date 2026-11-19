import torch
import numpy as np
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

def test_gae_uses_nonterminal_next_value_and_masks_terminal_bootstrap():
 gae=GAE(1.,1.); r=torch.zeros(1,2,1); v=torch.zeros(1,2,1); nv=torch.tensor([[[2.],[3.]]]); done=torch.tensor([[[0.],[1.]]]); adv,_=gae(r,done,v,nv)
 assert torch.allclose(adv,torch.tensor([[[2.],[0.]]]))

def test_gae_bootstraps_timeout_but_stops_recursive_propagation():
 gae=GAE(1.,1.); r=torch.zeros(1,3,1); v=torch.zeros(1,3,1); nv=torch.tensor([[[2.],[5.],[7.]]]); terminated=torch.zeros(1,3,1); episode_end=torch.tensor([[[0.],[1.],[0.]]]); adv,_=gae(r,terminated,v,nv,episode_end)
 assert torch.allclose(adv,torch.tensor([[[7.],[5.],[7.]]]))

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
 before={k:v.detach().clone() for k,v in p.actor.state_dict().items()}; log=p.update(rollout); assert all(torch.isfinite(torch.tensor(v)) for v in log.values() if not isinstance(v,int)); assert any(not torch.equal(before[k],v) for k,v in p.actor.state_dict().items()); p.step_schedulers(); ck=tmp_path/'p.pt'; p.save_state(ck,progress={'frames':7,'updates':2}); q=PSTOMAPPO(config=MAPPOConfig(training_epoch_num=1)); progress=q.load_model_weights(ck,load_optimizers=True,restore_rng=True); assert progress['frames']==7 and progress['updates']==2
 for k,v in p.actor.state_dict().items(): assert torch.equal(v,q.actor.state_dict()[k])
 payload=torch.load(ck,map_location='cpu',weights_only=False); assert isinstance(payload['numpy_rng_state'],dict); assert isinstance(payload['numpy_rng_state']['state'],list)

def test_lstm_frequency_counts_minibatches():
 torch.manual_seed(4); p=PSTOMAPPO(config=MAPPOConfig(training_epoch_num=4,num_minibatches=2,lstm_update_frequency=3)); T,B,A=2,1,2
 image=torch.rand(T,B,A,2,6,120); state=torch.rand(T,B,A,13); central=torch.rand(T,B,84); acts=[]; lps=[]; vals=[]
 for t in range(T):
  aa,ll,vv=p.act(image[t,0],state[t,0],central[t,0]); acts.append(aa); lps.append(ll); vals.append(vv)
 rollout={'image':image,'state':state,'central':central,'action':torch.stack(acts)[:,None].detach(),'log_prob':torch.stack(lps)[:,None].detach(),'value':torch.stack(vals)[:,None].detach(),'next_value':torch.zeros(T,B,1),'reward':torch.ones(T,B),'terminated':torch.zeros(T,B),'history':torch.rand(T,10,3),'future':torch.rand(T,10,3),'history_mask':torch.ones(T,10,dtype=torch.bool),'agent_dt':torch.full((T,1),1/30.),'supervised_valid':torch.ones(T,dtype=torch.bool)}
 log=p.update(rollout); assert log['minibatches']==8 and log['lstm_updates']==3


def test_heatmap_visibility_gate_and_no_future_actor_input():
 vis=torch.tensor([False]); h=heatmap(torch.zeros(1,3),torch.tensor([[1.,0,0,0.]]),torch.tensor([[1.,0,0.]]),torch.ones(1,10,3),visibility=vis); assert torch.allclose(h,torch.zeros_like(h))


def test_global_state_layout_is_84():
 from trace.pursuit.env import build_global_state_84
 p=torch.arange(9,dtype=torch.float32).reshape(3,3).numpy(); q=torch.zeros(3,4).numpy(); q[:,0]=1; v=torch.ones(3,3).numpy(); w=torch.full((3,3),2.).numpy(); a=torch.full((3,3),3.).numpy(); lidar=torch.arange(2*6*120,dtype=torch.float32).reshape(2,6,120).numpy()
 s=build_global_state_84(p,q,v,w,a,lidar); assert s.shape==(84,); assert torch.allclose(torch.as_tensor(s[39:48]),torch.full((9,),3.)); assert torch.allclose(torch.as_tensor(s[16:19]),torch.full((3,),2.)); assert torch.allclose(torch.as_tensor(s[26:29]),torch.full((3,),2.))


def test_visibility_occluded_and_reacquired():
 from trace.pursuit.visibility import target_visible
 pos=np.array([0.,0.,2.]); q=np.array([1.,0,0,0.]); target=np.array([5.,0.,2.]); clear=np.zeros((24,480),np.float32)
 assert target_visible(pos,q,target,clear)
 blocked=clear.copy(); blocked[7:11,0]=6.5  # hit distance 3.5 m, before target at 5 m
 assert not target_visible(pos,q,target,blocked)
 assert target_visible(pos,q,target,clear)


def test_official_reward_terms_and_escape_or():
 from trace.pursuit.rewards import official_reward
 pp=torch.tensor([[0.,-2.,2.],[0.,2.,2.]]); pv=torch.tensor([[1.,0.,0.],[1.,0.,0.]]); ev=torch.tensor([8.,0.,2.]); evv=torch.zeros(3); prox=torch.full((2,6,120),8.)
 r,c,e,d,t=official_reward(pp,pv,ev,prox,evader_vel=evv,step=201); assert bool(e) and bool(d); assert float(t['formation'])!=0.; assert torch.isfinite(r)


def test_actor_action_shape_and_yaw_conversion():
 from trace.pursuit.algorithm.mappo import PSTOMAPPO
 a=torch.tensor([[0.,0.],[1.,1.]]); q=torch.tensor([[1.,0,0,0],[.7071068,0,0,.7071068]])
 v=PSTOMAPPO.action_to_velocity(a,q); assert v.shape==(2,3) and torch.isfinite(v).all(); assert torch.allclose(torch.linalg.vector_norm(v[:,:2],dim=-1),torch.tensor([.72,.8]),atol=1e-3); assert v[1,1]<0


def test_reward_component_reference_values():
 from trace.pursuit.rewards import official_reward
 pp=torch.tensor([[0.,-2.,2.],[0.,2.,2.]]); pv=torch.tensor([[1.,0.,0.],[1.,0.,0.]]); ev=torch.tensor([8.,0.,2.]); prox=torch.full((2,6,120),8.)
 r,c,e,d,t=official_reward(pp,pv,ev,prox,evader_vel=torch.zeros(3),step=201)
 assert abs(float(t['pursuit'])-1.9402857)<1e-4
 assert abs(float(t['coordination'])-1.)<1e-6
 assert abs(float(t['obstacle'])-.4843306)<1e-4
 assert abs(float(t['formation'])-(-.3438776))<1e-4
 assert abs(float(r)-(-697.8105))<2e-3 and bool(e) and bool(d)


def test_lstm_right_padded_mask_uses_last_valid_position():
 from trace.pursuit.trajectory_prediction import TrajectoryPredictor
 m=TrajectoryPredictor(4,8,6); x=torch.zeros(1,10,4); x[0,:3,:3]=torch.tensor([[1.,0,0],[2.,0,0],[3.,0,0]]); mask=torch.tensor([[True,True,True,False,False,False,False,False,False,False]])
 y=m(x,mask=mask); assert y.shape==(1,6) and torch.isfinite(y).all()

def test_observation_history_does_not_fill_occlusions_and_keeps_dt():
 from trace.pursuit.env import PSTOPursuitEnv
 e=PSTOPursuitEnv.__new__(PSTOPursuitEnv); e.step_count=6; e.observed_evader_history=[np.array([1.,0.,2.]),np.array([2.,0.,2.])]; e.observed_evader_times=[0.,3.]
 h,m,dt=e._history_tensors(); assert m[:2].all() and not m[2:].any(); assert np.allclose(h[1].numpy(),[2.,0.,2.]); assert np.allclose(dt[:2].numpy().ravel(),[3/30.,3/30.])
 e.observed_evader_history=e.observed_evader_history[:1]; e.observed_evader_times=e.observed_evader_times[:1]; _,_,single_dt=e._history_tensors(); assert np.isclose(float(single_dt[0,0]),6/30.)

def test_time_scale_configs_are_explicit_and_consistent():
 import pathlib, yaml
 root=pathlib.Path(__file__).parents[2]
 official=yaml.safe_load((root/'configs/psto/official_reproduction.yaml').read_text())
 native=yaml.safe_load((root/'configs/psto/trace_native.yaml').read_text())
 assert official['environment']['policy_hz']==10. and official['environment']['frame_skip']==10
 assert official['environment']['episode_duration']==official['environment']['max_episode_steps']/official['environment']['policy_hz']
 assert native['environment']['policy_hz']==30. and native['environment']['frame_skip']==8
 assert native['environment']['episode_duration']==native['environment']['max_episode_steps']/native['environment']['policy_hz']
 assert official['experiment_name']!=native['experiment_name']

def test_visibility_rejects_target_beyond_sensor_range():
 from trace.pursuit.visibility import target_visible
 assert not target_visible(np.array([0.,0.,2.]),np.array([1.,0,0,0.]),np.array([10.5,0.,2.]),np.zeros((24,480),np.float32))
