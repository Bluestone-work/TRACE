import torch
from trace.pursuit.action import beta_action_to_velocity
from trace.pursuit.lidar import PSTOLidarAdapter
from trace.pursuit.models import FullModel
from trace.pursuit.algorithm.mappo import PSTOMAPPO

def test_action_bounds_and_yaw():
 a=torch.tensor([[0.,0.],[1.,1.]])
 v=beta_action_to_velocity(a,torch.tensor([[1.,0,0,0],[.7071,0,0,.7071]]))
 assert torch.allclose(torch.linalg.vector_norm(v[:,:2],dim=-1),torch.tensor([.72,.8]),atol=1e-3)

def test_lidar_shape():
 x=torch.rand(2,24,480); y=PSTOLidarAdapter().downsample(x); assert y.shape==(2,6,120)

def test_model_and_mappo_forward():
 image=torch.rand(2,2,6,120); state=torch.rand(2,13); central=torch.rand(2,84)
 p=PSTOMAPPO(); a,lp,v=p.act(image,state,central); assert a.shape==(2,2) and lp.shape==(2,) and v.shape==(2,1)
