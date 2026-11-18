"""Small dependency-light implementation of the official MAPPO components."""
import torch
from torch import nn
from ..models import FullModel
from ..trajectory_prediction import TrajectoryPredictor
class ValueNorm(nn.Module):
 def __init__(self): super().__init__(); self.register_buffer('mean',torch.zeros(1)); self.register_buffer('sq',torch.ones(1))
 def normalize(self,x): return (x-self.mean)/self.sq.sqrt().clamp_min(1e-6)
 def denormalize(self,x): return x*self.sq.sqrt()+self.mean
 def update(self,x): self.mean.mul_(.995).add_(.005*x.mean()); self.sq.mul_(.995).add_(.005*(x*x).mean())
class BetaActor(nn.Module):
 def __init__(self,dim=48,action_dim=2):
  super().__init__(); self.net=nn.Sequential(nn.Linear(dim,72),nn.LayerNorm(72),nn.LeakyReLU(.01),nn.Linear(72,36),nn.LayerNorm(36),nn.LeakyReLU(.01)); self.a=nn.Linear(36,action_dim); self.b=nn.Linear(36,action_dim)
 def forward(self,x): h=self.net(x); return 1+nn.functional.softplus(self.a(h)),1+nn.functional.softplus(self.b(h))
class PSTOMAPPO(nn.Module):
 def __init__(self,state_dim=13,central_dim=84):
  super().__init__(); self.encoder=FullModel(state_dim=state_dim); self.actor=BetaActor(); self.critic=nn.Sequential(nn.Linear(central_dim,128),nn.Tanh(),nn.Linear(128,1)); self.value_norm=ValueNorm(); self.lstm=TrajectoryPredictor(4,64,30)
 def act(self,image,state,central=None,deterministic=False):
  feat=self.encoder(image,state); alpha,beta=self.actor(feat); dist=torch.distributions.Beta(alpha,beta); a=(alpha/(alpha+beta) if deterministic else dist.rsample()); return a,dist.log_prob(a).sum(-1),self.critic(central) if central is not None else None
