"""Minimal physical PSTO environment facade; algorithm tensors stay framework independent."""
import numpy as np
from .scene import PSTOPursuitScene
from .rewards import pursuit_reward
class PSTOPursuitEnv:
 def __init__(self,**kwargs): self.scene=PSTOPursuitScene(**kwargs); self.step_count=0
 def reset(self): self.step_count=0; self.scene.reset(); return self.observe()
 def observe(self): return {'positions':np.stack([v.position for v in self.scene.vehicles]),'velocities':np.stack([v.velocity for v in self.scene.vehicles])}
 def step(self,commands):
  pos=self.scene.step(commands); self.step_count+=1; r,c,e=pursuit_reward(__import__('torch').as_tensor(pos[:2]),__import__('torch').as_tensor(pos[2]),self.step_count); return self.observe(),float(r.item()),bool(c.item() or e.item()),{'capture':bool(c.item()),'escaped':bool(e.item())}
 def close(self):self.scene.close()
