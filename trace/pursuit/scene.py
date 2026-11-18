"""Isaac Sim 6 physical three-vehicle scene.

Each vehicle has its own articulation, rotor view, actuator and controller state, while
all vehicles are advanced by one World at 240 Hz. Pose/velocity setters occur only in
reset().
"""
from pathlib import Path
import sys, numpy as np
_ROOT=Path(__file__).resolve().parents[2]
_NAV=_ROOT/'upstream/NavRL/isaac6-training'
if str(_NAV) not in sys.path: sys.path.insert(0,str(_NAV))
from quadrotor_dynamics import QuadrotorActuator
from rotor_flight import RotorFlight
import math
class PSTOVehicle(RotorFlight):
 def __init__(self,world,path,physics_dt):
  from isaacsim.core.prims import RigidPrim,SingleArticulation
  from isaacsim.core.utils.stage import add_reference_to_stage
  from pxr import PhysicsSchemaTools,PhysxSchema,Usd,UsdPhysics
  asset=_ROOT/'upstream/NavRL/isaac-training/third_party/OmniDrones/omni_drones/robots/assets/usd/hummingbird.usd'
  add_reference_to_stage(str(asset),path); self.world=world; self._self_query_ids=set()
  for prim in Usd.PrimRange(world.stage.GetPrimAtPath(path)):
   self._self_query_ids.add(PhysicsSchemaTools.sdfPathToInt(prim.GetPath()))
   if prim.HasAPI(UsdPhysics.RigidBodyAPI):
    api=PhysxSchema.PhysxRigidBodyAPI.Apply(prim); api.CreateDisableGravityAttr(False); api.CreateLinearDampingAttr(0.); api.CreateAngularDampingAttr(0.)
  name=path.rsplit('/',1)[-1]; self.drone=SingleArticulation(path,name=name); self.base_view=RigidPrim(path+'/base_link',name=name+'_body'); self.rotor_view=RigidPrim(path+'/rotor_[0-3]',name=name+'_rotors')
  for x in (self.drone,self.base_view,self.rotor_view): world.scene.add(x)
  self.actuator=QuadrotorActuator(physics_dt)
  self.vertical=24; self.horizontal=480
  import omni.physics.core
  self._query=omni.physics.core.get_physics_scene_query_interface()
 def _raycast_lidar(self):
  distances=np.full((self.vertical,self.horizontal),10.,dtype=np.float32); origin=self.position.astype(np.float64); vertical=np.deg2rad(np.linspace(-10.,20.,self.vertical))
  for hi in range(self.horizontal):
   yaw=2.*math.pi*hi/self.horizontal+self.lidar_yaw
   for vi,pitch in enumerate(vertical):
    direction=(math.cos(pitch)*math.cos(yaw),math.cos(pitch)*math.sin(yaw),math.sin(pitch))
    def report(hit):
     if hit.rigid_body not in self._self_query_ids and hit.collision not in self._self_query_ids: distances[vi,hi]=min(distances[vi,hi],float(hit.distance))
     return True
    self._query.raycast_all(tuple(float(x) for x in origin),direction,10.,report,True)
  return 10.-distances
 def reset(self,pos):
  self.rotor_joint_indices=np.array([self.drone.get_dof_index(f'rotor_{i}_joint') for i in range(4)])
  self.drone.set_world_pose(position=np.asarray(pos),orientation=np.array([1.,0,0,0])); self.drone.set_linear_velocity(np.zeros(3)); self.drone.set_angular_velocity(np.zeros(3)); self.actuator.reset(); self.drone.set_joint_positions(np.zeros(4),joint_indices=self.rotor_joint_indices); self.drone.set_joint_velocities(self.actuator.rotor_speeds.astype(np.float32),joint_indices=self.rotor_joint_indices); self._read_physics_state()
class PSTOPursuitScene:
 def __init__(self,positions=None,physics_dt=1/240,policy_hz=30.):
  from isaacsim.core.api import World
  self.physics_dt=physics_dt; self.policy_dt=1/policy_hz; self.physics_substeps=round(self.policy_dt/physics_dt); self.world=World(physics_dt=physics_dt,rendering_dt=self.policy_dt,stage_units_in_meters=1.)
  if positions is None: positions=((0,2.3,2), (0,-2.3,2), (8,0,2))
  self.vehicles=[PSTOVehicle(self.world,f'/World/Pursuit/Drone_{i}',physics_dt) for i in range(3)]; self.positions=np.asarray(positions,dtype=np.float32); self.world.reset()
  for v,p in zip(self.vehicles,self.positions): v.reset(p)
  self.world.render()
 def reset(self,positions=None):
  if positions is not None:self.positions=np.asarray(positions,dtype=np.float32)
  self.world.reset()
  for v,p in zip(self.vehicles,self.positions):v.reset(p)
  self.world.render()
 def step(self,commands):
  commands=np.asarray(commands,dtype=np.float32).reshape(3,3)
  for _ in range(self.physics_substeps):
   for v,c in zip(self.vehicles,commands):v._apply_rotor_control(c)
   self.world.step(render=False)
  for v in self.vehicles:v._read_physics_state()
  return np.stack([v.position for v in self.vehicles])
 def close(self): self.world.clear_instance()
