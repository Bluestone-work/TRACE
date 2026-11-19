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

def _portable_rng_state(rng):
 def convert(value):
  if isinstance(value, dict): return {k: convert(v) for k, v in value.items()}
  if isinstance(value, np.ndarray): return value.tolist()
  if isinstance(value, np.generic): return value.item()
  return value
 return convert(rng.bit_generator.state)
class PSTOVehicle(RotorFlight):
 def __init__(self,world,path,physics_dt):
  from isaacsim.core.prims import RigidPrim,SingleArticulation
  from isaacsim.core.utils.stage import add_reference_to_stage
  from pxr import PhysicsSchemaTools,PhysxSchema,Usd,UsdPhysics
  asset=_ROOT/'upstream/NavRL/isaac-training/third_party/OmniDrones/omni_drones/robots/assets/usd/hummingbird.usd'
  add_reference_to_stage(str(asset),path); self.world=world; self._self_query_ids=set(); self._scene_lidar_epoch=0; self._lidar_cache_epoch=-1; self._lidar_cache=None
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
  if self._lidar_cache_epoch == self._scene_lidar_epoch and self._lidar_cache is not None:
   return self._lidar_cache.copy()
  distances=np.full((self.vertical,self.horizontal),10.,dtype=np.float32); origin=self.position.astype(np.float64); vertical=np.deg2rad(np.linspace(-10.,20.,self.vertical))
  for hi in range(self.horizontal):
   yaw=2.*math.pi*hi/self.horizontal+self.lidar_yaw
   for vi,pitch in enumerate(vertical):
    direction=(math.cos(pitch)*math.cos(yaw),math.cos(pitch)*math.sin(yaw),math.sin(pitch))
    def report(hit):
     if hit.rigid_body not in self._self_query_ids and hit.collision not in self._self_query_ids: distances[vi,hi]=min(distances[vi,hi],float(hit.distance))
     return True
    self._query.raycast_all(tuple(float(x) for x in origin),direction,10.,report,True)
  self._lidar_cache=10.-distances; self._lidar_cache_epoch=self._scene_lidar_epoch
  return self._lidar_cache.copy()
 def _raycast_lidar_hit_points(self):
  """Return world-space points for the rays that hit geometry.

  The native adapter stores proximity (``range - hit_distance``), so convert
  only non-zero hits back to points.  Zero-valued no-hit beams are excluded;
  this is the representation consumed by the official evader repulsion term.
  """
  scan=self._raycast_lidar(); hit=np.argwhere(scan>1e-5)
  if hit.size==0: return np.zeros((0,3),np.float32)
  vi,hi=hit[:,0],hit[:,1]; pitch=np.deg2rad(np.linspace(-10.,20.,self.vertical,dtype=np.float32))[vi]; yaw=2.*np.pi*hi/self.horizontal+self.lidar_yaw
  direction=np.stack((np.cos(pitch)*np.cos(yaw),np.cos(pitch)*np.sin(yaw),np.sin(pitch)),axis=-1); distance=(10.-scan[vi,hi])[:,None]
  return (self.position[None,:]+direction*distance).astype(np.float32)
 def reset(self,pos):
  self._scene_lidar_epoch += 1; self._lidar_cache_epoch=-1; self._lidar_cache=None
  self.rotor_joint_indices=np.array([self.drone.get_dof_index(f'rotor_{i}_joint') for i in range(4)])
  self.drone.set_world_pose(position=np.asarray(pos),orientation=np.array([1.,0,0,0])); self.drone.set_linear_velocity(np.zeros(3)); self.drone.set_angular_velocity(np.zeros(3)); self.actuator.reset(); self.drone.set_joint_positions(np.zeros(4),joint_indices=self.rotor_joint_indices); self.drone.set_joint_velocities(self.actuator.rotor_speeds.astype(np.float32),joint_indices=self.rotor_joint_indices); self._read_physics_state()
class PSTOPursuitScene:
 def __init__(self,positions=None,physics_dt=1/240,policy_hz=30.,num_obstacles=18,seed=4348,randomize_resets=False,randomize_obstacles=False):
  from isaacsim.core.api import World
  self.physics_dt=physics_dt; self.policy_dt=1/policy_hz; self.physics_substeps=round(self.policy_dt/physics_dt); self.world=World(physics_dt=physics_dt,rendering_dt=self.policy_dt,stage_units_in_meters=1.)
  if positions is None: positions=((-3,2,2), (-3,-2,2), (0,0,2))
  self.vehicles=[PSTOVehicle(self.world,f'/World/Pursuit/Drone_{i}',physics_dt) for i in range(3)]
  from isaacsim.core.api.objects import FixedCuboid
  from pxr import UsdLux
  UsdLux.DomeLight.Define(self.world.stage,'/World/PursuitLight').CreateIntensityAttr(1200.)
  self.ground=FixedCuboid('/World/Pursuit/Ground','pursuit_ground',position=np.array([0.,0.,-.05]),scale=np.array([100.,100.,.1]),size=1.,color=np.array([.15,.15,.15])); self.world.scene.add(self.ground)
  self.obstacles=[]; self.obstacle_positions=[]; self.obstacle_scales=[]; rng=np.random.default_rng(seed)
  for oi in range(int(num_obstacles)):
   for _ in range(100):
    op=np.array([rng.uniform(-7.,7.),rng.uniform(-5.5,5.5),0.],np.float32)
    if all(np.linalg.norm(op[:2]-np.asarray(x)[:2])>1.5 for x in positions): break
   scale=np.array([rng.uniform(.35,.7),rng.uniform(.35,.7),rng.uniform(2.,3.5)],np.float32); op[2]=scale[2]/2.; self.obstacle_positions.append(op.copy()); self.obstacle_scales.append(scale.copy())
   obstacle=FixedCuboid(f'/World/Pursuit/Obstacle_{oi}',f'pursuit_obstacle_{oi}',position=op,scale=scale,size=1.,color=np.array([.28,.20,.12])); self.world.scene.add(obstacle); self.obstacles.append(obstacle)
  self.positions=np.asarray(positions,dtype=np.float32); self.initial_positions=self.positions.copy(); self.seed=int(seed); self.randomize_resets=bool(randomize_resets); self.randomize_obstacles=bool(randomize_obstacles); self.rng=np.random.default_rng(self.seed); self.obstacle_layout=[{'seed':self.seed,'positions':[op.tolist() for op in self.obstacle_positions]}]
  self.world.reset()
  for v,p in zip(self.vehicles,self.positions): v.reset(p)
  self.world.render()
 def reset(self,positions=None,randomize=None,seed=None,randomize_obstacles=None):
  if seed is not None:
   self.seed=int(seed); self.rng=np.random.default_rng(self.seed)
  do_randomize=self.randomize_resets if randomize is None else bool(randomize)
  do_obstacle_randomize=self.randomize_obstacles if randomize_obstacles is None else bool(randomize_obstacles)
  if positions is not None:self.positions=np.asarray(positions,dtype=np.float32)
  elif do_randomize:
   candidate=self.initial_positions.copy(); candidate[:,:2]+=self.rng.uniform(-.35,.35,size=(3,2)).astype(np.float32); candidate[:,2]=2.; self.positions=candidate
  if do_obstacle_randomize and self.obstacles:
   for obstacle,scale in zip(self.obstacles,self.obstacle_scales):
    for _ in range(100):
     candidate=np.array([self.rng.uniform(-7.,7.),self.rng.uniform(-5.5,5.5),float(scale[2])/2.],np.float32)
     if all(np.linalg.norm(candidate[:2]-p[:2])>1.5 for p in self.positions): break
    obstacle.set_world_pose(position=candidate)
    self.obstacle_positions[self.obstacles.index(obstacle)]=candidate
   self.obstacle_layout.append({'seed':self.seed,'positions':[op.tolist() for op in self.obstacle_positions]})
  self.world.reset()
  for v,p in zip(self.vehicles,self.positions):v.reset(p)
  if do_obstacle_randomize:
   for obstacle,op in zip(self.obstacles,self.obstacle_positions): obstacle.set_world_pose(position=op)
  self.world.render()
 def step(self,commands):
  commands=np.asarray(commands,dtype=np.float32).reshape(3,3)
  for _ in range(self.physics_substeps):
   for v,c in zip(self.vehicles,commands):v._apply_rotor_control(c)
   self.world.step(render=False)
  for v in self.vehicles:v._read_physics_state()
  for v in self.vehicles:
   v._scene_lidar_epoch += 1; v._lidar_cache_epoch=-1; v._lidar_cache=None
  return np.stack([v.position for v in self.vehicles])
 def close(self): self.world.clear_instance()
 def state_dict(self):
  """Serialize scene randomization state for deterministic checkpoint resume."""
  return {
   'seed': int(self.seed),
   'positions': np.asarray(self.positions, np.float32).copy(),
   'initial_positions': np.asarray(self.initial_positions, np.float32).copy(),
   'obstacle_positions': np.asarray(self.obstacle_positions, np.float32).copy(),
   'obstacle_scales': np.asarray(self.obstacle_scales, np.float32).copy(),
   'obstacle_layout': list(self.obstacle_layout),
   'rng_state': _portable_rng_state(self.rng),
  }
 def load_state_dict(self, state):
  """Restore scene RNG/layout after a checkpoint load without random draws."""
  if state is None:
   return
  self.seed = int(state.get('seed', self.seed))
  if state.get('positions') is not None:
   self.positions = np.asarray(state['positions'], np.float32).copy()
  if state.get('initial_positions') is not None:
   self.initial_positions = np.asarray(state['initial_positions'], np.float32).copy()
  if state.get('obstacle_positions') is not None:
   self.obstacle_positions = np.asarray(state['obstacle_positions'], np.float32).copy()
  if state.get('obstacle_scales') is not None:
   self.obstacle_scales = [np.asarray(x, np.float32).copy() for x in state['obstacle_scales']]
  if state.get('obstacle_layout') is not None:
   self.obstacle_layout = list(state['obstacle_layout'])
  if state.get('rng_state') is not None:
   self.rng.bit_generator.state = state['rng_state']
  for obstacle, op in zip(self.obstacles, self.obstacle_positions):
   obstacle.set_world_pose(position=np.asarray(op, np.float32))
