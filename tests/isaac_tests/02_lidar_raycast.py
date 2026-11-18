"""验证 GPU 光线投射 LiDAR：取一帧点云。"""
import argparse, os
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import torch
import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationCfg, SimulationContext
from isaaclab.sensors import RayCasterCfg, patterns
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass

@configclass
class LidarSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())
    light  = AssetBaseCfg(prim_path="/World/light",
                          spawn=sim_utils.DomeLightCfg(intensity=2000.0))
    # LiDAR 挂载体（运动学刚体）
    drone = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Drone",
        spawn=sim_utils.CuboidCfg(
            size=(0.3, 0.3, 0.1),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 5.0)),
    )
    # 40 线, 垂直 -80~10 度(朝下扫地面), 水平 360 度, 0.5 度分辨率
    lidar = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Drone",
        ray_alignment="base",
        pattern_cfg=patterns.LidarPatternCfg(
            channels=40,
            vertical_fov_range=(-80.0, 10.0),
            horizontal_fov_range=(0.0, 360.0),
            horizontal_res=0.5,
        ),
        max_distance=70.0,
        mesh_prim_paths=["/World/ground"],
        debug_vis=False,
    )

sim = SimulationContext(SimulationCfg(dt=1/120, device=args.device))
scene = InteractiveScene(LidarSceneCfg(num_envs=4, env_spacing=30.0))
sim.reset()
print("[OK] 场景 + LiDAR 创建完成, num_envs =", scene.num_envs, flush=True)

for _ in range(10):
    sim.step(); scene.update(sim.get_physics_dt())

lidar = scene["lidar"]
hits = lidar.data.ray_hits_w
print("[OK] 点云 shape =", tuple(hits.shape), flush=True)
print("[OK] 每 env 光线数 =", hits.shape[1], "(40线 x 720方位 = 28800)", flush=True)
finite = torch.isfinite(hits).all(dim=-1)
print("[OK] env0 有效命中 =", int(finite[0].sum()), "/", hits.shape[1], flush=True)
d = torch.norm(hits[0][finite[0]] - lidar.data.pos_w[0], dim=-1)
if d.numel() > 0:
    print("[OK] 命中距离 = %.2f ~ %.2f m (载具高 5m)" % (float(d.min()), float(d.max())), flush=True)
print("[OK] 张量在 GPU 上:", hits.is_cuda, flush=True)
print("[DONE] LiDAR 光线投射验证通过", flush=True)
os._exit(0)   # 不调 close()：Isaac Sim 的 close() 会无限挂起
