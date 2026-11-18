"""占据栅格集成验证：真实 LiDAR 观测驱动三态体素地图。

两架无人机在带静态障碍的 40x40 场景里沿圆周飞行，每步用两机的 LiDAR
更新同一张队伍地图，观察 known_ratio 增长。
"""
import argparse, math, os, sys
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--steps", type=int, default=240)
parser.add_argument("--num_envs", type=int, default=1)  # 地形只有 1 个 sub-terrain，多 env 会重叠
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import torch
from isaaclab.scene import InteractiveScene
from isaaclab.sim import SimulationCfg, SimulationContext

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from trace.envs.explore_scene import ExploreSceneCfg, SCENE_SIZE, lidar_ray_dirs_w
from trace.mapping.occupancy_map import VoxelMapCfg, VoxelOccupancyMap

sim = SimulationContext(SimulationCfg(dt=1 / 60, device=args.device))
scene = InteractiveScene(ExploreSceneCfg(num_envs=args.num_envs, env_spacing=SCENE_SIZE[0] + 10.0))
sim.reset()
print(f"[OK] 场景就绪: num_envs={scene.num_envs}", flush=True)

map_cfg = VoxelMapCfg(
    size=(SCENE_SIZE[0], SCENE_SIZE[1], 6.0),
    origin=(-SCENE_SIZE[0] / 2, -SCENE_SIZE[1] / 2, 0.0),
    resolution=0.25,
    max_range=12.0,
)
omap = VoxelOccupancyMap(scene.num_envs, map_cfg, device=sim.device)
print(f"[OK] 体素地图: {omap.nx}x{omap.ny}x{omap.nz} = {omap.num_voxels} 体素/env", flush=True)

drones = [scene["drone0"], scene["drone1"]]
lidars = [scene["lidar0"], scene["lidar1"]]
# 每个 env 的世界原点偏移（env_spacing 导致各 env 不在同一位置）
env_org = scene.env_origins                     # (E,3)

print("\n  step | known_ratio |  free   | occupied | unknown", flush=True)
for step in range(args.steps):
    # 两机沿反向圆周飞行，半径随步数扩大
    ang = 2 * math.pi * step / 160.0
    rad = 3.0 + 9.0 * min(1.0, step / 180.0)
    for i, d in enumerate(drones):
        sgn = 1.0 if i == 0 else -1.0
        root = d.data.default_root_state.clone()
        root[:, 0] = env_org[:, 0] + rad * math.cos(ang + sgn * math.pi)
        root[:, 1] = env_org[:, 1] + rad * math.sin(ang + sgn * math.pi) * sgn
        root[:, 2] = env_org[:, 2] + 1.5
        d.write_root_pose_to_sim(root[:, :7].contiguous())
        d.write_root_velocity_to_sim(torch.zeros_like(root[:, 7:13]))  # 清零速度防漂移
    sim.step()
    scene.update(sim.get_physics_dt())

    # 组装两机观测 -> 同一张队伍地图（坐标转换到各 env 的局部系）
    org = torch.stack([l.data.pos_w - env_org for l in lidars], dim=1)        # (E,A,3)
    hit = torch.stack([l.data.ray_hits_w - env_org.unsqueeze(1) for l in lidars], dim=1)
    # 传入光线方向：打空(inf)的光线才能正确清出 max_range 内的自由空间
    dirs = torch.stack([lidar_ray_dirs_w(l) for l in lidars], dim=1)          # (E,A,R,3)
    omap.update(org, hit, ray_dirs=dirs)

    if step % 40 == 0 or step == args.steps - 1:
        c = omap.counts()
        print(f"  {step:4d} |   {float(omap.known_ratio()[0]):.4f}    | {int(c['free'][0]):7d} |"
              f" {int(c['occupied'][0]):8d} | {int(c['unknown'][0]):8d}", flush=True)

c = omap.counts()
assert int(c["occupied"][0]) > 0, "应探测到障碍体素"
assert float(omap.known_ratio()[0]) > 0.05, "覆盖率过低，映射可能失效"
print(f"\n[OK] 最终 known_ratio = {float(omap.known_ratio()[0]):.4f}", flush=True)
print(f"[OK] occupied = {int(c['occupied'][0])}, free = {int(c['free'][0])}", flush=True)
print(f"[OK] 各 env known_ratio = {[round(float(x),4) for x in omap.known_ratio()]}", flush=True)
print("[DONE] 占据栅格集成验证通过", flush=True)
os._exit(0)
