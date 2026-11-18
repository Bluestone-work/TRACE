"""frontier / task node 集成验证：真实 LiDAR 建图 + 连通分量提取。"""
import argparse, math, os, sys, time
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--steps", type=int, default=400)
parser.add_argument("--num_envs", type=int, default=1)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import torch
from isaaclab.scene import InteractiveScene
from isaaclab.sim import SimulationCfg, SimulationContext

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from trace.envs.explore_scene import ExploreSceneCfg, SCENE_SIZE, lidar_ray_dirs_w
from trace.mapping.occupancy_map import VoxelMapCfg, VoxelOccupancyMap
from trace.mapping.frontier import FrontierCfg, frontier_mask, label_components, extract_task_nodes

sim = SimulationContext(SimulationCfg(dt=1 / 60, device=args.device))
scene = InteractiveScene(ExploreSceneCfg(num_envs=args.num_envs, env_spacing=SCENE_SIZE[0] + 10.0))
sim.reset()

omap = VoxelOccupancyMap(scene.num_envs, VoxelMapCfg(
    size=(SCENE_SIZE[0], SCENE_SIZE[1], 6.0),
    origin=(-SCENE_SIZE[0] / 2, -SCENE_SIZE[1] / 2, 0.0),
    resolution=0.25, max_range=12.0), device=sim.device)
fcfg = FrontierCfg(adjacency=6, cluster=26, min_cluster_voxels=8, max_label_iters=256)
print(f"[OK] 地图 {omap.nx}x{omap.ny}x{omap.nz} = {omap.num_voxels} 体素/env", flush=True)

drones = [scene["drone0"], scene["drone1"]]
lidars = [scene["lidar0"], scene["lidar1"]]
env_org = scene.env_origins

print("\n  step | known  | frontier | 簇数 | 最大簇 | 精确 | CCL耗时", flush=True)
for step in range(args.steps):
    ang = 2 * math.pi * step / 200.0
    rad = 3.0 + 10.0 * min(1.0, step / 300.0)
    for i, d in enumerate(drones):
        sgn = 1.0 if i == 0 else -1.0
        root = d.data.default_root_state.clone()
        root[:, 0] = env_org[:, 0] + rad * math.cos(ang + sgn * math.pi)
        root[:, 1] = env_org[:, 1] + rad * math.sin(ang + sgn * math.pi) * sgn
        root[:, 2] = env_org[:, 2] + 1.5
        d.write_root_pose_to_sim(root[:, :7].contiguous())
        d.write_root_velocity_to_sim(torch.zeros_like(root[:, 7:13]))
    sim.step()
    scene.update(sim.get_physics_dt())

    org = torch.stack([l.data.pos_w - env_org for l in lidars], dim=1)
    hit = torch.stack([l.data.ray_hits_w - env_org.unsqueeze(1) for l in lidars], dim=1)
    dirs = torch.stack([lidar_ray_dirs_w(l) for l in lidars], dim=1)
    omap.update(org, hit, ray_dirs=dirs)

    if step % 80 == 0 or step == args.steps - 1:
        fm = frontier_mask(omap, fcfg)
        torch.cuda.synchronize() if omap.device.type == "cuda" else None
        t0 = time.time()
        labels, conv = label_components(fm, fcfg)
        torch.cuda.synchronize() if omap.device.type == "cuda" else None
        dt = (time.time() - t0) * 1000
        nodes = extract_task_nodes(omap, labels, fcfg, max_nodes=64)
        nn = int(nodes["num_nodes"][0])
        big = int(nodes["n_voxels"][0][0]) if nn else 0
        print(f"  {step:4d} | {float(omap.known_ratio()[0]):.4f} | {int(fm[0].sum()):8d} |"
              f" {nn:4d} | {big:6d} | {str(conv):7s} | {dt:6.1f} ms", flush=True)
        last = (fm, labels, nodes, conv)

fm, labels, nodes, conv = last
nn = int(nodes["num_nodes"][0])
assert nn > 0, "应提取到 task node"
assert int(fm[0].sum()) > 0, "应存在 frontier 体素"
print(f"\n[OK] 最终 known_ratio = {float(omap.known_ratio()[0]):.4f}", flush=True)
print(f"[OK] task node 数 = {nn}, CCL 收敛 = {conv}", flush=True)
v = nodes["n_voxels"][0][nodes["valid"][0]]
print(f"[OK] 前 5 簇体素数 = {[int(x) for x in v[:5]]}", flush=True)
p = nodes["pos"][0][nodes["valid"][0]]
print(f"[OK] 前 3 簇质心 = {[[round(float(c),2) for c in q] for q in p[:3]]}", flush=True)
print(f"[OK] 前 3 簇 n_unknown = {[int(x) for x in nodes['n_unknown'][0][:3]]}", flush=True)
print("[DONE] frontier 集成验证通过", flush=True)
os._exit(0)
