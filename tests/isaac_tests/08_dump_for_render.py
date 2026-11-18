"""跑探索并导出 .npz 快照，供 Blender 侧渲染 3D 图。"""
import argparse, math, os, sys
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser()
parser.add_argument("--steps", type=int, default=300)
parser.add_argument("--out", type=str, default=None)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import torch
from isaaclab.scene import InteractiveScene
from isaaclab.sim import SimulationCfg, SimulationContext
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from trace.envs.explore_scene import ExploreSceneCfg, SCENE_SIZE, lidar_ray_dirs_w
from trace.mapping.occupancy_map import VoxelMapCfg, VoxelOccupancyMap
from trace.mapping.frontier import FrontierCfg, frontier_mask, label_components, extract_task_nodes
from trace.viz.dump_scene import dump_snapshot

out = args.out or os.path.join(ROOT, "runs", "viz", "snapshot.npz")
os.makedirs(os.path.dirname(out), exist_ok=True)
sim = SimulationContext(SimulationCfg(dt=1/60, device=args.device))
scene = InteractiveScene(ExploreSceneCfg(num_envs=1, env_spacing=SCENE_SIZE[0] + 10.0))
sim.reset()
omap = VoxelOccupancyMap(1, VoxelMapCfg(
    size=(SCENE_SIZE[0], SCENE_SIZE[1], 6.0),
    origin=(-SCENE_SIZE[0]/2, -SCENE_SIZE[1]/2, 0.0),
    resolution=0.25, max_range=12.0), device=sim.device)
fcfg = FrontierCfg(adjacency=6, cluster=26, min_cluster_voxels=8, method="cpu")
drones = [scene["drone0"], scene["drone1"]]; lidars = [scene["lidar0"], scene["lidar1"]]
eo = scene.env_origins; trails = [[] for _ in drones]

for step in range(args.steps):
    ang = 2*math.pi*step/200.0; rad = 3.0 + 10.0*min(1.0, step/250.0)
    for i, d in enumerate(drones):
        sgn = 1.0 if i == 0 else -1.0
        root = d.data.default_root_state.clone()
        root[:, 0] = eo[:, 0] + rad*math.cos(ang + sgn*math.pi)
        root[:, 1] = eo[:, 1] + rad*math.sin(ang + sgn*math.pi)*sgn
        root[:, 2] = eo[:, 2] + 1.5
        d.write_root_pose_to_sim(root[:, :7].contiguous())
        d.write_root_velocity_to_sim(torch.zeros_like(root[:, 7:13]))
    sim.step(); scene.update(sim.get_physics_dt())
    org = torch.stack([l.data.pos_w - eo for l in lidars], dim=1)
    hit = torch.stack([l.data.ray_hits_w - eo.unsqueeze(1) for l in lidars], dim=1)
    dirs = torch.stack([lidar_ray_dirs_w(l) for l in lidars], dim=1)
    omap.update(org, hit, ray_dirs=dirs)
    for i, d in enumerate(drones):
        p = (d.data.root_pos_w - eo)[0].detach().cpu().numpy()
        trails[i].append((float(p[0]), float(p[1]), float(p[2])))

fm = frontier_mask(omap, fcfg)
labels, _ = label_components(fm, fcfg)
nodes = extract_task_nodes(omap, labels, fcfg, max_nodes=64)
pos = [(d.data.root_pos_w - eo)[0].detach().cpu().numpy() for d in drones]
dump_snapshot(out, omap, 0, fmask=fm, task_nodes=nodes, drone_pos=pos, trails=trails, step=args.steps)
print(f"[OK] known_ratio={float(omap.known_ratio()[0]):.4f} "
      f"occupied={int(omap.occupied_mask[0].sum())} frontier={int(fm[0].sum())} "
      f"task={int(nodes['valid'][0].sum())}", flush=True)
print(f"[OK] 快照已导出: {out} ({os.path.getsize(out)/1e6:.1f} MB)", flush=True)
print("[DONE]", flush=True)
os._exit(0)
