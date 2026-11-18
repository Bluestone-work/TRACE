"""把 Isaac 场景的真几何体导出成 USD，并记录逐帧无人机位姿。

为什么要这一步：本机 Isaac 的 RTX 渲染管线起不来（docs/ISAAC_3D_plan.md 坑 9，
今天用 --enable_cameras 复验仍是 547ms 内段错误），所以拿不到 Isaac 自己渲的画面。
但场景的 USD 几何体是可以导出的 —— 导出后在 Blender 里渲真网格（地形、障碍、无人机），
画面内容就是 Isaac 场景本身，只是换了渲染器。

和 11_dump_sequence.py 的分工：11 导出体素地图（无人机"感知到的"世界），
这里导出场景几何（世界"真实"长什么样）。

用法:
    cd upstream/IsaacLab
    python -u ../../isaac_tests/12_export_usd.py --headless
"""
import argparse, os, sys
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--out", type=str, default=None)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import numpy as np
import torch
from isaaclab.scene import InteractiveScene
from isaaclab.sim import SimulationCfg, SimulationContext

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from trace.envs.explore_scene import ExploreSceneCfg, SCENE_SIZE

out_dir = args.out or os.path.join(ROOT, "runs", "viz", "usd")
os.makedirs(out_dir, exist_ok=True)

sim = SimulationContext(SimulationCfg(dt=1 / 60, device=args.device))
scene = InteractiveScene(ExploreSceneCfg(num_envs=1, env_spacing=SCENE_SIZE[0] + 10.0))
sim.reset()
for _ in range(5):
    sim.step(); scene.update(1 / 60)
print("[OK] 场景已建好", flush=True)

import omni.usd
from pxr import Usd, UsdGeom

stage = omni.usd.get_context().get_stage()
print(f"[OK] stage 根层 = {stage.GetRootLayer().identifier}", flush=True)

prims = [p for p in stage.Traverse() if p.IsA(UsdGeom.Mesh)]
print(f"[OK] stage 里 Mesh prim 数 = {len(prims)}", flush=True)
for p in prims[:12]:
    print(f"     {p.GetPath()}", flush=True)

usd_path = os.path.join(out_dir, "scene.usd")
flat = stage.Flatten()                      # 把引用全部内联，否则导出的只有一堆外部引用
flat.Export(usd_path)
print(f"[OK] USD 已导出: {usd_path} ({os.path.getsize(usd_path)/1e6:.1f} MB)", flush=True)

# 地形网格单独导出一份 .npz（顶点+面），Blender 侧万一 USD importer 不认就用这个兜底
eo = scene.env_origins
meta = {
    "env_origin": eo[0].detach().cpu().numpy().astype(np.float32),
    "scene_size": np.asarray(SCENE_SIZE, dtype=np.float32),
    "drone_init": np.asarray([(d.data.root_pos_w - eo)[0].detach().cpu().numpy()
                              for d in [scene["drone0"], scene["drone1"]]], dtype=np.float32),
}
np.savez(os.path.join(out_dir, "scene_meta.npz"), **meta)
print(f"[OK] 元数据: {out_dir}/scene_meta.npz  无人机初始位姿={meta['drone_init'].tolist()}", flush=True)
print("[DONE]", flush=True)
os._exit(0)
