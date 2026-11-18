"""用原生 Isaac Sim（RTX）渲染 TRACE 场景的真几何体 -> PNG 序列 + GIF。

必须用独立安装的 Isaac Sim 6.0.1 跑，不能用 conda 里 pip 装的 5.1：
5.1 的 RTX 插件（omni.hydra.rtx-1.0.0+69cbf6ad）在本机启动即段错误，
6.0.1 的（1.0.4+f9bf0dda）正常。诊断见 docs/ISAAC_3D_plan.md。

输入:
    runs/viz/usd/scene.usd          12_export_usd.py 导出的场景几何
    runs/viz/seq3d/<tag>/*.npz      11_dump_sequence.py 导出的逐帧无人机位姿

用法:
    ~/isaacsim/python.sh scripts/render_isaac_native.py \
        --usd runs/viz/usd/scene.usd --seq runs/viz/seq3d/ckpt_agent_960 \
        --out runs/viz/isaac3d/ckpt_agent_960
"""
import argparse, glob, os, sys

ap = argparse.ArgumentParser()
ap.add_argument("--usd", required=True)
ap.add_argument("--seq", required=True, help="含 snap_*.npz 的目录")
ap.add_argument("--out", required=True)
ap.add_argument("--res", type=int, nargs=2, default=[1280, 720])
ap.add_argument("--warmup", type=int, default=24, help="每帧渲染前空转步数，让 RTX 收敛")
ap.add_argument("--fps", type=int, default=6)
ap.add_argument("--focal", type=float, default=23.0, help="焦距 mm，越小视野越广")
a = ap.parse_args()

from isaacsim import SimulationApp
sim_app = SimulationApp({"headless": True, "enable_cameras": True})
print("[OK] Isaac Sim 已启动（RTX, headless）", flush=True)

import numpy as np
import omni.usd
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdShade
from isaacsim.core.utils.stage import open_stage
from isaacsim.sensors.camera import Camera
import isaacsim.core.utils.numpy.rotations as rot_utils
from isaacsim.core.api import SimulationContext

files = sorted(glob.glob(os.path.join(a.seq, "snap_*.npz")))
if not files:
    raise SystemExit(f"{a.seq} 里没有 snap_*.npz")
os.makedirs(a.out, exist_ok=True)
print(f"[OK] {len(files)} 帧待渲", flush=True)

open_stage(os.path.abspath(a.usd))
stage = omni.usd.get_context().get_stage()
print(f"[OK] 已载入 {a.usd}", flush=True)

d0 = dict(np.load(files[0]))
span = d0["dims"].astype(np.float32) * float(d0["res"])
cx, cy = float(d0["origin"][0] + span[0] / 2), float(d0["origin"][1] + span[1] / 2)
n_agents = int(len(d0["drone_pos"]))


def _mk_marker(path, color):
    """无人机标记：球体。

    一开始用 UsdGeom.Cone，两个锥体渲出来一个正常一个成了扁片 —— 锥体
    在这个俯视角下会被压扁、朝向也不直观。球体没有朝向问题，稳定可辨。
    """
    sph = UsdGeom.Sphere.Define(stage, path)
    sph.CreateRadiusAttr(0.8)
    sph.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    UsdGeom.Xformable(sph.GetPrim()).AddTranslateOp()
    return sph.GetPrim()


COLORS = [(0.12, 0.47, 0.71), (0.90, 0.40, 0.10), (0.20, 0.65, 0.25), (0.60, 0.30, 0.75)]
markers = [_mk_marker(f"/World/trace_uav_{i}", COLORS[i % len(COLORS)]) for i in range(n_agents)]
trail_prim = UsdGeom.Points.Define(stage, "/World/trace_trail")
trail_prim.CreateDisplayColorAttr([Gf.Vec3f(0.20, 0.55, 0.85)])
# 导出的 USD 只带原场景的 DomeLight，渲出来地面和障碍几乎同色、看不出高度。
# 补一个方向光拉开明暗。
dl = UsdLux.DistantLight.Define(stage, "/World/trace_sun")
dl.CreateIntensityAttr(1500.0)
dl.CreateAngleAttr(0.5)
UsdGeom.Xformable(dl.GetPrim()).AddRotateXYZOp().Set(Gf.Vec3f(-26.0, 0.0, 212.0))
dome = UsdLux.DomeLight.Define(stage, "/World/trace_dome")
dome.CreateIntensityAttr(140.0)

# 导出的 USD 里带着原始的 drone/lidar 刚体，它们停在初始位姿 (±2,0,1.5) 不动，
# 和我们按帧摆的标记同时出现在画面里会让人误读成"无人机没动"。藏掉。
terrain = stage.GetPrimAtPath("/World/ground/terrain/mesh")
if terrain and terrain.IsValid():
    mat = UsdShade.Material.Define(stage, "/World/trace_mat")
    sh = UsdShade.Shader.Define(stage, "/World/trace_mat/shader")
    sh.CreateIdAttr("UsdPreviewSurface")
    sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.62, 0.63, 0.66))
    sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.65)
    mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI(terrain).Bind(mat)
    print("[OK] 地形已绑浅灰材质", flush=True)

n_hidden = 0
for prim in stage.Traverse():
    nm = prim.GetName()
    if nm.startswith("drone") or nm.startswith("lidar"):
        UsdGeom.Imageable(prim).MakeInvisible()
        n_hidden += 1
print(f"[OK] 已隐藏 {n_hidden} 个原始 drone/lidar prim", flush=True)
print(f"[OK] 已加 {n_agents} 个无人机标记 + 轨迹点 + 补光", flush=True)

diag = float(np.linalg.norm(span[:2]))
cam = Camera(prim_path="/World/trace_cam", resolution=tuple(a.res), frequency=20,
             position=np.array([cx - diag * 0.56, cy - diag * 0.56, diag * 0.40]),
             orientation=rot_utils.euler_angles_to_quats(np.array([0.0, 24.0, 45.0]), degrees=True))
# 默认 24mm 焦距视野太窄，40x40 的场地会被裁掉一角（第一版渲出来 drone0 就在框外）。
# 直接写 USD 相机属性，避免 Camera 包装层的单位歧义。
_c = UsdGeom.Camera(stage.GetPrimAtPath("/World/trace_cam"))
_c.GetFocalLengthAttr().Set(a.focal)
_c.GetHorizontalApertureAttr().Set(20.955)
sim = SimulationContext()
sim.initialize_physics()
cam.initialize()
for _ in range(30):
    sim.render()
print("[OK] 相机就绪", flush=True)

from PIL import Image
pngs = []
for k, f in enumerate(files):
    d = dict(np.load(f))
    for i, prim in enumerate(markers):
        p = d["drone_pos"][i]
        UsdGeom.Xformable(prim).GetOrderedXformOps()[0].Set(Gf.Vec3d(float(p[0]), float(p[1]), float(p[2])))
    tr = d["trails"]
    if tr.size:
        pts = tr.reshape(-1, 3)[::3]
        trail_prim.CreatePointsAttr([Gf.Vec3f(*map(float, q)) for q in pts])
        trail_prim.CreateWidthsAttr([0.45] * len(pts))
    for _ in range(a.warmup):
        sim.render()
    rgba = cam.get_rgba()
    out = os.path.join(a.out, os.path.basename(f).replace(".npz", ".png"))
    Image.fromarray(np.asarray(rgba)[:, :, :3].astype(np.uint8)).save(out)
    pngs.append(out)
    print(f"  [{k+1}/{len(files)}] step {int(d['step']):4d} known={float(d['known_ratio']):.3f} -> {os.path.basename(out)}",
          flush=True)

import imageio.v2 as imageio
gif = a.out.rstrip("/") + ".gif"
imageio.mimsave(gif, [imageio.imread(p) for p in pngs], duration=1.0 / a.fps, loop=0)
print(f"[OK] {len(pngs)} 帧 -> {gif} ({os.path.getsize(gif)/1e6:.1f} MB)", flush=True)
print("[DONE]", flush=True)
sim_app.close()
os._exit(0)
