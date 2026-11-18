"""把 11_dump_sequence.py 导出的一串 .npz 渲成 3D PNG 序列 + GIF（Blender/Cycles CPU）。

在 trace_render 环境里跑（不是 trace_isaac）:
    conda activate trace_render
    python trace/viz/render3d_seq.py runs/viz/seq3d/ckpt_agent_960 \
        runs/viz/seq3d/ckpt_agent_960_3d.gif --samples 24 --res 960 720

相机固定（由地图尺寸算出，逐帧一致），所以序列可直接当动画看。
不依赖 Vulkan/RTX —— 本机 Isaac 的 RTX 渲染管线起不来，见 docs/ISAAC_3D_plan.md 坑 9。
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bpy                                   # noqa: E402
from render3d import build_scene             # noqa: E402


def _label(path: str, text: str) -> None:
    """渲完在图上压一行说明文字（Blender 里加文字对象更麻烦）。"""
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        return
    im = Image.open(path).convert("RGB")
    d = ImageDraw.Draw(im)
    size = max(11, im.size[0] // 72)
    font = None
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        if os.path.exists(p):
            font = ImageFont.truetype(p, size)
            break
    d.rectangle([0, 0, im.size[0], int(size * 1.7)], fill=(255, 255, 255))
    d.text((int(size * 0.5), int(size * 0.3)), text, fill=(20, 20, 20), font=font)
    im.save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("in_dir", help="含 snap_*.npz 的目录")
    ap.add_argument("out_gif")
    ap.add_argument("--samples", type=int, default=24)
    ap.add_argument("--res", type=int, nargs=2, default=[960, 720])
    ap.add_argument("--frontier-max", type=int, default=12000)
    ap.add_argument("--duration", type=float, default=0.35, help="每帧秒数")
    ap.add_argument("--no-frontier", action="store_true")
    # render3d.py 里的相机是按单帧 1600x1200 调的，用它出序列会把地面裁掉一角，
    # 看不到全场就没法判断环境。这里允许把相机往后拉、换短焦。
    ap.add_argument("--cam-dist", type=float, default=0.62, help="相机水平距离 / 地图对角线")
    ap.add_argument("--cam-height", type=float, default=0.46, help="相机高度 / 地图对角线")
    ap.add_argument("--lens", type=float, default=34.0, help="焦距 mm，越小视野越广")
    a = ap.parse_args()

    files = sorted(glob.glob(os.path.join(a.in_dir, "snap_*.npz")))
    if not files:
        raise SystemExit(f"{a.in_dir} 里没有 snap_*.npz")
    png_dir = os.path.join(a.in_dir, "png")
    os.makedirs(png_dir, exist_ok=True)
    print(f"[OK] {len(files)} 帧待渲，{a.res[0]}x{a.res[1]} / {a.samples} samples", flush=True)

    pngs, t_all = [], time.time()
    for i, f in enumerate(files):
        d = dict(np.load(f))
        sc = build_scene(d, show_frontier=not a.no_frontier, frontier_max=a.frontier_max)

        span = d["dims"].astype(np.float32) * float(d["res"])
        cx, cy = d["origin"][0] + span[0] / 2, d["origin"][1] + span[1] / 2
        diag = float(np.linalg.norm(span[:2]))
        sc.camera.location = (cx - diag * a.cam_dist, cy - diag * a.cam_dist, diag * a.cam_height)
        sc.camera.data.lens = a.lens

        sc.render.engine = "CYCLES"
        sc.cycles.device = "CPU"
        sc.cycles.samples = a.samples
        sc.cycles.use_denoising = True
        sc.render.resolution_x, sc.render.resolution_y = a.res
        sc.render.film_transparent = False
        sc.render.image_settings.file_format = "PNG"
        out = os.path.join(png_dir, os.path.basename(f).replace(".npz", ".png"))
        sc.render.filepath = out
        t0 = time.time()
        bpy.ops.render.render(write_still=True)
        _label(out, f"step {int(d['step']):4d}   known={float(d['known_ratio']):.3f}   "
                    f"occupied={len(d['occupied'])}   frontier={len(d['frontier'])}   "
                    f"task nodes={len(d['task_pos'])}")
        pngs.append(out)
        print(f"  [{i+1}/{len(files)}] {os.path.basename(out)}  {time.time()-t0:.1f}s", flush=True)

    import imageio.v2 as imageio
    imageio.mimsave(a.out_gif, [imageio.imread(p) for p in pngs], duration=a.duration, loop=0)
    print(f"[OK] 全部 {len(pngs)} 帧 {time.time()-t_all:.0f}s -> {a.out_gif}", flush=True)
    print("[DONE]", flush=True)


if __name__ == "__main__":
    main()
