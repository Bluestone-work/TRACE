"""跑 NavRL 的 quick-demo 并存成 GIF。

上游三个 demo 结尾都是 plt.show()，本机无显示环境跑不了，而且我们不改
upstream/NavRL 的代码。这里用 monkeypatch 的方式接管：
  1. matplotlib 后端切 Agg
  2. 截获 FuncAnimation 实例
  3. 把 plt.show() 换成 ani.save(...)

用法:
    conda activate navrl
    python scripts/run_navrl_demo.py simple   -o runs/viz/navrl/simple.gif
    python scripts/run_navrl_demo.py random   -o runs/viz/navrl/random.gif
    python scripts/run_navrl_demo.py multi    -o runs/viz/navrl/multi.gif
"""
from __future__ import annotations

import argparse
import os
import runpy
import sys
import time

DEMOS = {
    "simple": "simple-navigation.py",
    "random": "random-navigation.py",
    "multi": "multi-robot-navigation.py",
}

ap = argparse.ArgumentParser()
ap.add_argument("demo", choices=sorted(DEMOS))
ap.add_argument("-o", "--out", default=None)
ap.add_argument("--frames", type=int, default=None, help="覆盖上游的 frames=300")
ap.add_argument("--fps", type=int, default=25)
ap.add_argument("--dpi", type=int, default=70)
ap.add_argument("--navrl", default=None, help="NavRL 仓库路径")
a = ap.parse_args()

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
navrl = a.navrl or os.path.join(ROOT, "upstream", "NavRL")
demo_dir = os.path.join(navrl, "quick-demos")
script = os.path.join(demo_dir, DEMOS[a.demo])
if not os.path.isfile(script):
    raise SystemExit(f"找不到 {script}")

out = a.out or os.path.join(ROOT, "runs", "viz", "navrl", f"{a.demo}.gif")
os.makedirs(os.path.dirname(out), exist_ok=True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation

_captured: list = []
_orig_init = animation.FuncAnimation.__init__


def _init(self, fig, func, *args, **kw):
    if a.frames is not None:
        kw["frames"] = a.frames
    _orig_init(self, fig, func, *args, **kw)
    _captured.append(self)


animation.FuncAnimation.__init__ = _init


def _show(*_args, **_kw):
    if not _captured:
        print("[WARN] 没截到 FuncAnimation，跳过存图", flush=True)
        return
    ani = _captured[-1]
    n = getattr(ani, "_save_count", None) or a.frames or 300
    print(f"[OK] 开始渲染 {n} 帧 -> {out}", flush=True)
    t0 = time.time()
    ani.save(out, writer=animation.PillowWriter(fps=a.fps), dpi=a.dpi)
    print(f"[OK] {time.time()-t0:.1f}s, {os.path.getsize(out)/1e6:.1f} MB -> {out}", flush=True)


plt.show = _show

# 上游脚本用 `from utils import ...` 这种同目录相对导入，必须进它的目录
os.chdir(demo_dir)
sys.path.insert(0, demo_dir)
print(f"[OK] 跑 {DEMOS[a.demo]}（cwd={demo_dir}）", flush=True)
runpy.run_path(script, run_name="__main__")
print("[DONE]", flush=True)
