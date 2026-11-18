"""占据栅格 / frontier / task graph 可视化（matplotlib，不经 RTX）。

为什么不用 Isaac 自带的可视化：本机 RTX 渲染器不可用 —— GUI、
--enable_cameras、--livestream 三条路径全部崩在 rtx.scenedb.plugin
（见 docs/ISAAC_3D_plan.md 坑 9）。而且对这个项目来说，直接画体素地图和
task graph 比看渲染画面信息量大得多。
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")                       # 无显示环境
import matplotlib.pyplot as plt
import numpy as np
import torch

# 三态配色：unknown 灰 / free 白 / occupied 黑 / frontier 橙
_CMAP = np.array([
    [0.55, 0.55, 0.58],   # 0 unknown
    [1.00, 1.00, 1.00],   # 1 free
    [0.12, 0.12, 0.14],   # 2 occupied
    [1.00, 0.55, 0.10],   # 3 frontier
])


def topdown_rgb(omap, env_idx: int = 0, z_range: tuple[float, float] = (0.75, 3.5),
                fmask: torch.Tensor | None = None) -> np.ndarray:
    """把一个 env 的体素地图投影成俯视 RGB 图 (nx, ny, 3)。

    z_range 默认跳过地面层——地面在 z≈0 处整片 occupied，
    直接全高度投影会把整张图糊成黑色。
    """
    res, oz = omap.res, omap.cfg.origin[2]
    z0 = max(0, int((z_range[0] - oz) / res))
    z1 = min(omap.nz, int((z_range[1] - oz) / res))

    occ = omap.occupied_mask[env_idx, :, :, z0:z1].any(dim=-1)
    free = omap.free_mask[env_idx, :, :, z0:z1].any(dim=-1)

    code = torch.zeros_like(occ, dtype=torch.long)          # 默认 unknown
    code[free] = 1
    code[occ] = 2                                            # occupied 优先显示
    if fmask is not None:
        fr = fmask[env_idx, :, :, z0:z1].any(dim=-1)
        code[fr] = 3                                         # frontier 最高优先
    return _CMAP[code.detach().cpu().numpy()]


def plot_frame(omap, env_idx: int = 0, fmask=None, task_nodes=None,
               drone_pos=None, trails=None, title: str = "",
               z_range: tuple[float, float] = (0.75, 3.5)):
    """画一帧：俯视地图 + frontier + task node + 无人机位置/轨迹。

    Args:
        task_nodes: extract_task_nodes() 的返回字典
        drone_pos:  (A,3) 当前各机位置（地图局部坐标，米）
        trails:     list[list[(x,y)]] 各机历史轨迹
    """
    rgb = topdown_rgb(omap, env_idx, z_range, fmask)
    ox, oy = omap.cfg.origin[0], omap.cfg.origin[1]
    extent = [ox, ox + omap.nx * omap.res, oy, oy + omap.ny * omap.res]

    fig, ax = plt.subplots(figsize=(7.2, 7.2), dpi=110)
    # imshow 的第一维是行(y)，所以转置
    ax.imshow(np.transpose(rgb, (1, 0, 2)), origin="lower", extent=extent, interpolation="nearest")

    if task_nodes is not None:
        v = task_nodes["valid"][env_idx].detach().cpu().numpy()
        p = task_nodes["pos"][env_idx].detach().cpu().numpy()
        nu = task_nodes["n_unknown"][env_idx].detach().cpu().numpy()
        if v.any():
            s = 20 + 260 * (nu[v] / max(nu[v].max(), 1e-6))
            ax.scatter(p[v, 0], p[v, 1], s=s, facecolors="none",
                       edgecolors="#d62728", linewidths=1.6, zorder=3,
                       label=f"task node ({int(v.sum())})")
            for r, (x, y) in enumerate(p[v][:, :2]):
                ax.text(x, y, str(r), color="#d62728", fontsize=7,
                        ha="center", va="center", zorder=4)

    if trails is not None:
        for t in trails:
            if len(t) > 1:
                a = np.array(t)
                ax.plot(a[:, 0], a[:, 1], lw=1.2, color="#1f77b4", alpha=0.75, zorder=2)

    if drone_pos is not None:
        d = np.asarray(drone_pos)
        ax.scatter(d[:, 0], d[:, 1], s=90, marker="^", color="#1f77b4",
                   edgecolors="white", linewidths=1.2, zorder=5, label="UAV")

    ax.set_xlim(extent[0], extent[1]); ax.set_ylim(extent[2], extent[3])
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    ax.set_title(title, fontsize=10)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    fig.tight_layout()
    return fig


def save_frame(path: str, *args, **kwargs) -> None:
    fig = plot_frame(*args, **kwargs)
    fig.savefig(path)
    plt.close(fig)
