"""把一帧探索状态导出成 .npz，供 Blender 侧渲染。

为什么要两段式：本机 Isaac 的 RTX 渲染器不可用（见 docs/ISAAC_3D_plan.md 坑 9），
改用 Blender(bpy) 离屏渲染出 3D 图。而 bpy 和 isaacsim 的依赖锁冲突，
必须装在两个 conda 环境里，所以用 .npz 做中间格式：

    conda activate trace_isaac   -> 跑仿真 + dump_snapshot()
    conda activate trace_render  -> python trace/viz/render3d.py snapshot.npz out.png
"""

from __future__ import annotations

import numpy as np
import torch


def _nz_indices(mask: torch.Tensor, env_idx: int, max_n: int | None = None) -> np.ndarray:
    """取某 env 的 True 体素下标 (N,3)。超过 max_n 则随机下采样。"""
    idx = mask[env_idx].nonzero(as_tuple=False)
    if max_n is not None and idx.shape[0] > max_n:
        sel = torch.randperm(idx.shape[0], device=idx.device)[:max_n]
        idx = idx[sel]
    return idx.detach().cpu().numpy().astype(np.int32)


def dump_snapshot(path: str, omap, env_idx: int = 0, fmask: torch.Tensor | None = None,
                  task_nodes: dict | None = None, drone_pos=None, trails=None,
                  max_frontier: int = 20000, step: int = 0) -> str:
    """导出一帧。frontier 体素可能上万，用 max_frontier 控制渲染规模。"""
    data = {
        "origin": np.asarray(omap.cfg.origin, dtype=np.float32),
        "res": np.float32(omap.res),
        "dims": np.asarray([omap.nx, omap.ny, omap.nz], dtype=np.int32),
        "known_ratio": np.float32(float(omap.known_ratio()[env_idx])),
        "step": np.int32(step),
        "occupied": _nz_indices(omap.occupied_mask, env_idx),
    }
    data["frontier"] = (_nz_indices(fmask, env_idx, max_frontier)
                        if fmask is not None else np.zeros((0, 3), np.int32))

    if task_nodes is not None:
        v = task_nodes["valid"][env_idx]
        data["task_pos"] = task_nodes["pos"][env_idx][v].detach().cpu().numpy().astype(np.float32)
        data["task_nunk"] = task_nodes["n_unknown"][env_idx][v].detach().cpu().numpy().astype(np.float32)
    else:
        data["task_pos"] = np.zeros((0, 3), np.float32)
        data["task_nunk"] = np.zeros((0,), np.float32)

    data["drone_pos"] = (np.asarray(drone_pos, dtype=np.float32) if drone_pos is not None
                         else np.zeros((0, 3), np.float32))
    if trails:
        T = max(len(t) for t in trails)
        arr = np.zeros((len(trails), T, 3), np.float32)
        for i, t in enumerate(trails):
            a = np.asarray(t, dtype=np.float32)
            arr[i, :len(a)] = a
            if len(a) < T:
                arr[i, len(a):] = a[-1]
        data["trails"] = arr
    else:
        data["trails"] = np.zeros((0, 0, 3), np.float32)

    np.savez_compressed(path, **data)
    return path
