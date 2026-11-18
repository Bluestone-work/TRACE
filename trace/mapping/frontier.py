"""3D frontier 提取与连通分量标记 —— TRACE 的 task node 来源。

对应文档 3.1.1：C²-Explorer 用 Connected Component Labeling 把空间上不连通的
unknown 区域拆成独立 task unit。这里实现等价的 3D 版本：

    frontier 体素 := free 且 6(或26)-邻域中存在 unknown
    task unit     := frontier 体素的连通分量

CCL 用 GPU 上的迭代标签传播（每轮取邻域最大标签），避免每步把 61 万体素
同步回 CPU 跑 scipy。代价是迭代次数与连通分量的"直径"同阶，
细长结构收敛慢；用 max_label_iters 截断并在未收敛时给出警告。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

try:
    from scipy import ndimage
    _HAS_SCIPY = True
except ImportError:          # pragma: no cover
    ndimage = None
    _HAS_SCIPY = False


@dataclass
class FrontierCfg:
    """frontier 提取配置。"""

    # 判定 frontier 的邻接性：free 体素的邻域里有 unknown 才算 frontier。
    # 用 6（面相邻）是标准定义，26 会把斜对角的 unknown 也算进来，frontier 偏多。
    adjacency: int = 6
    # frontier 体素之间的聚类连通性。**必须用 26** ——
    # 6-连通会把数字化的曲面切碎：实测半径 7 m 的 frontier 环在 6-连通下
    # 被切成 32 个分量，26-连通下正确识别为 1 个。
    cluster: int = 26
    # 小于该体素数的簇视为噪声丢弃
    min_cluster_voxels: int = 8
    # 标签传播最大轮数（仅 method="gpu" 时用）
    max_label_iters: int = 256
    # CCL 实现："cpu" = scipy.ndimage.label（精确，默认）
    #           "gpu" = 迭代标签传播（免同步，但可能不收敛导致过分割）
    #           "auto" = 有 scipy 用 cpu，否则 gpu
    method: str = "cpu"
    # 只在该高度区间内找 frontier（米，地图局部坐标）。None 表示不限制
    z_range: tuple[float, float] | None = None


def _neighbor_offsets(connectivity: int, device) -> torch.Tensor:
    """返回邻域偏移 (K,3)，不含自身。"""
    if connectivity == 6:
        off = [(-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1)]
    elif connectivity == 26:
        off = [(i, j, k) for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)
               if not (i == 0 and j == 0 and k == 0)]
    else:
        raise ValueError(f"connectivity 只支持 6 或 26，收到 {connectivity}")
    return torch.tensor(off, dtype=torch.long, device=device)


def _dilate(mask: torch.Tensor, connectivity: int) -> torch.Tensor:
    """3D 形态学膨胀。mask: (E,nx,ny,nz) bool -> bool。"""
    x = mask.unsqueeze(1).float()
    if connectivity == 6:
        k = torch.zeros((1, 1, 3, 3, 3), device=mask.device)
        k[0, 0, 1, 1, :] = 1.0
        k[0, 0, 1, :, 1] = 1.0
        k[0, 0, :, 1, 1] = 1.0
    else:
        k = torch.ones((1, 1, 3, 3, 3), device=mask.device)
    return (F.conv3d(x, k, padding=1) > 0).squeeze(1)


@torch.no_grad()
def frontier_mask(omap, cfg: FrontierCfg = FrontierCfg()) -> torch.Tensor:
    """frontier 体素掩码 (E,nx,ny,nz) bool。

    定义：free 体素，且其邻域中存在 unknown 体素。
    用"膨胀 unknown 再与 free 相交"实现，等价且全程张量化。
    """
    free = omap.free_mask
    unknown = omap.unknown_mask
    fm = free & _dilate(unknown, cfg.adjacency)

    if cfg.z_range is not None:
        z0 = int(max(0, (cfg.z_range[0] - omap.cfg.origin[2]) / omap.res))
        z1 = int(min(omap.nz, (cfg.z_range[1] - omap.cfg.origin[2]) / omap.res))
        keep = torch.zeros(omap.nz, dtype=torch.bool, device=fm.device)
        keep[z0:z1] = True
        fm = fm & keep.view(1, 1, 1, -1)
    return fm


@torch.no_grad()
def label_components(mask: torch.Tensor, cfg: FrontierCfg = FrontierCfg()
                     ) -> tuple[torch.Tensor, bool]:
    """3D 连通分量标记。

    默认走 scipy.ndimage.label（精确）。GPU 迭代传播法在真实地图上
    256 轮经常不收敛——未收敛会把一个连通分量拆成多个标签，
    造成过分割，task unit 数量不可信。实测 160x160x24 的地图上
    scipy 路径约 30~60 ms，在文档设定的 1~2 Hz 高层决策频率下完全够用。

    Args:
        mask: (E,nx,ny,nz) bool

    Returns:
        labels:    (E,nx,ny,nz) int64，背景为 0，每个连通分量一个唯一正整数
                   （标签值全局唯一且不跨 env）
        exact:     是否为精确结果（scipy 路径恒为 True；GPU 路径为收敛标志）
    """
    method = cfg.method
    if method == "auto":
        method = "cpu" if _HAS_SCIPY else "gpu"
    if method == "cpu":
        if not _HAS_SCIPY:
            raise RuntimeError('method="cpu" 需要 scipy，请装 scipy 或改用 method="gpu"')
        return _label_scipy(mask, cfg), True
    return _label_gpu(mask, cfg)


def _label_scipy(mask: torch.Tensor, cfg: FrontierCfg) -> torch.Tensor:
    """scipy.ndimage.label 精确连通分量。逐 env 处理，标签全局唯一。"""
    E, nx, ny, nz = mask.shape
    V = nx * ny * nz
    structure = None
    if cfg.cluster == 6:
        structure = ndimage.generate_binary_structure(3, 1)   # 面相邻
    else:
        structure = ndimage.generate_binary_structure(3, 3)   # 26 邻域

    m_cpu = mask.detach().cpu().numpy()
    out = torch.zeros((E, nx, ny, nz), dtype=torch.long)
    for e in range(E):
        lab, n = ndimage.label(m_cpu[e], structure=structure)
        if n:
            t = torch.from_numpy(lab.astype("int64"))
            # 偏移保证标签不跨 env 冲突，且与 extract_task_nodes 的
            # env_of = (label-1)//V 反推逻辑一致
            out[e] = torch.where(t > 0, t + e * V, torch.zeros_like(t))
    labels = out.to(mask.device)
    if cfg.min_cluster_voxels > 1:
        labels = _drop_small(labels, cfg.min_cluster_voxels)
    return labels


@torch.no_grad()
def _label_gpu(mask: torch.Tensor, cfg: FrontierCfg) -> tuple[torch.Tensor, bool]:
    """GPU 迭代标签传播（免 CPU 同步，但可能不收敛）。"""
    E, nx, ny, nz = mask.shape
    dev = mask.device

    # 初始标签：每个前景体素取自己的全局线性下标 +1，背景为 0
    lin = torch.arange(E * nx * ny * nz, device=dev, dtype=torch.long).view(E, nx, ny, nz) + 1
    labels = torch.where(mask, lin, torch.zeros_like(lin))

    pad = 1
    converged = False
    for _ in range(cfg.max_label_iters):
        x = labels.unsqueeze(1).float()
        if cfg.cluster == 6:
            # 6-邻域最大值：分别沿三个轴做 1x1x3 / 1x3x1 / 3x1x1 的 max_pool
            m = x
            m = torch.maximum(m, F.max_pool3d(x, (3, 1, 1), 1, (pad, 0, 0)))
            m = torch.maximum(m, F.max_pool3d(x, (1, 3, 1), 1, (0, pad, 0)))
            m = torch.maximum(m, F.max_pool3d(x, (1, 1, 3), 1, (0, 0, pad)))
        else:
            m = F.max_pool3d(x, 3, 1, pad)
        new = m.squeeze(1).long()
        # 只在前景内传播，背景保持 0
        new = torch.where(mask, new, torch.zeros_like(new))
        if torch.equal(new, labels):
            converged = True
            break
        labels = new

    if cfg.min_cluster_voxels > 1:
        labels = _drop_small(labels, cfg.min_cluster_voxels)
    return labels, converged


def _drop_small(labels: torch.Tensor, min_voxels: int) -> torch.Tensor:
    """丢弃体素数不足的连通分量。"""
    flat = labels.view(-1)
    nz_mask = flat > 0
    if not bool(nz_mask.any()):
        return labels
    uniq, counts = torch.unique(flat[nz_mask], return_counts=True)
    keep = uniq[counts >= min_voxels]
    if keep.numel() == uniq.numel():
        return labels
    keep_set = torch.zeros(int(uniq.max()) + 1, dtype=torch.bool, device=labels.device)
    keep_set[keep] = True
    valid = torch.zeros_like(flat, dtype=torch.bool)
    valid[nz_mask] = keep_set[flat[nz_mask]]
    return torch.where(valid.view_as(labels), labels, torch.zeros_like(labels))


@torch.no_grad()
def extract_task_nodes(omap, labels: torch.Tensor, cfg: FrontierCfg = FrontierCfg(),
                       max_nodes: int = 64) -> dict[str, torch.Tensor]:
    """把连通分量汇总成 task node（文档 3.1.1 的 task unit）。

    Returns 字典，各张量第 0 维是 env，第 1 维按体素数降序填充到 max_nodes：
        pos        (E, M, 3) 代表点：簇内离质心最近的 frontier 体素，保证落在
                   frontier 上。**不要用质心当任务位置** —— 大的 frontier 环/壳
                   是一个连通分量，其质心落在环中心的已探明区域里，
                   会把无人机派去早就去过的地方（可视化里直接看到过这个 bug）
        centroid   (E, M, 3) 簇质心，仅作统计用
        n_voxels   (E, M)    簇内 frontier 体素数
        n_unknown  (E, M)    簇邻接的 unknown 体素计数（IG_m 的代理量）
        bbox_min   (E, M, 3) / bbox_max (E, M, 3) 包围盒（米）
        valid      (E, M)    bool，padding 位为 False
        num_nodes  (E,)      每个 env 的实际簇数（截断前）
    """
    E, nx, ny, nz = labels.shape
    V = nx * ny * nz
    dev = labels.device

    # 每个体素的"邻接 unknown 数"，作为 information gain 代理
    unk = omap.unknown_mask.unsqueeze(1).float()
    if cfg.adjacency == 6:
        k = torch.zeros((1, 1, 3, 3, 3), device=dev)
        k[0, 0, 1, 1, :] = 1.0; k[0, 0, 1, :, 1] = 1.0; k[0, 0, :, 1, 1] = 1.0
    else:
        k = torch.ones((1, 1, 3, 3, 3), device=dev)
    unk_adj = F.conv3d(unk, k, padding=1).squeeze(1)              # (E,nx,ny,nz)

    flat = labels.view(-1)
    fg = flat > 0
    out = {
        "pos": torch.zeros(E, max_nodes, 3, device=dev),
        "centroid": torch.zeros(E, max_nodes, 3, device=dev),
        "n_voxels": torch.zeros(E, max_nodes, device=dev),
        "n_unknown": torch.zeros(E, max_nodes, device=dev),
        "bbox_min": torch.zeros(E, max_nodes, 3, device=dev),
        "bbox_max": torch.zeros(E, max_nodes, 3, device=dev),
        "valid": torch.zeros(E, max_nodes, dtype=torch.bool, device=dev),
        "num_nodes": torch.zeros(E, dtype=torch.long, device=dev),
    }
    if not bool(fg.any()):
        return out

    uniq, inv = torch.unique(flat[fg], return_inverse=True)       # inv: (N,)
    K = uniq.numel()
    # 标签不跨 env（池化在 batch 维独立），可由标签值反推 env
    env_of = ((uniq - 1) // V).long()                             # (K,)

    # 体素中心的局部世界坐标
    lin_idx = torch.arange(E * V, device=dev)[fg]
    within = lin_idx % V
    ix = within // (ny * nz)
    iy = (within // nz) % ny
    iz = within % nz
    pts = torch.stack([ix, iy, iz], dim=-1).float()
    pts = omap.origin + (pts + 0.5) * omap.res                    # (N,3)

    cnt = torch.zeros(K, device=dev).scatter_add_(0, inv, torch.ones_like(inv, dtype=torch.float))
    ctr = torch.zeros(K, 3, device=dev)
    for d in range(3):
        ctr[:, d] = torch.zeros(K, device=dev).scatter_add_(0, inv, pts[:, d])
    ctr = ctr / cnt.unsqueeze(-1)

    uadj = unk_adj.view(-1)[fg]
    ig = torch.zeros(K, device=dev).scatter_add_(0, inv, uadj)

    # 代表点：簇内离质心最近的 frontier 体素（保证在 frontier 上）
    d2 = ((pts - ctr[inv]) ** 2).sum(dim=-1)
    min_d2 = torch.full((K,), float("inf"), device=dev).scatter_reduce(0, inv, d2, reduce="amin")
    is_rep = d2 <= min_d2[inv] + 1e-9
    rep_idx = torch.zeros(K, dtype=torch.long, device=dev)
    rep_idx.scatter_(0, inv[is_rep], torch.arange(inv.numel(), device=dev)[is_rep])
    rep = pts[rep_idx]                                            # (K,3)

    bmin = torch.full((K, 3), float("inf"), device=dev)
    bmax = torch.full((K, 3), float("-inf"), device=dev)
    for d in range(3):
        bmin[:, d] = bmin[:, d].scatter_reduce(0, inv, pts[:, d], reduce="amin")
        bmax[:, d] = bmax[:, d].scatter_reduce(0, inv, pts[:, d], reduce="amax")

    # 按 env 分组、按体素数降序，填充到 max_nodes
    for e in range(E):
        sel = (env_of == e).nonzero(as_tuple=True)[0]
        out["num_nodes"][e] = sel.numel()
        if sel.numel() == 0:
            continue
        order = torch.argsort(cnt[sel], descending=True)[:max_nodes]
        s = sel[order]
        n = s.numel()
        out["pos"][e, :n] = rep[s]
        out["centroid"][e, :n] = ctr[s]
        out["n_voxels"][e, :n] = cnt[s]
        out["n_unknown"][e, :n] = ig[s]
        out["bbox_min"][e, :n] = bmin[s]
        out["bbox_max"][e, :n] = bmax[s]
        out["valid"][e, :n] = True
    return out
