"""批量三态 log-odds 体素占据栅格（GPU, torch）。

算法与参数沿用 NavRL 的 map_manager（ros2/map_manager/include/map_manager/occupancyMap.cpp），
但重写为按 num_envs 批量化的张量实现，以便接 Isaac Lab 的并行环境。

三态语义（TRACE 文档 4.4 要求区分"没看到"和"看到了但没障碍"）:
    unknown    occ <  p_min_log          初始化为 p_min_log - unknown_flag
    free       p_min_log <= occ < p_occ_log
    occupied   occ >= p_occ_log

每次 update 的更新规则（hit 优先，与 NavRL 一致）:
    delta = p_hit_log  若该体素本帧被任一光线命中
          = p_miss_log 否则（仅被光线穿过）
    unknown 体素被 miss 命中时直接置为 p_min_log（free 先验）
    其余 occ = clamp(occ + delta, p_min_log, p_max_log)

注意一个有意的滞后：真实障碍体素从 unknown 起需要约 4 次 hit 才会越过 p_occ_log,
这是 NavRL 的噪声过滤设计。但 unknown -> known 在第一次观测就翻转，
所以探索覆盖率指标不会滞后。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


def _logit(p: float) -> float:
    return math.log(p / (1.0 - p))


@dataclass
class VoxelMapCfg:
    """体素地图配置。size/origin 单位为米，origin 是地图最小角。"""

    size: tuple[float, float, float] = (40.0, 40.0, 6.0)
    origin: tuple[float, float, float] = (-20.0, -20.0, 0.0)
    resolution: float = 0.2

    # log-odds 概率参数（NavRL cfg/map_param.yaml 的默认值）
    p_hit: float = 0.70
    p_miss: float = 0.35
    p_min: float = 0.12
    p_max: float = 0.97
    p_occ: float = 0.80
    unknown_flag: float = 0.01

    # 光线最大有效距离（米）。超出此距离的光线只清空自由空间，不标占据
    max_range: float = 15.0
    # 光线下采样步长：1 表示全用，k 表示每 k 根取一根（省显存）
    ray_stride: int = 1
    # 单次 update 内按光线分块，限制峰值显存
    ray_chunk: int = 16384


class VoxelOccupancyMap:
    """按环境批量化的三态体素占据栅格。

    地图张量形状 (num_envs, nx, ny, nz)，存 log-odds。
    多机共享同一张队伍地图（对应文档里的多机地图融合），
    update 时把各机的光线拼在一起即可。
    """

    def __init__(self, num_envs: int, cfg: VoxelMapCfg, device: str | torch.device = "cuda:0"):
        self.cfg = cfg
        self.num_envs = num_envs
        self.device = torch.device(device)

        res = cfg.resolution
        self.res = res
        self.nx = int(round(cfg.size[0] / res))
        self.ny = int(round(cfg.size[1] / res))
        self.nz = int(round(cfg.size[2] / res))
        self.num_voxels = self.nx * self.ny * self.nz

        self.origin = torch.tensor(cfg.origin, dtype=torch.float32, device=self.device)
        self.dims = torch.tensor([self.nx, self.ny, self.nz], dtype=torch.long, device=self.device)

        self.p_hit_log = _logit(cfg.p_hit)
        self.p_miss_log = _logit(cfg.p_miss)
        self.p_min_log = _logit(cfg.p_min)
        self.p_max_log = _logit(cfg.p_max)
        self.p_occ_log = _logit(cfg.p_occ)
        self.unknown_init = self.p_min_log - cfg.unknown_flag

        # 每根光线的采样点数：覆盖 max_range
        self.num_samples = int(math.ceil(cfg.max_range / res))

        self.log_odds = torch.full(
            (num_envs, self.nx, self.ny, self.nz),
            self.unknown_init, dtype=torch.float32, device=self.device,
        )

    # ---------------- 状态查询 ----------------

    @property
    def unknown_mask(self) -> torch.Tensor:
        return self.log_odds < self.p_min_log

    @property
    def occupied_mask(self) -> torch.Tensor:
        return self.log_odds >= self.p_occ_log

    @property
    def free_mask(self) -> torch.Tensor:
        return (self.log_odds >= self.p_min_log) & (self.log_odds < self.p_occ_log)

    @property
    def known_mask(self) -> torch.Tensor:
        return self.log_odds >= self.p_min_log

    def known_ratio(self) -> torch.Tensor:
        """每个环境的已知体素占比，形状 (num_envs,)。探索进度指标。"""
        return self.known_mask.flatten(1).float().mean(dim=1)

    def counts(self) -> dict[str, torch.Tensor]:
        return {
            "unknown": self.unknown_mask.flatten(1).sum(dim=1),
            "free": self.free_mask.flatten(1).sum(dim=1),
            "occupied": self.occupied_mask.flatten(1).sum(dim=1),
        }

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None:
            self.log_odds.fill_(self.unknown_init)
        else:
            self.log_odds[env_ids] = self.unknown_init

    # ---------------- 坐标与索引 ----------------

    def world_to_index(self, pts: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """世界坐标 -> 体素下标。返回 (idx, in_bounds)，idx 形状同 pts[..., :3]。"""
        idx = torch.floor((pts - self.origin) / self.res).long()
        in_bounds = ((idx >= 0) & (idx < self.dims)).all(dim=-1)
        return idx, in_bounds

    def _flat_index(self, env_idx: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        """(env, ix, iy, iz) -> 展平到 (num_envs*nx*ny*nz) 的线性下标。"""
        ix, iy, iz = idx[..., 0], idx[..., 1], idx[..., 2]
        return (env_idx * self.num_voxels) + (ix * self.ny + iy) * self.nz + iz

    # ---------------- 核心更新 ----------------

    @torch.no_grad()
    def update(
        self,
        origins: torch.Tensor,
        hits: torch.Tensor,
        ray_dirs: torch.Tensor | None = None,
    ) -> None:
        """用一帧 LiDAR 观测更新地图。

        Args:
            origins:  (E, 3) 或 (E, A, 3) —— 传感器世界坐标。
            hits:     (E, R, 3) 或 (E, A, R, 3) —— 命中点世界坐标，
                      直接传 ``RayCaster.data.ray_hits_w``。未命中为 inf。
            ray_dirs: 可选 (E, R, 3) 或 (E, A, R, 3) —— 单位方向向量。
                      给了它，未命中的光线会沿该方向把 max_range 内清成 free；
                      不给则这些方向保持 unknown（偏保守，探索率会被低估）。
        """
        if origins.dim() == 2:
            origins = origins.unsqueeze(1)
        if hits.dim() == 3:
            hits = hits.unsqueeze(1)
        if ray_dirs is not None and ray_dirs.dim() == 3:
            ray_dirs = ray_dirs.unsqueeze(1)

        E, A, R = hits.shape[0], hits.shape[1], hits.shape[2]
        assert E == self.num_envs, f"origins/hits 的 env 维 {E} != num_envs {self.num_envs}"

        s = max(1, int(self.cfg.ray_stride))
        if s > 1:
            hits = hits[:, :, ::s]
            if ray_dirs is not None:
                ray_dirs = ray_dirs[:, :, ::s]
            R = hits.shape[2]

        # 把 agent 维并入光线维：每根光线配一个原点
        hits = hits.reshape(E, A * R, 3)
        ray_org = origins.unsqueeze(2).expand(E, A, R, 3).reshape(E, A * R, 3)
        dirs = ray_dirs.reshape(E, A * R, 3) if ray_dirs is not None else None

        hit_cnt = torch.zeros(E * self.num_voxels, dtype=torch.int32, device=self.device)
        obs_cnt = torch.zeros(E * self.num_voxels, dtype=torch.int32, device=self.device)

        total = A * R
        chunk = max(1, int(self.cfg.ray_chunk))
        for beg in range(0, total, chunk):
            end = min(beg + chunk, total)
            self._accumulate(
                ray_org[:, beg:end], hits[:, beg:end],
                None if dirs is None else dirs[:, beg:end],
                hit_cnt, obs_cnt,
            )

        self._apply_log_odds(hit_cnt, obs_cnt)

    def _accumulate(
        self,
        org: torch.Tensor,      # (E, C, 3)
        hit: torch.Tensor,      # (E, C, 3)
        dirs: torch.Tensor | None,
        hit_cnt: torch.Tensor,
        obs_cnt: torch.Tensor,
    ) -> None:
        """对一块光线做采样并累加 hit/观测计数。"""
        E, C = org.shape[0], org.shape[1]
        res, mr = self.res, self.cfg.max_range

        finite = torch.isfinite(hit).all(dim=-1)                 # (E, C)
        vec = torch.where(finite.unsqueeze(-1), hit - org, torch.zeros_like(org))
        dist = vec.norm(dim=-1)                                  # (E, C)

        # 方向：命中光线用 (hit-org) 归一化；未命中光线用外部给的 ray_dirs
        safe = dist.clamp(min=1e-6)
        ray_d = vec / safe.unsqueeze(-1)
        if dirs is not None:
            ray_d = torch.where(finite.unsqueeze(-1), ray_d, dirs)
            usable = finite | torch.isfinite(dirs).all(dim=-1)
        else:
            usable = finite

        # 命中且在量程内 -> 标占据；超量程或未命中 -> 只清空到 max_range
        in_range = finite & (dist <= mr)
        carve = torch.where(in_range, dist, torch.full_like(dist, mr))

        # 沿光线等间隔采样自由空间。
        # 不留排除余量：若某采样点与命中点落在同一体素，靠 _apply_log_odds 的
        # hit 优先规则（hit_cnt>0 即取 p_hit_log）解决，不会误判为 free。
        # 早期版本用 t < carve - res/2 排除命中体素，结果在每个表面前留下一层
        # 永不被观测的 unknown 壳，会产生假 frontier 并低估覆盖率。
        t = (torch.arange(self.num_samples, device=self.device, dtype=torch.float32) + 0.5) * res
        pts = org.unsqueeze(2) + ray_d.unsqueeze(2) * t.view(1, 1, -1, 1)      # (E,C,S,3)
        free_ok = (t.view(1, 1, -1) < carve.unsqueeze(-1)) & usable.unsqueeze(-1)

        env_id = torch.arange(E, device=self.device).view(E, 1, 1).expand(E, C, self.num_samples)
        idx, ib = self.world_to_index(pts)
        sel = free_ok & ib
        if sel.any():
            flat = self._flat_index(env_id[sel], idx[sel])
            obs_cnt.scatter_add_(0, flat, torch.ones_like(flat, dtype=torch.int32))

        # 命中体素：观测 + 命中 各加一
        if in_range.any():
            env_id_h = torch.arange(E, device=self.device).view(E, 1).expand(E, C)
            idx_h, ib_h = self.world_to_index(hit)
            sel_h = in_range & ib_h
            if sel_h.any():
                flat_h = self._flat_index(env_id_h[sel_h], idx_h[sel_h])
                ones = torch.ones_like(flat_h, dtype=torch.int32)
                obs_cnt.scatter_add_(0, flat_h, ones)
                hit_cnt.scatter_add_(0, flat_h, ones)

    def _apply_log_odds(self, hit_cnt: torch.Tensor, obs_cnt: torch.Tensor) -> None:
        """按 NavRL 的 hit 优先规则写回 log-odds。"""
        occ = self.log_odds.view(-1)
        observed = obs_cnt > 0
        if not bool(observed.any()):
            return
        hit_any = hit_cnt > 0
        unknown = occ < self.p_min_log

        # unknown 体素仅被穿过（未命中）-> 直接给 free 先验
        to_free = observed & unknown & (~hit_any)
        occ[to_free] = self.p_min_log

        upd = observed & (~to_free)
        delta = torch.where(
            hit_any[upd],
            torch.full((int(upd.sum()),), self.p_hit_log, device=self.device),
            torch.full((int(upd.sum()),), self.p_miss_log, device=self.device),
        )
        occ[upd] = (occ[upd] + delta).clamp(self.p_min_log, self.p_max_log)
