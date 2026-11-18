"""每机独立 belief + 体素来源记录 + 通信模型。

为什么要这个模块：原来 `VoxelOccupancyMap` 是"每个 env 一张共享图、各机光线拼起来
一次写入"，等价于假设全队共享一份完美融合的地图。这让两件事不成立：

  1. Actor 的观测（task graph / g_topo）其实读的是全局信息，部署时拿不到；
  2. 创新点四的 `u_i = V(s) − V(M_i(s))` 没有地基 —— 共享图里无法区分
     "哪些体素是 i 独有的贡献"。

这里把 belief 拆成三层，slot 编号 = env_idx * A + agent_idx：

    own    (E*A)  只由本机光线更新。**这就是体素来源记录** ——
                  不是 bitmask，而是每机各自的 log-odds 证据强度，
                  所以"去掉 i 的观测"是去掉一份证据，而非翻一个标志位。
    recv   (E*A)  通过通信收到的他机证据，累积且持久（链路断了不丢）。
    fused  (E*A)  own ⊕ recv，**Actor 唯一可以读的东西**。按需合成。

    team   (E)    ⊕ 所有 own，仅训练期 critic / 奖励 / 评测可用。按需合成。

这里必须区分两类信息，混在一起会出错：

  **覆盖来源记录（单调）**：`own.known_mask` —— "a 是否观测过 v"。
  log-odds 下界被 clamp 在 p_min_log，所以一旦 known 就永远 known，
  天然单调。观测过这件事不会"取消"，边际覆盖只能建立在它上面。

  **占据状态估计（必须可更新）**：fused 的 log-odds —— "v 现在是空还是堵"。
  障碍会消失、也会误检，所以这一层必须能被新观测推翻。

早先版本两层都用 `max` 融合，结果过期障碍永远清不掉：recv 只增不减，
而 `fused = max(own, recv)` 让本机已经清成 free 的体素被旧的二手证据顶回占据。
现在的规则：

    recv 投递    覆盖式写入，只写发送方观测过的体素（新报告能推翻旧报告）
    fused 合成   一手优先：本机观测过的体素用自己的证据，其余才用收到的

覆盖式写入对重传仍然幂等（同样的数据写进去值不变），但允许降级，
所以队友把障碍清掉后，下一条消息就能把我这边的也清掉。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from trace.mapping.occupancy_map import VoxelMapCfg, VoxelOccupancyMap


@dataclass
class CommCfg:
    """机间通信模型。全部以控制步为时间单位。"""

    enabled: bool = True
    range_m: float = 25.0          # 超出此距离链路断开
    every: int = 5                 # 每隔多少控制步尝试发一次（地图+状态）
    drop_prob: float = 0.1         # 单次消息独立丢失概率
    delay_steps: int = 2           # 状态消息延迟（地图消息的"延迟"体现为 every 的间隔）
    max_age: int = 60              # 消息年龄归一化上限（也是"从未收到"的取值）


class _MapView:
    """给 frontier_mask / extract_task_nodes 用的轻量视图。

    那两个函数只需要 free/unknown/known 三个 mask 加几个几何参数，
    不需要完整的 VoxelOccupancyMap。这样就不必改 frontier.py。
    """

    __slots__ = ("log_odds", "cfg", "res", "nx", "ny", "nz", "origin", "dims",
                 "_p_min", "_p_occ", "num_voxels")

    def __init__(self, log_odds: torch.Tensor, ref: VoxelOccupancyMap):
        self.log_odds = log_odds
        self.cfg = ref.cfg
        self.res = ref.res
        self.nx, self.ny, self.nz = ref.nx, ref.ny, ref.nz
        self.origin = ref.origin               # extract_task_nodes 用它把体素下标换成米
        self.dims = ref.dims
        self.num_voxels = ref.num_voxels
        self._p_min = ref.p_min_log
        self._p_occ = ref.p_occ_log

    @property
    def unknown_mask(self) -> torch.Tensor:
        return self.log_odds < self._p_min

    @property
    def occupied_mask(self) -> torch.Tensor:
        return self.log_odds >= self._p_occ

    @property
    def free_mask(self) -> torch.Tensor:
        return (self.log_odds >= self._p_min) & (self.log_odds < self._p_occ)

    @property
    def known_mask(self) -> torch.Tensor:
        return self.log_odds >= self._p_min

    def known_ratio(self) -> torch.Tensor:
        return self.known_mask.flatten(1).float().mean(dim=1)


class MultiAgentBelief:
    """每机独立 belief + 来源记录 + 通信。

    Args:
        num_envs: 并行环境数 E
        num_agents: 每个环境的无人机数 A
        map_cfg: 体素地图配置（所有层共用）
        comm_cfg: 通信模型
    """

    def __init__(self, num_envs: int, num_agents: int, map_cfg: VoxelMapCfg,
                 comm_cfg: CommCfg | None = None, device: str | torch.device = "cuda:0"):
        self.E, self.A = int(num_envs), int(num_agents)
        self.S = self.E * self.A
        self.comm = comm_cfg or CommCfg()
        self.device = torch.device(device)

        # own：每机直接观测证据。这一张同时承担"体素来源记录"的职责。
        self.own = VoxelOccupancyMap(self.S, map_cfg, device=device)
        self.unknown_init = self.own.unknown_init
        self.p_min_log = self.own.p_min_log
        self.p_max_log = self.own.p_max_log

        # recv：收到的他机证据（已转成 evidence，>=0）。覆盖式写入，可被新报告推翻。
        self.recv_ev = torch.zeros_like(self.own.log_odds)
        # 上一步的团队已知掩码，用来算"本步新增"。(E, V) bool
        self._team_known_prev = torch.zeros((self.E, self.own.num_voxels),
                                            dtype=torch.bool, device=self.device)

        # 合成结果的缓冲区，避免每次重新分配
        self._fused_buf = torch.empty_like(self.own.log_odds)
        self._team_buf = torch.empty((self.E, self.own.nx, self.own.ny, self.own.nz),
                                     dtype=torch.float32, device=self.device)

        # 通信状态：消息年龄 (E,A,A)，age[e,a,b] = a 上次收到 b 的消息距今多少控制步
        self.msg_age = torch.full((self.E, self.A, self.A), float(self.comm.max_age),
                                  dtype=torch.float32, device=self.device)
        # a 持有的 b 的状态快照（相对坐标会在 env 侧算，这里存地图局部系的绝对量）
        self.peer_state = torch.zeros((self.E, self.A, self.A, 6), device=self.device)
        self.peer_valid = torch.zeros((self.E, self.A, self.A), dtype=torch.bool, device=self.device)
        # 状态消息延迟用的历史环形缓冲 (D+1, E, A, 6)
        D = max(0, int(self.comm.delay_steps))
        self._hist = torch.zeros((D + 1, self.E, self.A, 6), device=self.device)
        self._hist_ptr = 0
        self._tick = 0

    # ---------------- 证据与合成 ----------------

    def _evidence(self, log_odds: torch.Tensor) -> torch.Tensor:
        """log-odds -> 非负证据强度。unknown 为 0。"""
        return (log_odds - self.unknown_init).clamp(min=0.0)

    def fused(self) -> _MapView:
        """Actor 可读的 belief：一手优先（本机观测过的用自己的，其余用收到的）。

        形状 (E*A, nx, ny, nz)。一手优先而非取 max，是为了让占据估计可更新：
        本机把某体素重新观测成 free 之后，不该被旧的二手证据顶回占据。
        """
        ev = torch.where(self.own.known_mask, self._evidence(self.own.log_odds), self.recv_ev)
        torch.add(ev, self.unknown_init, out=self._fused_buf)
        self._fused_buf.clamp_(self.unknown_init, self.p_max_log)
        return _MapView(self._fused_buf, self.own)

    def fused_slice(self, slots: torch.Tensor) -> _MapView:
        """只合成给定 slot 的 belief，用于错开重算时省 CCL 开销。规则同 fused()。"""
        ev = torch.where(self.own.known_mask[slots],
                         self._evidence(self.own.log_odds[slots]), self.recv_ev[slots])
        return _MapView((ev + self.unknown_init).clamp(self.unknown_init, self.p_max_log), self.own)

    def team(self) -> _MapView:
        """训练期可用的全局地图：⊕ 所有 own。形状 (E, nx, ny, nz)。"""
        ev = self._evidence(self.own.log_odds).view(self.E, self.A, -1).amax(dim=1)
        torch.add(ev.view(self._team_buf.shape), self.unknown_init, out=self._team_buf)
        self._team_buf.clamp_(self.unknown_init, self.p_max_log)
        return _MapView(self._team_buf, self.own)

    # ---------------- 观测更新 ----------------

    @torch.no_grad()
    def update_from_rays(self, origins: torch.Tensor, hits: torch.Tensor,
                         ray_dirs: torch.Tensor | None = None) -> None:
        """用一帧各机 LiDAR 更新 own。

        Args:
            origins: (E, A, 3)
            hits:    (E, A, R, 3)
            ray_dirs:(E, A, R, 3) 或 None
        """
        E, A, R = hits.shape[0], hits.shape[1], hits.shape[2]
        assert (E, A) == (self.E, self.A), f"收到 (E,A)=({E},{A})，期望 ({self.E},{self.A})"
        # 把 (env, agent) 压成 slot 维：每个 slot 只吃自己那 R 根光线，
        # 总光线数不变，所以 _accumulate 的峰值显存和共享图版本一致。
        self.own.update(
            origins.reshape(self.S, 3),
            hits.reshape(self.S, R, 3),
            None if ray_dirs is None else ray_dirs.reshape(self.S, R, 3),
        )

    # ---------------- 覆盖量与边际贡献 ----------------

    def known_per_agent(self) -> torch.Tensor:
        """每机自身直接观测到的已知体素数 (E, A)。"""
        return self.own.known_mask.flatten(1).sum(dim=1).view(self.E, self.A).float()

    def team_known_count(self) -> torch.Tensor:
        """团队已知体素数 (E,)。等于各机 own 已知集合的并集大小。"""
        k = self.own.known_mask.view(self.E, self.A, -1)
        return k.any(dim=1).sum(dim=1).float()

    def exclusive_coverage(self) -> torch.Tensor:
        """**累计**独占覆盖 (E, A)：到目前为止只有 i 观测到的体素数。

        `= |team_known| - |team_known \\ i 的直接观测|`，精确值（不是近似），
        因为 known 的判据"至少一机观测过"让并集之差正好是独有体素数。

        这是一个**指标**，也是 `V(s) - V(M_i(s))` 的接口量，**不要直接当奖励**：
        它是存量，拿它做差分会在"队友重复观测了我原本独占的体素"时变成负数，
        把本步真实的新贡献一起吞掉。按步奖励请用 `stepwise_marginal()`。
        """
        k = self.own.known_mask.view(self.E, self.A, -1)
        cnt = k.sum(dim=1)                                  # (E, V) 每体素被几机看到
        only_i = k & (cnt == 1).unsqueeze(1)                # (E, A, V)
        return only_i.sum(dim=2).float()

    # 兼容旧名字（第一阶段的测试/日志在用）
    marginal_coverage = exclusive_coverage

    @torch.no_grad()
    def stepwise_marginal(self, commit: bool = True) -> torch.Tensor:
        """**本步**团队新增覆盖按贡献者均分到各机 (E, A)。奖励就该用这个。

        口径：只看"这一步第一次被团队知道"的体素；若同一步有 k 架机同时观测到它，
        每架记 1/k。于是严格守恒：

            Σ_i stepwise_marginal_i == Δ|team_known|

        守恒很重要 —— 它让这一项成为团队信息增益的一个分解（difference reward /
        aristocrat utility 的形式），而不是额外加在团队奖励之上的一笔，
        所以替换 r_info 时总奖励尺度不变，消融实验才不会被尺度差异污染。

        和存量差分的区别：历史覆盖不会被反复奖励（只算新增），
        而且队友重复观测我原本独占的体素时，不会把我本步的真实新贡献抹掉。

        Args:
            commit: True 时推进"上一步团队已知掩码"。一步只应调用一次。
        """
        k = self.own.known_mask.view(self.E, self.A, -1)
        team_now = k.any(dim=1)                             # (E, V)
        newly = team_now & (~self._team_known_prev)         # (E, V) 本步首次被团队知道
        out = torch.zeros((self.E, self.A), device=self.device)
        if bool(newly.any()):
            contrib = k & newly.unsqueeze(1)                # (E, A, V)
            cnt = contrib.sum(dim=1).clamp(min=1)           # (E, V) 同时发现的机数
            inv = torch.where(newly, 1.0 / cnt.float(), torch.zeros_like(cnt, dtype=torch.float32))
            for a in range(self.A):                         # 逐机算，避免 (E,A,V) 的大临时张量
                out[:, a] = (contrib[:, a].float() * inv).sum(dim=1)
        if commit:
            self._team_known_prev = team_now.clone()
        return out

    def coverage_without(self, agent: int) -> torch.Tensor:
        """去掉某机直接观测后的团队覆盖体素数 (E,)。给 V(M_i(s)) 用。"""
        k = self.own.known_mask.view(self.E, self.A, -1)
        keep = [j for j in range(self.A) if j != agent]
        if not keep:
            return torch.zeros(self.E, device=self.device)
        return k[:, keep].any(dim=1).sum(dim=1).float()

    # ---------------- 通信 ----------------

    @torch.no_grad()
    def step_comm(self, positions: torch.Tensor, velocities: torch.Tensor) -> torch.Tensor:
        """推进一个控制步的通信。返回本步实际投递的链路掩码 (E,A,A) bool。

        地图消息：投递时把 b 的 own 证据写入 a 的 recv（覆盖式写入，见模块注释）。
                  地图的"陈旧度"体现为距上次成功投递的步数，即 msg_age。
        状态消息：带 delay_steps 的延迟，投递的是 D 步前的快照。

        未投递的链路 msg_age 每步 +1，从未收到过的保持 max_age。
        这意味着通信中断时 Actor 看到的是"队友状态全零 + 年龄拉满"，
        它仍然能构造出完整观测并决策 —— 这是去中心化的硬要求。
        """
        D = self._hist.shape[0] - 1
        self._hist[self._tick % (D + 1)] = torch.cat([positions, velocities], dim=-1)

        self.msg_age += 1.0
        self.msg_age.clamp_(max=float(self.comm.max_age))

        delivered = torch.zeros((self.E, self.A, self.A), dtype=torch.bool, device=self.device)
        self._tick += 1
        if not self.comm.enabled or self.A < 2:
            return delivered
        if self.comm.every > 1 and (self._tick % self.comm.every) != 0:
            return delivered

        d = torch.cdist(positions, positions)                       # (E,A,A)
        in_range = d <= self.comm.range_m
        eye = torch.eye(self.A, dtype=torch.bool, device=self.device).unsqueeze(0)
        in_range = in_range & (~eye)
        if self.comm.drop_prob > 0.0:
            keep = torch.rand((self.E, self.A, self.A), device=self.device) >= self.comm.drop_prob
            delivered = in_range & keep
        else:
            delivered = in_range

        if not bool(delivered.any()):
            return delivered

        # 地图证据投递：覆盖式写入，只写发送方观测过的体素。
        # 不用 max —— max 单调，过期障碍永远清不掉（见模块注释）。
        ev = self._evidence(self.own.log_odds).view(self.E, self.A, -1)
        src_known = self.own.known_mask.view(self.E, self.A, -1)
        recv = self.recv_ev.view(self.E, self.A, -1)
        for b in range(self.A):
            idx = delivered[:, :, b].nonzero(as_tuple=False)
            if idx.numel() == 0:
                continue
            e_i, a_i = idx[:, 0], idx[:, 1]
            recv[e_i, a_i] = torch.where(src_known[e_i, b], ev[e_i, b], recv[e_i, a_i])

        # 状态消息（带延迟）
        old = self._hist[(self._tick - 1 - D) % (D + 1)]             # (E,A,6)
        for b in range(self.A):
            idx = delivered[:, :, b].nonzero(as_tuple=False)
            if idx.numel() == 0:
                continue
            e_i, a_i = idx[:, 0], idx[:, 1]
            self.peer_state[e_i, a_i, b] = old[e_i, b]
            self.peer_valid[e_i, a_i, b] = True
            self.msg_age[e_i, a_i, b] = float(D)                     # 收到时信息已经 D 步旧
        return delivered

    def comm_stats(self) -> dict[str, torch.Tensor]:
        # A=1 时没有非对角链路。对空张量直接求 mean 会得到 NaN，
        # 进而污染 TensorBoard/评测日志；单机实验应把“无队友”记为 0。
        if self.A < 2:
            return {
                "msg_age_mean": torch.zeros((), device=self.device),
                "peer_known_frac": torch.zeros((), device=self.device),
                "recv_evidence_frac": (self.recv_ev > 0).flatten(1).float().mean(),
            }
        off = ~torch.eye(self.A, dtype=torch.bool, device=self.device).unsqueeze(0)
        return {
            "msg_age_mean": self.msg_age[off.expand(self.E, -1, -1)].mean(),
            "peer_known_frac": self.peer_valid[off.expand(self.E, -1, -1)].float().mean(),
            "recv_evidence_frac": (self.recv_ev > 0).flatten(1).float().mean(),
        }

    # ---------------- 重置 ----------------

    @torch.no_grad()
    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None:
            self.own.reset(None)
            self.recv_ev.zero_()
            self.msg_age.fill_(float(self.comm.max_age))
            self.peer_state.zero_()
            self.peer_valid.zero_()
            self._hist.zero_()
            self._team_known_prev.zero_()
            return
        ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        # slot = env*A + agent，所以每个 env 对应连续 A 个 slot
        slots = (ids.unsqueeze(1) * self.A + torch.arange(self.A, device=self.device)).reshape(-1)
        self.own.reset(slots)
        self.recv_ev[slots] = 0.0
        self.msg_age[ids] = float(self.comm.max_age)
        self.peer_state[ids] = 0.0
        self.peer_valid[ids] = False
        self._hist[:, ids] = 0.0
        self._team_known_prev[ids] = False
