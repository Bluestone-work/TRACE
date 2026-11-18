"""TRACE-Explorer 多机探索环境（DirectMARLEnv）。

把 occupancy map -> frontier -> task graph 这条链接到 MARL 接口上，
对应文档 5.1 自己点明的"真正需要重点开发的部分"：Task Graph -> MARL Interface。

当前版本的定位与边界：
  已实现  三态占据栅格、3D frontier/task node、task graph 进 observation、
          CTDE 特权状态、文档 4.5 的奖励结构、两时间尺度（高层重算 task graph 频率
          与低层控制频率分离）、**每机独立 belief + 体素来源记录 + Actor 观测去中心化**
  占位    高层 task 选择用贪心启发式（n_unknown/dist 最大），
          这正是创新点三 Foresighted Task-Value Head 要替换的对象，
          留在这里作为对照 baseline
  未实现  动态障碍（需 MultiMeshRayCaster）、CBF 安全层、
          真实四旋翼动力学（现在是速度指令直接积分位姿）

去中心化契约（decentralized=True 时的硬约束）:
    Actor 观测的每一项都只能来自 本机感知 / 本机 belief(own ⊕ 收到的 recv) /
    已收到的通信消息。全局地图 self.omap 只允许出现在 critic 的 state、奖励、
    终止判定、日志和可视化里。切 decentralized=False 可退回原共享地图基线。

观测布局（每机 2 机时 164 维；去中心化下每个队友多 1 维 MsgAge）:
    [0:3]     ego 位置（相对地图中心，按地图尺寸归一化）
    [3:6]     ego 速度（use_true_velocity 时为位姿差分实测值，否则为指令）
    [6:42]    36 方位 LiDAR 最小距离环（按 lidar max_distance 归一化）
    [42:154]  task graph: 16 个 task node x 7 维，**由本机 belief 建图得到**
              [Δpos(3), 距离(1), n_voxels(1), n_unknown(1), valid(1)]
    [154:...] 队友：相对位置(3) + 相对速度(3) + MsgAge(1)，(A-1) 组。
              去中心化下这些是"上次收到的快照"，从未收到则为 0 且 MsgAge 拉满
    [末 3]    g_topo：本机任务选择结果的相对方向

特权状态（CTDE critic，109 维）:
    [0:12]    全部 agent 的位置+速度
    [12:13]   known_ratio
    [13:109]  task graph: 16 x [pos(3), n_voxels, n_unknown, valid]
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from isaaclab.envs import DirectMARLEnv, DirectMARLEnvCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

from trace.envs.explore_scene import (
    SCENE_SIZE,
    ExploreSceneCfg,
    lidar_ray_dirs_w,
    set_scene_agents,
)
from trace.mapping.belief import CommCfg, MultiAgentBelief
from trace.mapping.frontier import FrontierCfg, extract_task_nodes, frontier_mask, label_components
from trace.mapping.occupancy_map import VoxelMapCfg, VoxelOccupancyMap

N_LIDAR_BINS = 36
MAX_TASK_NODES = 16
MAX_SCENE_AGENTS = 4          # explore_scene.py 里定义了 4 套 droneN/lidarN
TASK_FEAT_DIM = 7
STATE_TASK_FEAT_DIM = 6


def _obs_dim(num_agents: int, decentralized: bool = False) -> int:
    """每机观测维度。

    去中心化模式下每个队友多一维 MsgAge —— 策略必须知道手里的队友状态有多旧，
    否则在真实异步通信下会把陈旧信息当实时信息用。
    """
    peer = 7 if decentralized else 6
    return 3 + 3 + N_LIDAR_BINS + MAX_TASK_NODES * TASK_FEAT_DIM + (num_agents - 1) * peer + 3


def _state_dim(num_agents: int) -> int:
    return num_agents * 6 + 1 + MAX_TASK_NODES * STATE_TASK_FEAT_DIM


@configclass
class TraceExploreEnvCfg(DirectMARLEnvCfg):
    """TRACE 探索环境配置。num_agents 支持 1~4，场景里备了 4 套 droneN/lidarN。"""

    num_agents: int = 2
    # 下面四项由 __post_init__ 按 num_agents 重算，这里只给 2 机的默认值占位
    possible_agents: list[str] = ["drone0", "drone1"]

    decimation: int = 4                    # 物理 60 Hz / 4 = 低层控制 15 Hz（文档 10~50 Hz）
    episode_length_s: float = 30.0
    sim: SimulationCfg = SimulationCfg(dt=1 / 60, render_interval=4)
    scene: ExploreSceneCfg = ExploreSceneCfg(num_envs=1, env_spacing=SCENE_SIZE[0] + 10.0)

    action_spaces: dict[str, int] = {"drone0": 3, "drone1": 3}
    observation_spaces: dict[str, int] = {"drone0": _obs_dim(2), "drone1": _obs_dim(2)}
    state_space: int = _state_dim(2)

    # ---- TRACE 专有 ----
    map_cfg: VoxelMapCfg = VoxelMapCfg(
        size=(SCENE_SIZE[0], SCENE_SIZE[1], 6.0),
        origin=(-SCENE_SIZE[0] / 2, -SCENE_SIZE[1] / 2, 0.0),
        resolution=0.25,
        max_range=12.0,
    )
    frontier_cfg: FrontierCfg = FrontierCfg(adjacency=6, cluster=26, min_cluster_voxels=8, method="cpu")

    # 两时间尺度：地图更新频率 与 task graph 重算频率（文档 4.2）
    # 必须为 1：奖励粒度要和建图节奏对齐。设为 2 时奇数步 Δ体素=0、r_info=0，
    # 只剩惩罚，实测恰好约 50% 的步为负、奖励中位数 -0.064，策略拿到的是噪声。
    map_update_every: int = 1               # 每个控制步建一次图 -> 15 Hz
    task_graph_every: int = 15              # 每 15 个控制步重算 task graph -> 1 Hz
    max_task_nodes: int = MAX_TASK_NODES

    # 运动与安全
    max_speed: float = 3.0                  # m/s
    z_range: tuple[float, float] = (0.8, 4.5)
    collision_dist: float = 0.6             # LiDAR 最小距离低于此值判碰撞

    # 奖励权重（文档 4.5：不宜过度堆叠）
    #
    # 尺度标定过程：原来 r_info = Δknown_ratio*100，随机策略下约 0.14/步，
    # 而两机相距 4 m 时去冗余惩罚 0.5*exp(-4/3) ≈ 0.13 —— 同量级，探索收益被抵消，
    # 实测平均奖励 -0.15 ~ +0.04，策略学不到"该去探索"。
    # 改为按新增已知体素数归一化（比 ratio 稳定、可解释）：
    # 实测一个有效步新增约 860 个体素，取 info_voxel_norm=800 使 r_info ≈ 1.0，
    # 同时把去冗余权重压到 0.2（最坏情况两机重合时惩罚 0.2），量级差 5 倍。
    w_info: float = 1.0
    info_voxel_norm: float = 800.0
    # r_info 上限。无人机进入新区域时一次更新可揭示两万多体素，
    # 实测单步奖励峰值 +33.4，方差过大不利于 PPO。截断保留信号、限制方差。
    info_clip: float = 5.0
    w_redundancy: float = 0.2
    w_collision: float = 10.0
    w_time: float = 0.01
    w_complete: float = 10.0                # 达到 done_known_ratio 的一次性奖励
    redundancy_sigma: float = 3.0           # 队友过近的去冗余惩罚尺度（米）
    done_known_ratio: float = 0.90           # 脚本化螺旋飞行实测可达 0.91，0.95 够不到

    # ---- 去中心化（本阶段新增）----
    # False = 原共享地图基线，一切行为与改造前一致，用于 ablation 对照。
    # True  = 每机独立 belief；Actor 的全部输入只由本机感知 + 收到的通信消息产生。
    decentralized: bool = True
    comm: CommCfg = CommCfg()
    # 只用于测量：decentralized=False 时也并行维护一份按机 belief，供 Overlap 之类
    # 需要"谁看到了什么"的指标使用。它绝不进入观测和奖励，只是为了让基线组
    # 也能算出同口径的重叠率 —— 否则四组实验的 Overlap 没法比。
    measure_per_agent: bool = False
    # 用实测速度（位姿差分）而不是速度指令。占位动力学下 root_lin_vel_w 恒为 0，
    # 而指令速度在撞到地图边界 clamp 时与实际位移不符 —— 喂指令会让策略看不见边界。
    use_true_velocity: bool = True
    # 信息项/去冗余项的口径：
    #   "overlap"  基线。所有机共享同一个团队信息增益 r_info，再各自吃一个
    #              基于机间距离的去冗余惩罚。
    #   "marginal" 把共享的 r_info **替换**成本机的边际新增覆盖（difference reward），
    #              并去掉距离惩罚（边际项已经承担了去冗余的职责）。
    #              因为 Σ_i marginal_i = Δ|team_known|，乘 A 后跨机均值与 r_info 相等，
    #              所以两种口径的信息项总体尺度一致，消融不会被尺度差异污染。
    reward_mode: str = "overlap"
    w_marginal: float = 1.0                  # marginal 口径的额外权重，默认 1 即尺度对齐
    # task graph 重算的事件触发阈值：本机 belief 自上次重算以来新增已知体素不足此数就跳过。
    # 精确 CCL 单次 130~175 ms，按机重算时开销乘 A，必须省。
    task_graph_min_new_voxels: int = 120
    # 各机错开重算（round-robin），把 CCL 开销摊到 task_graph_every 个控制步里
    task_graph_stagger: bool = True

    def __post_init__(self):
        """按 num_envs 铺地形网格，并按 num_agents 生成 agent 列表与空间维度。

        地形：terrain_type="generator" 时 env_origins 由 sub-terrain 原点决定、不看
        env_spacing；只有 1 个 sub-terrain 时所有 env 会叠在原点
        （见 docs/ISAAC_3D_plan.md 坑 8）。这里把网格开到够用。

        agent：场景里定义了 4 套 droneN/lidarN，多余的置 None 等于不 spawn。
        观测/动作/状态空间都按实际 num_agents 和 decentralized 重算，
        所以换 agent 数或切换去中心化开关不需要改任何别的地方。
        """
        n = self.scene.num_envs
        rows = int(math.ceil(math.sqrt(n)))
        cols = int(math.ceil(n / rows))
        self.scene.terrain.terrain_generator.num_rows = rows
        self.scene.terrain.terrain_generator.num_cols = cols

        A = int(self.num_agents)
        if not 1 <= A <= MAX_SCENE_AGENTS:
            raise ValueError(f"num_agents 必须在 1~{MAX_SCENE_AGENTS}，收到 {A}")
        self.possible_agents = [f"drone{i}" for i in range(A)]
        self.action_spaces = {n_: 3 for n_ in self.possible_agents}
        obs_d = _obs_dim(A, self.decentralized)
        self.observation_spaces = {n_: obs_d for n_ in self.possible_agents}
        self.state_space = _state_dim(A)
        set_scene_agents(self.scene, A)

        if self.reward_mode not in ("overlap", "marginal"):
            raise ValueError(f'reward_mode 只能是 "overlap" 或 "marginal"，收到 {self.reward_mode}')
        if self.reward_mode == "marginal" and not self.decentralized:
            # 边际覆盖要靠 per-agent own 图的来源记录算，共享图下无从区分
            raise ValueError('reward_mode="marginal" 需要 decentralized=True')


class TraceExploreEnv(DirectMARLEnv):
    """多机 3D 探索环境。动作为速度指令，task graph 通过观测进入策略。"""

    cfg: TraceExploreEnvCfg

    def __init__(self, cfg: TraceExploreEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.A = cfg.num_agents
        self.drones = [self.scene[f"drone{i}"] for i in range(self.A)]
        self.lidars = [self.scene[f"lidar{i}"] for i in range(self.A)]

        self.dec = bool(cfg.decentralized)
        # 测量用的影子 belief：只在基线路径 + measure_per_agent 时存在
        self.shadow = None
        if (not self.dec) and bool(cfg.measure_per_agent):
            self.shadow = MultiAgentBelief(self.num_envs, self.A, cfg.map_cfg,
                                           CommCfg(enabled=False), device=self.device)
        if self.dec:
            # 每机独立 belief。self.omap 指向"团队地图视图"，仅训练期 critic/奖励/
            # 评测/可视化可读；Actor 一律走 self.belief.fused()。
            self.belief = MultiAgentBelief(self.num_envs, self.A, cfg.map_cfg,
                                           cfg.comm, device=self.device)
            self.omap = self.belief.team()
        else:
            self.belief = None
            self.omap = VoxelOccupancyMap(self.num_envs, cfg.map_cfg, device=self.device)
        self._ctrl_dt = cfg.sim.dt * cfg.decimation
        self._step_ctr = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        # 建图/任务图的调度用全局 tick。早先用 self._step_ctr[0]（env 0 的计数器），
        # env 0 一重置就把所有 env 的节奏打乱了。
        self._tick = 0

        M = cfg.max_task_nodes
        # 团队 task graph：特权信息，只进 critic 的 state 和日志，不进 Actor 观测
        self._task_pos = torch.zeros(self.num_envs, M, 3, device=self.device)
        self._task_nvox = torch.zeros(self.num_envs, M, device=self.device)
        self._task_nunk = torch.zeros(self.num_envs, M, device=self.device)
        self._task_valid = torch.zeros(self.num_envs, M, dtype=torch.bool, device=self.device)
        # 每机从自己 belief 建的 task graph：这才是 Actor 观测的来源
        self._ltask_pos = torch.zeros(self.num_envs, self.A, M, 3, device=self.device)
        self._ltask_nvox = torch.zeros(self.num_envs, self.A, M, device=self.device)
        self._ltask_nunk = torch.zeros(self.num_envs, self.A, M, device=self.device)
        self._ltask_valid = torch.zeros(self.num_envs, self.A, M, dtype=torch.bool,
                                        device=self.device)
        # 每机当前选中的 task（贪心启发式，待创新点三替换）
        self._sel_task = torch.zeros(self.num_envs, self.A, 3, device=self.device)
        # 事件触发用：各机上次重算 task graph 时自己 belief 的已知体素数
        self._last_graph_known = torch.zeros(self.num_envs, self.A, device=self.device)
        # 位姿差分测速（占位动力学下 root_lin_vel_w 恒为 0）
        self._prev_pos = torch.zeros(self.num_envs, self.A, 3, device=self.device)
        self._true_vel = torch.zeros(self.num_envs, self.A, 3, device=self.device)

        self._prev_known = torch.zeros(self.num_envs, device=self.device)   # 已知体素数（个）
        self._vel_cmd = torch.zeros(self.num_envs, self.A, 3, device=self.device)
        self._collided = torch.zeros(self.num_envs, self.A, dtype=torch.bool, device=self.device)
        # DirectMARLEnv 会在计算 done 后立即 reset 环境；评估脚本需要这些
        # “终止前一帧”的值，不能再从 reset 后的地图反推。
        self._last_done_collision = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._last_done_explored = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._last_done_timeout = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._last_done_known_ratio = torch.zeros(self.num_envs, device=self.device)
        self._last_done_overlap_rate = torch.zeros(self.num_envs, device=self.device)

        self._map_span = torch.tensor(cfg.map_cfg.size, device=self.device)

    def _setup_scene(self):
        # scene 由 cfg.scene（ExploreSceneCfg）构建，这里只需保留 InteractiveScene 默认行为
        pass

    # ---------------- 动作 ----------------

    def _pre_physics_step(self, actions: dict[str, torch.Tensor]) -> None:
        """动作为 [-1,1]^3 的速度指令，缩放到 max_speed。"""
        for i, name in enumerate(self.cfg.possible_agents):
            a = actions[name].clamp(-1.0, 1.0)
            self._vel_cmd[:, i] = a * self.cfg.max_speed

    def _apply_action(self) -> None:
        """速度指令直接积分位姿（占位动力学，后续换真实四旋翼）。"""
        for i, d in enumerate(self.drones):
            root = d.data.root_state_w.clone()
            new_pos = root[:, :3] + self._vel_cmd[:, i] * self.cfg.sim.dt
            # 限制在地图范围与高度带内
            lo = self.scene.env_origins + torch.tensor(
                [self.cfg.map_cfg.origin[0] + 1.0, self.cfg.map_cfg.origin[1] + 1.0,
                 self.cfg.z_range[0]], device=self.device)
            hi = self.scene.env_origins + torch.tensor(
                [self.cfg.map_cfg.origin[0] + self.cfg.map_cfg.size[0] - 1.0,
                 self.cfg.map_cfg.origin[1] + self.cfg.map_cfg.size[1] - 1.0,
                 self.cfg.z_range[1]], device=self.device)
            root[:, :3] = new_pos.clamp(lo, hi)
            d.write_root_pose_to_sim(root[:, :7].contiguous())
            d.write_root_velocity_to_sim(torch.zeros_like(root[:, 7:13]))

    # ---------------- 建图与任务图 ----------------

    def _local_lidar(self):
        """返回 (origins (E,A,3), hits (E,A,R,3), dirs (E,A,R,3))，均为地图局部坐标。"""
        eo = self.scene.env_origins
        org = torch.stack([l.data.pos_w - eo for l in self.lidars], dim=1)
        hit = torch.stack([l.data.ray_hits_w - eo.unsqueeze(1) for l in self.lidars], dim=1)
        dirs = torch.stack([lidar_ray_dirs_w(l) for l in self.lidars], dim=1)
        return org, hit, dirs

    def _update_map(self) -> None:
        org, hit, dirs = self._local_lidar()
        if not self.dec:
            self.omap.update(org, hit, ray_dirs=dirs)
            if self.shadow is not None:
                self.shadow.update_from_rays(org, hit, ray_dirs=dirs)   # 只为指标
            return
        # 每机只用自己的光线更新自己的 own 图 —— own 同时就是体素来源记录
        self.belief.update_from_rays(org, hit, ray_dirs=dirs)
        self.belief.step_comm(self._agent_pos(), self._true_vel)
        self.omap = self.belief.team()          # 刷新团队视图（训练期专用）

    def _agent_pos(self) -> torch.Tensor:
        """各机位置，地图局部坐标 (E,A,3)。"""
        eo = self.scene.env_origins
        return torch.stack([d.data.root_pos_w - eo for d in self.drones], dim=1)

    def _update_true_velocity(self) -> None:
        """位姿差分测速。

        占位动力学每步把刚体速度写零，所以 root_lin_vel_w 恒为 0；而速度指令在
        撞到地图边界被 clamp 时与实际位移不符。用差分得到的是真实达成速度，
        既是部署时能测到的量，也让策略能"感觉到"边界墙。
        """
        pos = self._agent_pos()
        self._true_vel = (pos - self._prev_pos) / self._ctrl_dt
        self._prev_pos = pos.clone()

    def _rebuild_task_graph(self) -> None:
        """重算团队 task graph（特权信息）。开销约 130~175 ms。

        去中心化模式下这张图只进 critic 的 state 和日志；Actor 用的是
        `_rebuild_local_graphs()` 产出的按机图。
        """
        fm = frontier_mask(self.omap, self.cfg.frontier_cfg)
        labels, _ = label_components(fm, self.cfg.frontier_cfg)
        nodes = extract_task_nodes(self.omap, labels, self.cfg.frontier_cfg,
                                   max_nodes=self.cfg.max_task_nodes)
        self._task_pos = nodes["pos"]
        self._task_nvox = nodes["n_voxels"]
        self._task_nunk = nodes["n_unknown"]
        self._task_valid = nodes["valid"]
        if not self.dec:
            self._select_tasks()

    def _graph_due(self) -> torch.Tensor:
        """返回本步该重算 task graph 的 slot 下标 (K,)，slot = env*A + agent。

        两级省开销：
          1. 错开（stagger）：agent a 只在 tick % every == a*every//A 附近轮到，
             把 A 次 CCL 摊到 task_graph_every 个控制步里，而不是同一步全做。
          2. 事件触发：自上次重算以来本机 belief 新增已知体素不足阈值就跳过 ——
             没新信息就没有新 frontier，重算纯属浪费。
        """
        A, ev = self.A, max(1, int(self.cfg.task_graph_every))
        if self.cfg.task_graph_stagger and A > 1:
            phase = self._tick % ev
            agents = [a for a in range(A) if (a * ev) // A == phase]
        else:
            agents = list(range(A)) if (self._tick % ev) == 0 else []
        if not agents:
            return torch.empty(0, dtype=torch.long, device=self.device)

        known = self.belief.known_per_agent()                       # (E,A)
        first = self._ltask_valid.any(dim=2).logical_not()          # 还没建过图的
        grew = (known - self._last_graph_known) >= float(self.cfg.task_graph_min_new_voxels)
        due = torch.zeros(self.num_envs, A, dtype=torch.bool, device=self.device)
        due[:, agents] = True
        due = due & (grew | first)
        idx = due.nonzero(as_tuple=False)
        if idx.numel() == 0:
            return torch.empty(0, dtype=torch.long, device=self.device)
        return idx[:, 0] * A + idx[:, 1]

    def _rebuild_local_graphs(self) -> None:
        """按机从各自 belief（own ⊕ recv）重建 task graph。Actor 观测的唯一来源。"""
        slots = self._graph_due()
        if slots.numel() == 0:
            return
        view = self.belief.fused_slice(slots)
        fm = frontier_mask(view, self.cfg.frontier_cfg)
        labels, _ = label_components(fm, self.cfg.frontier_cfg)
        nodes = extract_task_nodes(view, labels, self.cfg.frontier_cfg,
                                   max_nodes=self.cfg.max_task_nodes)
        e_i, a_i = slots // self.A, slots % self.A
        self._ltask_pos[e_i, a_i] = nodes["pos"]
        self._ltask_nvox[e_i, a_i] = nodes["n_voxels"]
        self._ltask_nunk[e_i, a_i] = nodes["n_unknown"]
        self._ltask_valid[e_i, a_i] = nodes["valid"]
        self._last_graph_known[e_i, a_i] = self.belief.known_per_agent()[e_i, a_i]
        self._select_tasks_local(e_i, a_i)

    def _select_tasks_local(self, e_i: torch.Tensor, a_i: torch.Tensor) -> None:
        """本机任务选择：只用自己的 task graph 和自己的位置。

        规则与中心化版 `_select_tasks()` 完全一致（效用 = n_unknown/(距离+1)），
        区别只在"读的是谁的图"。注意原 `_select_tasks()` 本来也是每机各自贪心，
        中心化体现在它读共享地图，而不是做了全局分配 —— 所以这里不改规则，
        只把输入换成本机 belief，便于与基线做干净对照。
        """
        pos = self._agent_pos()[e_i, a_i]                           # (K,3)
        tp = self._ltask_pos[e_i, a_i]                              # (K,M,3)
        dist = (tp - pos.unsqueeze(1)).norm(dim=-1)
        util = self._ltask_nunk[e_i, a_i] / (dist + 1.0)
        valid = self._ltask_valid[e_i, a_i]
        util = torch.where(valid, util, torch.full_like(util, -1.0))
        best = util.argmax(dim=1)
        chosen = tp[torch.arange(tp.shape[0], device=self.device), best]
        has = valid.any(dim=1, keepdim=True)
        self._sel_task[e_i, a_i] = torch.where(has, chosen, pos)

    def _select_tasks(self) -> None:
        """贪心高层选择：效用 = n_unknown / (距离 + eps)。

        这是创新点三 Foresighted Task-Value Head 的替换对象。
        换成学习式只需把这里改成 argmax_m Q_{i,m}^exp（文档 4.2）。
        """
        eo = self.scene.env_origins
        for i, d in enumerate(self.drones):
            p = (d.data.root_pos_w - eo).unsqueeze(1)                 # (E,1,3)
            dist = (self._task_pos - p).norm(dim=-1)                  # (E,M)
            util = self._task_nunk / (dist + 1.0)
            util = torch.where(self._task_valid, util, torch.full_like(util, -1.0))
            best = util.argmax(dim=1)                                 # (E,)
            chosen = self._task_pos[torch.arange(self.num_envs, device=self.device), best]
            has = self._task_valid.any(dim=1, keepdim=True)
            # 没有可用 task 时退化为保持当前位置
            self._sel_task[:, i] = torch.where(has, chosen, p.squeeze(1))

    def _lidar_ring(self, agent_idx: int) -> torch.Tensor:
        """把一机的点云压成 36 方位的最小距离环（文档 4.4 的角度分区思想）。"""
        l = self.lidars[agent_idx]
        hits = l.data.ray_hits_w                                      # (E,R,3)
        pos = l.data.pos_w.unsqueeze(1)
        v = hits - pos
        d = v.norm(dim=-1)
        finite = torch.isfinite(d)
        maxd = self.cfg.map_cfg.max_range
        d = torch.where(finite, d.clamp(max=maxd), torch.full_like(d, maxd))
        az = torch.atan2(v[..., 1], v[..., 0])
        az = torch.where(torch.isfinite(az), az, torch.zeros_like(az))
        bin_id = ((az + math.pi) / (2 * math.pi) * N_LIDAR_BINS).long().clamp(0, N_LIDAR_BINS - 1)
        ring = torch.full((self.num_envs, N_LIDAR_BINS), maxd, device=self.device)
        ring = ring.scatter_reduce(1, bin_id, d, reduce="amin")
        return ring / maxd

    # ---------------- 观测 / 特权状态 ----------------

    def _get_observations(self) -> dict[str, torch.Tensor]:
        """构造各机观测。

        **去中心化模式下这里的每一项都必须由本机感知 + 本机 belief + 已收到的
        通信消息产生**，不得读 self.omap（团队图）或他机的真实状态。
        """
        self._step_ctr += 1
        self._tick += 1
        self._update_true_velocity()
        if self._tick % self.cfg.map_update_every == 0:
            self._update_map()
        if self.dec:
            self._rebuild_local_graphs()                 # Actor 用：按机图，错开 + 事件触发
            if self._tick % self.cfg.task_graph_every == 0:
                self._rebuild_task_graph()               # critic 用：团队图（特权）
        elif self._tick % self.cfg.task_graph_every == 0:
            self._rebuild_task_graph()

        pos = self._agent_pos()
        ego_vel = self._true_vel if self.cfg.use_true_velocity else self._vel_cmd
        half = self._map_span / 2

        obs = {}
        for i, name in enumerate(self.cfg.possible_agents):
            parts = [
                pos[:, i] / half,                                    # ego 位置
                ego_vel[:, i] / self.cfg.max_speed,                  # ego 速度
                self._lidar_ring(i),                                 # LiDAR 环
            ]
            # task graph：去中心化时用本机自己建的图
            if self.dec:
                tp, tnv, tnu, tv = (self._ltask_pos[:, i], self._ltask_nvox[:, i],
                                    self._ltask_nunk[:, i], self._ltask_valid[:, i])
            else:
                tp, tnv, tnu, tv = (self._task_pos, self._task_nvox,
                                    self._task_nunk, self._task_valid)
            dp = tp - pos[:, i].unsqueeze(1)                         # (E,M,3)
            dist = dp.norm(dim=-1, keepdim=True)
            tfeat = torch.cat([
                dp / half,
                dist / self._map_span.norm(),
                (tnv / 2000.0).unsqueeze(-1),
                (tnu / 5000.0).unsqueeze(-1),
                tv.float().unsqueeze(-1),
            ], dim=-1)                                               # (E,M,7)
            parts.append(tfeat.flatten(1))

            # 队友：去中心化时只能用"收到过的"状态快照 + 消息年龄；
            # 从未收到则为零向量 + 年龄拉满，策略仍能照常前向。
            for j in range(self.A):
                if j == i:
                    continue
                if self.dec:
                    ps = self.belief.peer_state[:, i, j]              # (E,6) 地图局部系
                    valid = self.belief.peer_valid[:, i, j].float().unsqueeze(-1)
                    parts.append(((ps[:, :3] - pos[:, i]) / half) * valid)
                    parts.append(((ps[:, 3:] - ego_vel[:, i]) / self.cfg.max_speed) * valid)
                    parts.append((self.belief.msg_age[:, i, j]
                                  / float(self.cfg.comm.max_age)).unsqueeze(-1))
                else:
                    parts.append((pos[:, j] - pos[:, i]) / half)
                    parts.append((self._vel_cmd[:, j] - self._vel_cmd[:, i]) / self.cfg.max_speed)
            # g_topo：本机任务选择的结果
            parts.append((self._sel_task[:, i] - pos[:, i]) / half)
            obs[name] = torch.cat(parts, dim=-1)
        return obs

    def _get_states(self) -> torch.Tensor:
        """CTDE 中心化 critic 的特权全局状态。

        创新点四 Counterfactual Marginal Utility 就在这里落地：
        u_i = V(s) - V(M_i(s))，M_i 把 agent i 的状态与其独有地图证据一起屏蔽。
        """
        pos = self._agent_pos()
        vel = self._true_vel if self.cfg.use_true_velocity else self._vel_cmd
        parts = []
        for i in range(self.A):
            parts.append(pos[:, i] / (self._map_span / 2))
            parts.append(vel[:, i] / self.cfg.max_speed)
        parts.append(self.omap.known_ratio().unsqueeze(-1))
        sfeat = torch.cat([
            self._task_pos / (self._map_span / 2),
            (self._task_nvox / 2000.0).unsqueeze(-1),
            (self._task_nunk / 5000.0).unsqueeze(-1),
            self._task_valid.float().unsqueeze(-1),
        ], dim=-1)
        parts.append(sfeat.flatten(1))
        return torch.cat(parts, dim=-1)

    # ---------------- 奖励 / 终止 ----------------

    def _get_rewards(self) -> dict[str, torch.Tensor]:
        """文档 4.5 的奖励结构（不过度堆叠）:

            r = w_I * r_info + w_U * r_utility - w_R * r_redundancy
                - w_C * r_collision - w_T * r_time

        r_utility（创新点四的 counterfactual utility）尚未实现，
        现用队友过近的去冗余代理项占位 —— 对应 MARVEL 里手工版的
        calculate_overlap_reward，将来由中心化 critic 的 V(s)-V(M_i(s)) 替换。
        """
        n_known = self.omap.known_mask.flatten(1).sum(dim=1).float()
        r_info = ((n_known - self._prev_known) / self.cfg.info_voxel_norm
                  ).clamp(max=self.cfg.info_clip)
        self._prev_known = n_known.clone()
        known_ratio = self.omap.known_ratio()
        completed = (known_ratio >= self.cfg.done_known_ratio).float()

        overlap = self.calculate_overlap_reward()
        use_marg = self.cfg.reward_mode == "marginal"
        marginal = self.calculate_marginal_coverage() if use_marg else None

        rewards = {}
        for i, name in enumerate(self.cfg.possible_agents):
            col = self._collided[:, i].float()
            # 信息项：共享团队增益（基线） 或 本机边际增益（difference reward）
            info_term = marginal[:, i] if use_marg else r_info
            r = (self.cfg.w_info * info_term
                 + self.cfg.w_complete * completed
                 - self.cfg.w_collision * col
                 - self.cfg.w_time)
            if not use_marg:
                r = r - self.cfg.w_redundancy * overlap[:, i]
            rewards[name] = r

        # 供 skrl 的 environment_info: log 写进 tensorboard
        log = {
            "known_ratio": known_ratio.mean(),
            "n_task_nodes": self._task_valid.float().sum(dim=1).mean(),
            "r_info": r_info.mean(),
            "overlap_proxy": overlap.mean(),
        }
        if self.dec:
            log["local_task_nodes"] = self._ltask_valid.float().sum(dim=2).mean()
            log["exclusive_voxels"] = self.belief.exclusive_coverage().mean()
            if use_marg:
                log["r_marginal_mean"] = marginal.mean()
            log.update({k: v for k, v in self.belief.comm_stats().items()})
        self.extras["log"] = log
        return rewards

    # ---------------- 评测指标 ----------------

    def _metric_belief(self):
        """算指标用的按机 belief：去中心化走 belief，基线走影子 belief。"""
        return self.belief if self.dec else self.shadow

    def coverage_metrics(self) -> dict[str, torch.Tensor]:
        """四组实验可比的同口径指标。没有按机 belief 时重叠率返回 NaN。

        overlap_rate = 1 - |团队并集| / Σ_i |各机各自观测到的|
            0   完全不重叠（各机分工彻底）
            ->1 高度重复（都在看同一片）
        这是 MARVEL 里"感知重叠率"的体素版，也是创新点四要压的量。
        """
        out = {"known_ratio": self.omap.known_ratio()}
        mb = self._metric_belief()
        if mb is None:
            out["overlap_rate"] = torch.full((self.num_envs,), float("nan"), device=self.device)
            out["exclusive_frac"] = torch.full((self.num_envs,), float("nan"), device=self.device)
            return out
        per = mb.known_per_agent()                       # (E,A)
        team = mb.team_known_count()                     # (E,)
        tot = per.sum(dim=1).clamp(min=1.0)
        out["overlap_rate"] = 1.0 - team / tot
        out["exclusive_frac"] = mb.exclusive_coverage().sum(dim=1) / team.clamp(min=1.0)
        return out

    # ---------------- 去冗余项：两种实现，供消融对照 ----------------

    def calculate_overlap_reward(self) -> torch.Tensor:
        """基线去冗余代理项 (E,A)：队友越近惩罚越大。

        对应 MARVEL 里手工版的 `calculate_overlap_reward`。它只看机间距离，
        完全不看实际观测重叠，所以是个粗代理 —— 保留它是为了和边际覆盖做对照。
        """
        pos = self._agent_pos()
        red = torch.zeros(self.num_envs, self.A, device=self.device)
        for i in range(self.A):
            for j in range(self.A):
                if i == j:
                    continue
                dij = (pos[:, j] - pos[:, i]).norm(dim=-1)
                red[:, i] += torch.exp(-dij / self.cfg.redundancy_sigma)
        return red

    def calculate_marginal_coverage(self, commit: bool = True) -> torch.Tensor:
        """本步边际新增覆盖 (E,A)，口径与 r_info 对齐，可直接替换 r_info。

        底层用 `MultiAgentBelief.stepwise_marginal()`：只算"本步第一次被团队知道"
        的体素，同一步被 k 架机同时观测到就各记 1/k，因此严格守恒
        `Σ_i marginal_i = Δ|team_known|`。

        这里乘 A 再按 `info_voxel_norm` 归一化、按 `info_clip` 截断，于是

            mean_i(marginal_i) == r_info

        也就是说换成这个口径不改变信息项的总体尺度，只改变它在各机间的分配 ——
        消融实验要比的正是这个分配方式。

        **仅用于训练期奖励，绝不进入 Actor 观测**（它需要全局的 own 图集合）。

        为什么不用"累计独占覆盖做差分"：那是存量差分，队友重复观测到本机原本
        独占的体素时会变负，把本步真实的新贡献一起抹掉；截断到 0 又会让贡献
        被静默吞掉。累计独占量保留在 `belief.exclusive_coverage()` 里当指标。

        Args:
            commit: True 时推进"上一步团队已知掩码"。一步只应调用一次。
        """
        if not self.dec:
            return torch.zeros(self.num_envs, self.A, device=self.device)
        m = self.belief.stepwise_marginal(commit=commit)         # 单位：体素数
        return ((self.A * m / self.cfg.info_voxel_norm).clamp(max=self.cfg.info_clip)
                * self.cfg.w_marginal)

    def _get_dones(self) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        # 碰撞判定：LiDAR 环最小距离低于阈值
        for i in range(self.A):
            ring = self._lidar_ring(i) * self.cfg.map_cfg.max_range
            self._collided[:, i] = ring.min(dim=1).values < self.cfg.collision_dist

        time_out = self.episode_length_buf >= self.max_episode_length - 1
        explored = self.omap.known_ratio() >= self.cfg.done_known_ratio
        collision = self._collided.any(dim=1)
        terminated = collision | explored

        # 保存 done 判定时的指标。_reset_idx 随后会清空地图和碰撞标志，
        # 所以这些值必须在这里捕获。
        self._last_done_collision = collision.clone()
        self._last_done_explored = explored.clone()
        self._last_done_timeout = time_out.clone()
        self._last_done_known_ratio = self.omap.known_ratio().clone()
        # overlap_rate 需要逐机 belief 的并集统计，平时不应每步计算；只在
        # 真正结束/超时时保存终止时刻的值。
        if bool((terminated | time_out).any()):
            self._last_done_overlap_rate = self.coverage_metrics()["overlap_rate"].clone()

        return ({a: terminated for a in self.cfg.possible_agents},
                {a: time_out for a in self.cfg.possible_agents})

    def _reset_idx(self, env_ids: Sequence[int] | None) -> None:
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self.drones[0]._ALL_INDICES
        super()._reset_idx(env_ids)

        ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        if self.shadow is not None:
            self.shadow.reset(ids)
        if self.dec:
            self.belief.reset(ids)
            self._ltask_pos[ids] = 0.0
            self._ltask_nvox[ids] = 0.0
            self._ltask_nunk[ids] = 0.0
            self._ltask_valid[ids] = False
            self._last_graph_known[ids] = 0.0
        else:
            self.omap.reset(ids)
        self._prev_known[ids] = 0.0   # 单位：体素数
        self._step_ctr[ids] = 0
        self._collided[ids] = False
        self._vel_cmd[ids] = 0.0
        self._true_vel[ids] = 0.0

        # 无人机回到初始位姿
        for i, d in enumerate(self.drones):
            root = d.data.default_root_state[ids].clone()
            root[:, :3] += self.scene.env_origins[ids]
            d.write_root_pose_to_sim(root[:, :7].contiguous(), ids)
            d.write_root_velocity_to_sim(torch.zeros_like(root[:, 7:13]), ids)

        # 重置后立刻建一次图并初始化任务图，避免首步观测全零
        self.scene.update(self.cfg.sim.dt)
        # 差分测速的基准位姿必须在建图前落定，否则首步会算出巨大的虚假速度
        self._prev_pos[ids] = self._agent_pos()[ids]
        self._update_map()
        self._rebuild_task_graph()
        if self.dec:
            # 首次建图：绕过事件触发，强制给每机建一张图
            slots = (ids.unsqueeze(1) * self.A
                     + torch.arange(self.A, device=self.device)).reshape(-1)
            self._force_local_graphs(slots)
            # 把重置后立刻观测到的那一片标成"团队已知"，否则下一步的 stepwise_marginal
            # 会把它当成新发现，给出一个巨大的虚假首步奖励。
            # 与基线给 _prev_known 播种（下一行）是同一件事的两种口径。
            self.belief.stepwise_marginal(commit=True)
        self._prev_known[ids] = self.omap.known_mask.flatten(1).sum(dim=1).float()[ids]

    def _force_local_graphs(self, slots: torch.Tensor) -> None:
        """不走事件触发，强制为指定 slot 重建按机 task graph（重置时用）。"""
        if slots.numel() == 0:
            return
        view = self.belief.fused_slice(slots)
        fm = frontier_mask(view, self.cfg.frontier_cfg)
        labels, _ = label_components(fm, self.cfg.frontier_cfg)
        nodes = extract_task_nodes(view, labels, self.cfg.frontier_cfg,
                                   max_nodes=self.cfg.max_task_nodes)
        e_i, a_i = slots // self.A, slots % self.A
        self._ltask_pos[e_i, a_i] = nodes["pos"]
        self._ltask_nvox[e_i, a_i] = nodes["n_voxels"]
        self._ltask_nunk[e_i, a_i] = nodes["n_unknown"]
        self._ltask_valid[e_i, a_i] = nodes["valid"]
        self._last_graph_known[e_i, a_i] = self.belief.known_per_agent()[e_i, a_i]
        self._select_tasks_local(e_i, a_i)
