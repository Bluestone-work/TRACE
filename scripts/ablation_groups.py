"""四组消融的配置定义 —— 训练和评测共用这一份，避免两边漂移。

分组（每一组只比上一组多一个变量）：

    A base            原始基线：共享地图，共享 r_info + 距离去冗余惩罚
    B belief          + 每机独立 belief、Actor 观测去中心化（通信理想：全连通/无丢包/无延迟）
    C belief_comm     + 通信随机化（有限半径、丢包、延迟、按间隔发送）
    D belief_marginal + 信息项换成本步边际新增覆盖（difference reward），去掉距离惩罚

注意 A 组观测 163 维、B/C/D 组 164 维（每个队友多一维 MsgAge），
网络第一层因此差 256 个权重（约 0.19% 参数量）。这是"保持基线不变"的必然代价，
不是实验设计失误 —— 具体量级见 isaac_tests/15_ablation_setup.py 的实测。
"""

from __future__ import annotations

GROUPS = ("base", "belief", "belief_comm", "belief_marginal")

# 通信随机化的参数。理想通信 = 全连通、每步发、不丢、不延迟。
COMM_IDEAL = dict(enabled=True, range_m=1.0e9, every=1, drop_prob=0.0, delay_steps=0)
COMM_REAL = dict(enabled=True, range_m=25.0, every=5, drop_prob=0.2, delay_steps=2)


def apply_group(env_cfg, group: str, measure: bool = False) -> dict:
    """把分组设置写进 env_cfg，返回一份人类可读的摘要（用于落盘记录）。

    Args:
        env_cfg: TraceExploreEnvCfg 实例
        group:   GROUPS 之一
        measure: True 时让基线组也维护影子 belief，以便算同口径的 Overlap
    """
    if group not in GROUPS:
        raise ValueError(f"group 必须是 {GROUPS} 之一，收到 {group}")

    if group == "base":
        env_cfg.decentralized = False
        env_cfg.reward_mode = "overlap"
        env_cfg.measure_per_agent = bool(measure)
        comm = None
    else:
        env_cfg.decentralized = True
        env_cfg.measure_per_agent = False
        comm = COMM_IDEAL if group == "belief" else COMM_REAL
        for k, v in comm.items():
            setattr(env_cfg.comm, k, v)
        env_cfg.reward_mode = "marginal" if group == "belief_marginal" else "overlap"

    env_cfg.__post_init__()
    return {
        "group": group,
        "decentralized": bool(env_cfg.decentralized),
        "reward_mode": env_cfg.reward_mode,
        "measure_per_agent": bool(env_cfg.measure_per_agent),
        "comm": comm,
        "num_agents": int(env_cfg.num_agents),
        "obs_dim": int(next(iter(env_cfg.observation_spaces.values()))),
        "state_dim": int(env_cfg.state_space),
        # 奖励权重全组一致，只有口径不同 —— 记下来便于复查没有被改动
        "w_info": env_cfg.w_info,
        "info_voxel_norm": env_cfg.info_voxel_norm,
        "info_clip": env_cfg.info_clip,
        "w_redundancy": env_cfg.w_redundancy,
        "w_marginal": env_cfg.w_marginal,
        "w_collision": env_cfg.w_collision,
        "w_time": env_cfg.w_time,
        "w_complete": env_cfg.w_complete,
        "done_known_ratio": env_cfg.done_known_ratio,
    }
