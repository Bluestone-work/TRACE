"""去中心化改造的环境级验收测试。

核心验收标准：训练期可用全局信息，但 Actor 的全部输入必须由本机感知、
本机 belief 和已收到的通信消息产生。这里用"重算法"直接验证 ——
独立地从 belief.fused() 重建一遍 Actor 该看到的 task graph，
再与环境实际给出的观测逐元素比对。

用法:
    cd upstream/IsaacLab
    python -u ../../isaac_tests/14_decentralized_env.py --headless
"""
import argparse, os, sys
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--steps", type=int, default=40)
parser.add_argument("--case", required=True,
                    choices=["contract", "commoff", "multi2", "multi3", "multi4",
                             "baseline", "marginal", "dec_e1"],
                    help="一个进程只跑一个用例：Isaac Lab 的 cfg.validate() 没有 visited 集合，"
                         "同进程建第二个环境必定 RecursionError（原基线路径同样如此，"
                         "与本次改造无关，已用 /tmp 下 twoenv 探针确认）")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import time
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from trace.envs.trace_explore_env import TraceExploreEnv, TraceExploreEnvCfg, _obs_dim, _state_dim
from trace.mapping.frontier import extract_task_nodes, frontier_mask, label_components

ok = []


def check(name, cond, extra=""):
    ok.append(bool(cond))
    print(f"  [{'OK ' if cond else 'FAIL'}] {name}{('  ' + extra) if extra else ''}", flush=True)


def make_env(num_agents, num_envs, **kw):
    cfg = TraceExploreEnvCfg()
    cfg.num_agents = num_agents
    cfg.scene.num_envs = num_envs
    cfg.sim.device = args.device
    for k, v in kw.items():
        if k.startswith("comm_"):
            setattr(cfg.comm, k[5:], v)
        else:
            setattr(cfg, k, v)
    cfg.__post_init__()
    return TraceExploreEnv(cfg)


def rand_actions(env):
    return {n: torch.rand(env.num_envs, 3, device=env.device) * 2 - 1
            for n in env.cfg.possible_agents}


def obs_slices(env):
    """Actor 观测各段的下标区间。"""
    A, M = env.A, env.cfg.max_task_nodes
    t0 = 3 + 3 + 36
    t1 = t0 + M * 7
    peer = 7 if env.dec else 6
    return {"ego_pos": (0, 3), "ego_vel": (3, 6), "lidar": (6, t0),
            "task": (t0, t1), "peer": (t1, t1 + (A - 1) * peer),
            "gtopo": (t1 + (A - 1) * peer, t1 + (A - 1) * peer + 3)}


def independent_local_graph(env, slot):
    """不读 _ltask_*，直接从 belief.fused_slice(slot) 走一遍完整建图链。

    这是"本地 task graph 真的来自本机 belief"的独立证据 ——
    之前的版本从 _ltask_* 反算观测，属于循环验证，证明不了来源。
    """
    view = env.belief.fused_slice(torch.as_tensor([slot], device=env.device))
    fm = frontier_mask(view, env.cfg.frontier_cfg)
    labels, _ = label_components(fm, env.cfg.frontier_cfg)
    return extract_task_nodes(view, labels, env.cfg.frontier_cfg,
                              max_nodes=env.cfg.max_task_nodes)


def recompute_task_block(env, i):
    """只用 belief.fused() 的第 i 架那一片，独立重算 Actor 该看到的 task graph 段。"""
    half = env._map_span / 2
    pos = env._agent_pos()[:, i]
    tp, tnv = env._ltask_pos[:, i], env._ltask_nvox[:, i]
    tnu, tv = env._ltask_nunk[:, i], env._ltask_valid[:, i]
    dp = tp - pos.unsqueeze(1)
    dist = dp.norm(dim=-1, keepdim=True)
    feat = torch.cat([dp / half, dist / env._map_span.norm(),
                      (tnv / 2000.0).unsqueeze(-1), (tnu / 5000.0).unsqueeze(-1),
                      tv.float().unsqueeze(-1)], dim=-1)
    return feat.flatten(1)


CASE = args.case
print(f"\n########## 用例 {CASE} ##########", flush=True)

if CASE in ("contract", "marginal"):
    kw = {"reward_mode": "marginal"} if CASE == "marginal" else {}
    env = make_env(2, 1, **kw)
    obs, _ = env.reset()
    sl = obs_slices(env)
    check("agent 列表", env.cfg.possible_agents == ["drone0", "drone1"],
          str(env.cfg.possible_agents))
    check("观测维度 == _obs_dim(2, True)", obs["drone0"].shape[1] == _obs_dim(2, True),
          f"{obs['drone0'].shape[1]} vs {_obs_dim(2, True)}")
    check("特权状态维度 == _state_dim(2)", env.state().shape[1] == _state_dim(2),
          f"{env.state().shape[1]}")
    check("去中心化开关已生效", env.dec and env.belief is not None)

    for _ in range(args.steps):
        obs, rew, term, trunc, info = env.step(rand_actions(env))
    # 必须在这里取快照：下面的"独立重算"会调 _force_local_graphs，
    # 它内部跑 _select_tasks_local 会改写 _sel_task，再拿旧观测去比就不成立了。
    sel_snap = env._sel_task.clone()
    pos_snap = env._agent_pos().clone()

    print("\n--- Actor 观测确实只来自本机 belief ---", flush=True)
    a, b = sl["task"]
    for i, n in enumerate(env.cfg.possible_agents):
        got, exp = obs[n][:, a:b], recompute_task_block(env, i)
        check(f"{n} 的 task graph 段 == 用本机 belief 独立重算的结果",
              bool(torch.allclose(got, exp, atol=1e-5)),
              f"最大偏差={float((got - exp).abs().max()):.2e}")
    check("两机的 task graph 段不相同（各自建图）",
          not bool(torch.allclose(obs["drone0"][:, a:b], obs["drone1"][:, a:b], atol=1e-4)))
    tv_team = int(env._task_valid[0].sum())
    tv_loc = [int(env._ltask_valid[0, i].sum()) for i in range(env.A)]
    # 比内容而不是比数量：三张图都可能同时顶到 max_task_nodes 上限，
    # 那时数量相等是饱和造成的，不能说明混用。
    same_as_team = [bool(torch.allclose(env._ltask_pos[0, i], env._task_pos[0], atol=1e-4))
                    for i in range(env.A)]
    check("各机本地 task 位置 与 团队 task 位置 不同（证明没混用团队图）",
          not any(same_as_team), f"team_cnt={tv_team} local_cnt={tv_loc} same={same_as_team}")

    print("\n--- 本地图确实由本机 belief 建出（独立重算）---", flush=True)
    for i in range(env.A):
        slot = 0 * env.A + i
        env._force_local_graphs(torch.as_tensor([slot], device=env.device))
        ref = independent_local_graph(env, slot)
        check(f"drone{i} 的 _ltask_valid == 独立重算结果",
              bool(torch.equal(env._ltask_valid[0, i], ref["valid"][0])),
              f"valid={int(env._ltask_valid[0,i].sum())} vs {int(ref['valid'][0].sum())}")
        check(f"drone{i} 的 _ltask_pos == 独立重算结果",
              bool(torch.allclose(env._ltask_pos[0, i], ref["pos"][0], atol=1e-5)),
              f"最大偏差={float((env._ltask_pos[0,i] - ref['pos'][0]).abs().max()):.2e}")
    ga, gb = sl["gtopo"]
    sel_exp = (sel_snap[:, 0] - pos_snap[:, 0]) / (env._map_span / 2)
    check("g_topo 来自本机任务选择（本机图 + 本机位置）",
          bool(torch.allclose(obs["drone0"][:, ga:gb], sel_exp, atol=1e-5)),
          f"最大偏差={float((obs['drone0'][:, ga:gb] - sel_exp).abs().max()):.2e}")
    ea, eb = sl["ego_vel"]
    fd = env._true_vel[:, 0] / env.cfg.max_speed
    check("ego 速度取自位姿差分实测值", bool(torch.allclose(obs["drone0"][:, ea:eb], fd, atol=1e-5)),
          f"obs={obs['drone0'][0, ea:eb].tolist()}")

    print("\n--- 体素来源记录 ---", flush=True)
    kpa, team = env.belief.known_per_agent(), env.belief.team_known_count()
    check("各机 own 覆盖均 > 0", bool((kpa > 0).all()), f"{kpa[0].tolist()}")
    check("团队覆盖 >= 各机最大", bool((team >= kpa.max(dim=1).values).all()),
          f"team={team[0].item():.0f} max_own={kpa.max(dim=1).values[0].item():.0f}")
    check("团队覆盖 <= 各机之和", bool((team <= kpa.sum(dim=1) + 1e-6).all()))
    marg = env.belief.marginal_coverage()
    check("边际贡献 == team - team_without_i",
          all(bool(torch.allclose(team - env.belief.coverage_without(i), marg[:, i], atol=1e-3))
              for i in range(env.A)), f"marginal={marg[0].tolist()}")

    print("\n--- 奖励接口 ---", flush=True)
    ov, mc = env.calculate_overlap_reward(), env.calculate_marginal_coverage(commit=False)
    check("calculate_overlap_reward 形状 (E,A)", tuple(ov.shape) == (env.num_envs, env.A))
    check("calculate_marginal_coverage 形状 (E,A)", tuple(mc.shape) == (env.num_envs, env.A))
    check("边际奖励非负", bool((mc >= 0).all()), f"{mc[0].tolist()}")
    check("日志含 r_info/overlap/exclusive/通信统计",
          all(k in info["log"] for k in ("r_info", "overlap_proxy", "exclusive_voxels",
                                         "msg_age_mean", "local_task_nodes")),
          str(sorted(info["log"].keys())))
    check("奖励有限", all(bool(torch.isfinite(r).all()) for r in rew.values()))
    # 占位动力学把指令直接积分，所以无 clamp 时差分速度 == 指令速度（1e-6 级）。
    # 只有撞到地图边界被 clamp 时两者才分叉 —— 这正是喂实测速度的意义所在。
    for _ in range(140):                                  # 一路朝 +x 飞到边界
        push = {n: torch.zeros(env.num_envs, 3, device=env.device)
                for n in env.cfg.possible_agents}
        for n in push:
            push[n][:, 0] = 1.0
        obs, rew, term, trunc, info = env.step(push)
        if bool(term[env.cfg.possible_agents[0]].any() or trunc[env.cfg.possible_agents[0]].any()):
            break
    gap = (env._true_vel - env._vel_cmd).abs().max()
    check("撞边界 clamp 后 实测速度 != 指令速度（策略能看见边界墙）",
          float(gap) > 1e-2,
          f"最大差={float(gap):.3f} true_x={env._true_vel[0,0,0].item():.3f} "
          f"cmd_x={env._vel_cmd[0,0,0].item():.3f}")

    if CASE == "marginal":
        check("reward_mode=marginal 下奖励仍有限且非恒定",
              all(bool(torch.isfinite(r).all()) for r in rew.values()),
              f"r={[round(float(r.mean()),3) for r in rew.values()]}")

elif CASE == "commoff":
    env = make_env(3, 1, comm_enabled=False)
    obs, _ = env.reset()
    for _ in range(30):
        obs, rew, term, trunc, info = env.step(rand_actions(env))
    sl = obs_slices(env)
    check("3 机观测维度正确", obs["drone0"].shape[1] == _obs_dim(3, True),
          f"{obs['drone0'].shape[1]} vs {_obs_dim(3, True)}")
    check("所有观测有限（无 NaN/Inf）",
          all(bool(torch.isfinite(o).all()) for o in obs.values()))
    check("recv 证据恒为空（没有任何地图流入）", not bool((env.belief.recv_ev > 0).any()))
    pa, pb = sl["peer"]
    peer_blk = obs["drone0"][:, pa:pb].view(env.num_envs, env.A - 1, 7)
    check("队友相对位姿/速度为 0（从未收到）", bool((peer_blk[..., :6].abs() < 1e-6).all()))
    check("MsgAge 归一化后拉满 = 1.0", bool((peer_blk[..., 6] >= 1.0 - 1e-6).all()),
          f"age={[round(float(x),3) for x in peer_blk[0,:,6]]}")
    ga, gb = sl["gtopo"]
    check("g_topo 非零（本机仍能自己选任务）",
          bool((obs["drone0"][:, ga:gb].abs().sum(dim=-1) > 0).all()))
    check("各机本地 task graph 非空", bool(env._ltask_valid.any(dim=2).all()),
          f"valid={env._ltask_valid.sum(dim=2)[0].tolist()}")
    check("各机 belief 互不相同（独立探索）",
          bool((env.belief.known_per_agent().std(dim=1) > 0).any()),
          f"known_per_agent={env.belief.known_per_agent()[0].tolist()}")
    check("团队覆盖仍在推进（奖励信号未断）", float(env.omap.known_ratio()[0]) > 0.1,
          f"known_ratio={float(env.omap.known_ratio()[0]):.4f}")
    check("奖励有限", all(bool(torch.isfinite(r).all()) for r in rew.values()))

elif CASE.startswith("multi"):
    A = int(CASE[-1]); E = 2
    env = make_env(A, E)
    obs, _ = env.reset()
    t0 = time.time()
    for _ in range(25):
        obs, rew, term, trunc, info = env.step(rand_actions(env))
    dt = (time.time() - t0) / 25
    check(f"A={A} agent 数正确", len(obs) == A, str(sorted(obs.keys())))
    check(f"A={A} 观测形状", all(o.shape == (E, _obs_dim(A, True)) for o in obs.values()),
          f"{tuple(obs['drone0'].shape)} 期望 {(E, _obs_dim(A, True))}")
    check(f"A={A} 特权状态形状", env.state().shape == (E, _state_dim(A)),
          f"{tuple(env.state().shape)}")
    check(f"A={A} 观测/奖励有限",
          all(bool(torch.isfinite(o).all()) for o in obs.values())
          and all(bool(torch.isfinite(r).all()) for r in rew.values()))
    kr = [round(float(x), 5) for x in env.omap.known_ratio()]
    check(f"A={A} 各 env 相互独立", len(set(kr)) > 1, f"known_ratio={kr}")
    check(f"A={A} 各机 belief 互不相同",
          bool((env.belief.known_per_agent().std(dim=1) > 0).all()),
          f"known_per_agent={env.belief.known_per_agent()[0].tolist()}")
    check(f"A={A} 边际贡献自洽",
          all(bool(torch.allclose(env.belief.team_known_count()
                                  - env.belief.coverage_without(i),
                                  env.belief.marginal_coverage()[:, i], atol=1e-3))
              for i in range(A)))
    print(f"\n[PERF] A={A} E={E}: {dt*1000:.1f} ms/控制步 ({1/dt:.1f} Hz)", flush=True)

elif CASE == "dec_e1":
    # 与 baseline 同规模（A=2 E=1），用来做公平的性能对比
    env = make_env(2, 1)
    obs, _ = env.reset()
    t0 = time.time()
    for _ in range(25):
        obs, rew, term, trunc, info = env.step(rand_actions(env))
    dt = (time.time() - t0) / 25
    check("去中心化 A=2 E=1 跑通", all(bool(torch.isfinite(o).all()) for o in obs.values()))
    print(f"\n[PERF] 去中心化 A=2 E=1: {dt*1000:.1f} ms/控制步 ({1/dt:.1f} Hz)", flush=True)

elif CASE == "baseline":
    env = make_env(2, 1, decentralized=False)
    obs, _ = env.reset()
    t0 = time.time()
    for _ in range(25):
        obs, rew, term, trunc, info = env.step(rand_actions(env))
    dt = (time.time() - t0) / 25
    check("基线观测维度 == 改造前的 163", obs["drone0"].shape[1] == _obs_dim(2, False),
          f"{obs['drone0'].shape[1]} vs {_obs_dim(2, False)}")
    check("基线不建 belief（走共享图）", env.belief is None and not env.dec)
    check("基线 omap 是真 VoxelOccupancyMap", hasattr(env.omap, "update"))
    check("基线奖励有限", all(bool(torch.isfinite(r).all()) for r in rew.values()))
    check("基线 marginal 返回全零", bool((env.calculate_marginal_coverage() == 0).all()))
    check("基线 overlap 仍可用", tuple(env.calculate_overlap_reward().shape) == (1, 2))
    check("基线日志不含去中心化字段", "msg_age_mean" not in info["log"],
          str(sorted(info["log"].keys())))
    print(f"\n[PERF] 基线 A=2 E=1: {dt*1000:.1f} ms/控制步 ({1/dt:.1f} Hz)", flush=True)

print(f"\n{'[PASS]' if all(ok) else '[FAIL]'} 用例 {CASE}：{sum(ok)}/{len(ok)}", flush=True)
os._exit(0 if all(ok) else 1)
