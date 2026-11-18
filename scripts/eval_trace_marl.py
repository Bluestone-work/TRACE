"""加载 checkpoint，按分组跑多个 episode，输出可比的评测指标（JSON）。

指标口径（四组完全一致；基线组靠 measure_per_agent 的影子 belief 也能算 Overlap）：
    coverage       episode 结束时的团队 known_ratio
    overlap_rate   1 - |团队并集| / Σ_i |各机各自观测到的|，越小越说明分工好
    success        是否因"探索达标"终止（known_ratio >= done_known_ratio）
    steps_to_X     首次达到覆盖率 X 的控制步数（探索时间）。没达到记 None
    ep_len         episode 步数
    collided       是否发生过碰撞

用法:
    cd upstream/IsaacLab
    python -u ../../scripts/eval_trace_marl.py --group belief \
        --ckpt <path>/best_agent.pt --episodes 8 --num_envs 4 --out out.json
"""
import argparse, json, os, sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--group", required=True)
parser.add_argument("--ckpt", required=True)
parser.add_argument("--task", default="Trace-Explore-MARL-v0")
parser.add_argument("--episodes", type=int, default=8)
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--num_agents", type=int, default=2)
parser.add_argument("--seed", type=int, default=1234)
parser.add_argument("--max_steps", type=int, default=600, help="单 episode 步数上限（兜底）")
parser.add_argument("--stochastic", action="store_true")
parser.add_argument("--out", default=None)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import numpy as np
import torch
import gymnasium as gym
from isaaclab_rl.skrl import SkrlVecEnvWrapper
from isaaclab_tasks.utils.hydra import load_cfg_from_registry
from skrl.utils.runner.torch import Runner

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import trace.tasks  # noqa: F401
from scripts.ablation_groups import apply_group

THRESHOLDS = (0.5, 0.6, 0.7, 0.8)

env_cfg = load_cfg_from_registry(args.task, "env_cfg_entry_point")
env_cfg.scene.num_envs = args.num_envs
env_cfg.sim.device = args.device
env_cfg.seed = args.seed
env_cfg.num_agents = args.num_agents
env_cfg.__post_init__()
summary = apply_group(env_cfg, args.group, measure=True)
print(f"[INFO] 分组={args.group} obs_dim={summary['obs_dim']} "
      f"decentralized={summary['decentralized']} reward_mode={summary['reward_mode']}", flush=True)

env = SkrlVecEnvWrapper(gym.make(args.task, cfg=env_cfg), ml_framework="torch")
base = env.unwrapped
names = list(base.cfg.possible_agents)
A, E = base.A, base.num_envs

agent_cfg = load_cfg_from_registry(args.task, "skrl_mappo_cfg_entry_point")
agent_cfg["seed"] = args.seed
agent_cfg["trainer"]["timesteps"] = 1
agent = Runner(env, agent_cfg).agent
agent.load(args.ckpt)
agent.enable_training_mode(False)
print(f"[INFO] 已载入 {args.ckpt}", flush=True)


def act(obs, states):
    with torch.no_grad():
        a, outputs = agent.act(obs, states, timestep=0, timesteps=1)
    return a if args.stochastic else {n: outputs[n]["mean_actions"] for n in names}


torch.manual_seed(args.seed)
env._reset_once = True                       # wrapper 只在第一次真重置
obs, _ = env.reset()
states = env.state()

eps = []                                     # 收集到的 episode 记录
cur_len = np.zeros(E, dtype=np.int64)
cur_coll = np.zeros(E, dtype=bool)
cur_hit = [{t: None for t in THRESHOLDS} for _ in range(E)]
step = 0
while len(eps) < args.episodes and step < args.max_steps * args.episodes:
    # 在 step 之前读：环境在 step 内部会自动重置，之后读到的是新 episode 的地图
    met = base.coverage_metrics()
    kr_pre = met["known_ratio"].detach().cpu().numpy().copy()
    ov_pre = met["overlap_rate"].detach().cpu().numpy().copy()
    # 碰撞标志也必须在 step 之前读：_reset_idx 会把 _collided 清零，
    # step 之后读到的是重置后的 False（第一版就是这么把 collision_rate 记成 0 的）
    cur_coll |= base._collided.any(dim=1).detach().cpu().numpy()

    obs, rew, term, trunc, info = env.step(act(obs, states))
    states = env.state()
    step += 1
    cur_len += 1

    met2 = base.coverage_metrics()
    kr_now = met2["known_ratio"].detach().cpu().numpy()
    d_term = term[names[0]].detach().cpu().numpy()
    d_trunc = trunc[names[0]].detach().cpu().numpy()
    for e in range(E):
        kr_for_thresholds = (float(base._last_done_known_ratio[e])
                             if (d_term[e] or d_trunc[e]) else kr_now[e])
        for t in THRESHOLDS:
            if cur_hit[e][t] is None and kr_for_thresholds >= t:
                cur_hit[e][t] = int(cur_len[e])

    for e in range(E):
        if not (d_term[e] or d_trunc[e]):
            continue
        # DirectMARLEnv 在返回 step 前会自动 reset 已终止环境；使用环境在
        # _get_dones 中保存的终止瞬间指标，避免“最后一步刚达标却被算成碰撞”。
        success = bool(base._last_done_explored[e])
        is_coll = bool(base._last_done_collision[e])
        kr_final = float(base._last_done_known_ratio[e]) if (d_term[e] or d_trunc[e]) \
            else float(kr_pre[e])
        ov_final = float(base._last_done_overlap_rate[e]) if (d_term[e] or d_trunc[e]) \
            else float(ov_pre[e])
        cur_coll[e] = bool(cur_coll[e]) or is_coll
        eps.append({
            "coverage": kr_final,
            "overlap_rate": ov_final,
            "success": success,
            "ep_len": int(cur_len[e]),
            "collided": bool(cur_coll[e]),
            "reason": ("explored" if success else "collision" if is_coll else "timeout"),
            **{f"steps_to_{t}": cur_hit[e][t] for t in THRESHOLDS},
        })
        cur_len[e] = 0
        cur_coll[e] = False
        cur_hit[e] = {t: None for t in THRESHOLDS}
        if len(eps) >= args.episodes:
            break
    if step % 100 == 0:
        print(f"  step {step:4d} 已收集 {len(eps)}/{args.episodes} episode", flush=True)

eps = eps[:args.episodes]


def agg(key):
    v = [e[key] for e in eps if e[key] is not None]
    if not v:
        return {"mean": None, "std": None, "n": 0}
    a = np.asarray(v, dtype=np.float64)
    return {"mean": float(a.mean()), "std": float(a.std()), "n": int(a.size)}


result = {
    "group": args.group, "ckpt": args.ckpt, "seed": args.seed,
    "num_envs": E, "num_agents": A, "episodes": len(eps),
    "group_summary": summary,
    "metrics": {k: agg(k) for k in
                ["coverage", "overlap_rate", "ep_len"] + [f"steps_to_{t}" for t in THRESHOLDS]},
    "success_rate": float(np.mean([e["success"] for e in eps])) if eps else None,
    "collision_rate": float(np.mean([e["collided"] for e in eps])) if eps else None,
    "reasons": {r: sum(1 for e in eps if e["reason"] == r)
                for r in {e["reason"] for e in eps}},
    "episodes_detail": eps,
}
print(json.dumps({k: v for k, v in result.items() if k != "episodes_detail"},
                 indent=2, ensure_ascii=False), flush=True)
if args.out:
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(result, fh, indent=2, ensure_ascii=False)
    print(f"[OK] 已写出 {args.out}", flush=True)
print("[DONE]", flush=True)
os._exit(0)
