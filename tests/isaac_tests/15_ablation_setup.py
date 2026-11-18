"""消融实验的前置核对：观测维度、网络参数量、奖励尺度。

目的是确认四组之间**只有预期的那一个变量在变**。一个进程只能查一组
（Isaac Lab 的 cfg.validate() 不允许同进程建第二个环境）。

用法:
    cd upstream/IsaacLab
    python -u ../../isaac_tests/15_ablation_setup.py --headless --group base
"""
import argparse, json, os, sys
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--group", required=True)
parser.add_argument("--steps", type=int, default=120)
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--num_agents", type=int, default=2)
parser.add_argument("--seed", type=int, default=7)
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

env_cfg = load_cfg_from_registry(args.task if hasattr(args, "task") else "Trace-Explore-MARL-v0",
                                 "env_cfg_entry_point")
TASK = "Trace-Explore-MARL-v0"
env_cfg.scene.num_envs = args.num_envs
env_cfg.sim.device = args.device
env_cfg.seed = args.seed
env_cfg.num_agents = args.num_agents
env_cfg.__post_init__()
summary = apply_group(env_cfg, args.group, measure=True)

env = SkrlVecEnvWrapper(gym.make(TASK, cfg=env_cfg), ml_framework="torch")
base = env.unwrapped
names = list(base.cfg.possible_agents)
A, E = base.A, base.num_envs

agent_cfg = load_cfg_from_registry(TASK, "skrl_mappo_cfg_entry_point")
agent_cfg["seed"] = args.seed
agent_cfg["trainer"]["timesteps"] = 1
agent = Runner(env, agent_cfg).agent

# ---------- 参数量 ----------
def count(m):
    return int(sum(p.numel() for p in m.parameters()))

pol = {n: count(agent.policies[n]) for n in names}
val = {n: count(agent.values[n]) for n in names}
print(f"[OK] 分组 {args.group}", flush=True)
print(f"  观测维度 = {summary['obs_dim']}   特权状态维度 = {summary['state_dim']}", flush=True)
print(f"  policy 参数量 = {pol}", flush=True)
print(f"  critic 参数量 = {val}", flush=True)
print(f"  合计（单机） = {pol[names[0]] + val[names[0]]}", flush=True)

# ---------- 奖励尺度 ----------
torch.manual_seed(args.seed)
env._reset_once = True
obs, _ = env.reset()
states = env.state()
rec = {k: [] for k in ["r_total", "r_info", "overlap_proxy", "r_marginal_mean",
                       "known_ratio", "n_task_nodes", "local_task_nodes"]}
import time
t0 = time.time()
for _ in range(args.steps):
    acts = {n: torch.rand(E, 3, device=base.device) * 2 - 1 for n in names}
    obs, rew, term, trunc, info = env.step(acts)
    states = env.state()
    rec["r_total"].append(float(sum(r.mean() for r in rew.values()) / A))
    for k in ["r_info", "overlap_proxy", "r_marginal_mean", "known_ratio",
              "n_task_nodes", "local_task_nodes"]:
        if k in info.get("log", {}):
            rec[k].append(float(info["log"][k]))
dt = (time.time() - t0) / args.steps


def stat(v):
    if not v:
        return None
    a = np.asarray(v, dtype=np.float64)
    return {"mean": float(a.mean()), "std": float(a.std()),
            "min": float(a.min()), "max": float(a.max())}


scales = {k: stat(v) for k, v in rec.items()}
print("\n[OK] 随机策略下的奖励尺度（%d 步）" % args.steps, flush=True)
for k, v in scales.items():
    if v is None:
        continue
    print(f"  {k:18s} mean={v['mean']:+9.4f}  std={v['std']:7.4f}  "
          f"[{v['min']:+8.3f}, {v['max']:+8.3f}]", flush=True)
print(f"\n[PERF] {dt*1000:.1f} ms/控制步 ({1/dt:.1f} Hz), num_envs={E}, A={A}", flush=True)

out = {"group": args.group, "group_summary": summary,
       "policy_params": pol, "critic_params": val,
       "obs_dim": summary["obs_dim"], "state_dim": summary["state_dim"],
       "reward_scales": scales, "ms_per_step": dt * 1000.0,
       "num_envs": E, "num_agents": A, "steps": args.steps}
if args.out:
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2, ensure_ascii=False)
    print(f"[OK] 已写出 {args.out}", flush=True)
print("[DONE]", flush=True)
os._exit(0)
