"""贪心 baseline：按 g_topo 朝选中的 task node 飞，同时标定奖励尺度。

这个脚本化策略也是后面和学习策略对比的 baseline —— 它用的正是
_select_tasks() 的贪心启发式（效用 = n_unknown/(距离+1)），
也就是创新点三 Foresighted Task-Value Head 要超越的对象。
"""
import argparse, os, sys
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser()
parser.add_argument("--steps", type=int, default=450)
parser.add_argument("--num_envs", type=int, default=2)
parser.add_argument("--random", action="store_true", help="改用随机动作做对照")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from trace.envs.trace_explore_env import TraceExploreEnv, TraceExploreEnvCfg

cfg = TraceExploreEnvCfg()
cfg.scene.num_envs = args.num_envs
cfg.sim.device = args.device
env = TraceExploreEnv(cfg)
obs, _ = env.reset()
A = cfg.num_agents
names = env.cfg.possible_agents
print(f"[OK] 策略 = {'随机' if args.random else '贪心(朝 task node)'}, "
      f"num_envs={env.num_envs}, agents={A}", flush=True)

r_hist = {n: [] for n in names}
print("\n  step | known  | task | 本步奖励 | 累计均值", flush=True)
for step in range(args.steps):
    acts = {}
    for i, n in enumerate(names):
        if args.random:
            acts[n] = torch.rand(env.num_envs, 3, device=env.device) * 2 - 1
        else:
            # obs 末 3 维就是 g_topo（选中 task 的相对方向，已按地图半尺寸归一化）
            g = obs[n][:, -3:]
            a = g / g.norm(dim=-1, keepdim=True).clamp(min=1e-6)   # 单位方向
            a[:, 2] *= 0.3                                          # 抑制垂直分量
            acts[n] = a.clamp(-1, 1)
    obs, rew, term, trunc, _ = env.step(acts)
    for n in names:
        r_hist[n].append(float(rew[n].mean()))
    if step % 50 == 0 or step == args.steps - 1:
        cur = sum(r_hist[n][-1] for n in names) / A
        avg = sum(sum(r_hist[n]) for n in names) / (A * len(r_hist[names[0]]))
        print(f"  {step:4d} | {float(env.omap.known_ratio()[0]):.4f} | "
              f"{int(env._task_valid[0].sum()):4d} | {cur:+8.3f} | {avg:+8.3f}", flush=True)

allr = [v for n in names for v in r_hist[n]]
import statistics as st
print(f"\n[OK] 最终 known_ratio = {[round(float(x),4) for x in env.omap.known_ratio()]}", flush=True)
print(f"[OK] 奖励: 均值={st.mean(allr):+.4f} 中位={st.median(allr):+.4f} "
      f"最小={min(allr):+.3f} 最大={max(allr):+.3f}", flush=True)
pos = sum(1 for v in allr if v > 0) / len(allr)
print(f"[OK] 正奖励步占比 = {pos*100:.1f}%", flush=True)
print("[DONE]", flush=True)
os._exit(0)
