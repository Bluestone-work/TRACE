"""跑一段 rollout，每隔 N 步导出一个 .npz 快照，供 Blender 侧渲成 3D GIF。

和 08_dump_for_render.py 的区别：08 是脚本化螺旋飞行、只导出最后一帧；
这里走真环境 TraceExploreEnv + 可选策略（random/greedy/ckpt），导出整段序列。

用法:
    cd upstream/IsaacLab
    python -u ../../isaac_tests/11_dump_sequence.py --headless --steps 450 --every 15 \
        --policy ckpt --ckpt /abs/path/agent_960.pt --tag ckpt_agent_960
"""
import argparse, os, sys
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--policy", type=str, default="greedy", choices=["random", "greedy", "ckpt"])
parser.add_argument("--ckpt", type=str, default=None)
parser.add_argument("--tag", type=str, default=None)
parser.add_argument("--steps", type=int, default=450)
parser.add_argument("--every", type=int, default=15)
parser.add_argument("--task", type=str, default="Trace-Explore-MARL-v0")
parser.add_argument("--out", type=str, default=None)
parser.add_argument("--stochastic", action="store_true")
parser.add_argument("--seed", type=int, default=42)
# 去中心化后观测多了 MsgAge 维（163 -> 164），2026-11-01 训练的 checkpoint 是 163 维输入，
# 所以这里默认退回共享图基线才能加载旧权重。要看去中心化环境的行为就加 --decentralized。
parser.add_argument("--decentralized", action="store_true",
                    help="用每机独立 belief 的去中心化环境（与旧 checkpoint 不兼容）")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import torch
import gymnasium as gym
from isaaclab_rl.skrl import SkrlVecEnvWrapper
from isaaclab_tasks.utils.hydra import load_cfg_from_registry

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import trace.tasks  # noqa: F401
from trace.mapping.frontier import frontier_mask, label_components, extract_task_nodes
from trace.viz.dump_scene import dump_snapshot

tag = args.tag or args.policy
out_dir = args.out or os.path.join(ROOT, "runs", "viz", "seq3d", tag)
os.makedirs(out_dir, exist_ok=True)

env_cfg = load_cfg_from_registry(args.task, "env_cfg_entry_point")
env_cfg.scene.num_envs = 1
env_cfg.sim.device = args.device
env_cfg.seed = args.seed
env_cfg.decentralized = args.decentralized
env_cfg.__post_init__()
env = SkrlVecEnvWrapper(gym.make(args.task, cfg=env_cfg), ml_framework="torch")
base = env.unwrapped
names = list(base.cfg.possible_agents)
fcfg = base.cfg.frontier_cfg

agent = None
if args.policy == "ckpt":
    from skrl.utils.runner.torch import Runner
    acfg = load_cfg_from_registry(args.task, "skrl_mappo_cfg_entry_point")
    acfg["seed"] = args.seed
    acfg["trainer"]["timesteps"] = 1
    agent = Runner(env, acfg).agent
    agent.load(args.ckpt)
    agent.enable_training_mode(False)
    print(f"[OK] 已载入 {args.ckpt}", flush=True)


def act(obs, states):
    if args.policy == "random":
        return {n: torch.rand(1, 3, device=base.device) * 2 - 1 for n in names}
    if args.policy == "greedy":
        out = {}
        for n in names:
            g = obs[n][:, -3:]
            a = g / g.norm(dim=-1, keepdim=True).clamp(min=1e-6)
            a[:, 2] *= 0.3
            out[n] = a.clamp(-1, 1)
        return out
    a, outputs = agent.act(obs, states, timestep=0, timesteps=1)
    return a if args.stochastic else {n: outputs[n]["mean_actions"] for n in names}


torch.manual_seed(args.seed)
env._reset_once = True                      # wrapper 只在第一次真重置，见 10_rollout_gif.py
obs, _ = env.reset()
states = env.state()
trails = [[] for _ in names]
n_dump, reason = 0, f"跑满 {args.steps} 步"
print(f"[OK] 策略={tag} 输出目录={out_dir}", flush=True)

for step in range(args.steps):
    with torch.no_grad():
        actions = act(obs, states)
    obs, rew, term, trunc, info = env.step(actions)
    states = env.state()
    truncated, terminated = bool(trunc[names[0]][0]), bool(term[names[0]][0])
    log = info.get("log", {}) if isinstance(info, dict) else {}
    kr = float(log["known_ratio"]) if "known_ratio" in log else float(base.omap.known_ratio()[0])

    if terminated or truncated:
        reason = ("episode 超时" if truncated
                  else f"探索达标 (known>={base.cfg.done_known_ratio:.2f})"
                  if kr >= base.cfg.done_known_ratio else "碰撞终止")
        print(f"  第 {step} 步终止：{reason}  known={kr:.4f}", flush=True)
        break

    # 轨迹带 z，3D 渲染要用
    pos = [(d.data.root_pos_w - base.scene.env_origins)[0].detach().cpu().numpy() for d in base.drones]
    for i, p in enumerate(pos):
        trails[i].append((float(p[0]), float(p[1]), float(p[2])))

    if step % args.every == 0 or step == args.steps - 1:
        fm = frontier_mask(base.omap, fcfg)
        labels, _ = label_components(fm, fcfg)
        nodes = extract_task_nodes(base.omap, labels, fcfg, max_nodes=base.cfg.max_task_nodes)
        p = os.path.join(out_dir, f"snap_{step:04d}.npz")
        dump_snapshot(p, base.omap, 0, fmask=fm, task_nodes=nodes,
                      drone_pos=pos, trails=trails, step=step)
        n_dump += 1
        print(f"  step {step:4d}  known={kr:.4f}  occ={int(base.omap.occupied_mask[0].sum())} "
              f"frontier={int(fm[0].sum())} task={int(nodes['valid'][0].sum())} -> {os.path.basename(p)}",
              flush=True)

print(f"[OK] 共导出 {n_dump} 个快照到 {out_dir}（终止原因：{reason}）", flush=True)
print("[DONE]", flush=True)
os._exit(0)
