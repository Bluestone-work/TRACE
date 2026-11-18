"""策略 rollout -> PNG 序列 + GIF，用来肉眼检查环境和策略行为是否可信。

和 07_visualize.py 的区别：07 自己搭场景、脚本化螺旋飞行，既不走环境也不走策略；
这里直接驱动 TraceExploreEnv —— 真观测、真奖励、真终止条件，策略可选
    random          随机动作（≈ 训练起点）
    greedy          09 的贪心启发式：朝 g_topo 指的 task node 飞（创新点三的对照）
    ckpt:<path>     训练好的 MAPPO checkpoint
一次启动跑完多组，出每组 GIF + 对比表 + 并排 GIF。

用法:
    cd upstream/IsaacLab
    python -u ../../isaac_tests/10_rollout_gif.py --headless \
        --spec random,greedy,ckpt:/abs/path/agent_960.pt,ckpt:/abs/path/best_agent.pt
"""
import argparse, os, sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--spec", type=str, default="random,greedy",
                    help="逗号分隔的策略列表：random / greedy / ckpt:<路径>")
parser.add_argument("--steps", type=int, default=450, help="每组最多跑多少控制步（450 = 一整个 episode）")
parser.add_argument("--every", type=int, default=15, help="每隔多少步出一帧（15 步 = 1 秒仿真时间）")
parser.add_argument("--task", type=str, default="Trace-Explore-MARL-v0")
parser.add_argument("--out", type=str, default=None)
parser.add_argument("--stochastic", action="store_true", help="checkpoint 用采样动作而非分布均值")
parser.add_argument("--keep_going", action="store_true",
                    help="终止后继续跑（环境会自动重置、地图清零，帧上会标出）")
parser.add_argument("--seed", type=int, default=42)
# 去中心化后观测多了 MsgAge 维（163 -> 164），2026-11-01 训练的 checkpoint 是 163 维输入，
# 所以这里默认退回共享图基线才能加载旧权重。要看去中心化环境的行为就加 --decentralized。
parser.add_argument("--decentralized", action="store_true",
                    help="用每机独立 belief 的去中心化环境（与旧 checkpoint 不兼容）")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import imageio.v2 as imageio
import numpy as np
import torch
import gymnasium as gym
from isaaclab_rl.skrl import SkrlVecEnvWrapper
from isaaclab_tasks.utils.hydra import load_cfg_from_registry

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import trace.tasks  # noqa: F401  —— 完成 gym 注册
from trace.mapping.frontier import frontier_mask
from trace.viz.map_viz import save_frame

out_root = args.out or os.path.join(ROOT, "runs", "viz", "rollout")
os.makedirs(out_root, exist_ok=True)

# ---------------- 解析 spec ----------------
specs = []
for raw in [s for s in args.spec.split(",") if s.strip()]:
    s = raw.strip()
    if s in ("random", "greedy"):
        specs.append({"tag": s, "kind": s, "path": None})
    elif s.startswith("ckpt:"):
        p = s[len("ckpt:"):]
        if not os.path.isfile(p):
            raise FileNotFoundError(f"checkpoint 不存在: {p}")
        specs.append({"tag": "ckpt_" + os.path.splitext(os.path.basename(p))[0], "kind": "ckpt", "path": p})
    else:
        raise ValueError(f"无法识别的 spec: {s}")
need_agent = any(s["kind"] == "ckpt" for s in specs)
print(f"[OK] 将跑 {len(specs)} 组: {[s['tag'] for s in specs]}", flush=True)

# ---------------- 建环境 ----------------
env_cfg = load_cfg_from_registry(args.task, "env_cfg_entry_point")
env_cfg.scene.num_envs = 1
env_cfg.sim.device = args.device
env_cfg.seed = args.seed
env_cfg.decentralized = args.decentralized
env_cfg.__post_init__()
raw_env = gym.make(args.task, cfg=env_cfg)
env = SkrlVecEnvWrapper(raw_env, ml_framework="torch")
base = env.unwrapped                                  # TraceExploreEnv，拿 omap / task graph
names = list(base.cfg.possible_agents)
fcfg = base.cfg.frontier_cfg

agent = None
if need_agent:
    from skrl.utils.runner.torch import Runner
    agent_cfg = load_cfg_from_registry(args.task, "skrl_mappo_cfg_entry_point")
    agent_cfg["seed"] = args.seed
    agent_cfg["trainer"]["timesteps"] = 1            # 只借 Runner 建网络，不训练
    agent = Runner(env, agent_cfg).agent
    print("[OK] MAPPO 网络已建好（CTDE: policy 吃观测, critic 吃特权状态）", flush=True)


def act(kind, obs, states):
    """按策略算动作。返回 {agent_name: (1,3)} 的速度指令。"""
    if kind == "random":
        return {n: torch.rand(1, 3, device=base.device) * 2 - 1 for n in names}
    if kind == "greedy":
        out = {}
        for n in names:
            g = obs[n][:, -3:]                        # 末 3 维 = g_topo，选中 task 的相对方向
            a = g / g.norm(dim=-1, keepdim=True).clamp(min=1e-6)
            a[:, 2] *= 0.3                            # 抑制垂直分量
            out[n] = a.clamp(-1, 1)
        return out
    a, outputs = agent.act(obs, states, timestep=0, timesteps=1)
    if args.stochastic:
        return a
    return {n: outputs[n]["mean_actions"] for n in names}


def rollout(spec):
    """跑一组，出帧 + GIF，返回统计。"""
    tag, kind = spec["tag"], spec["kind"]
    d = os.path.join(out_root, tag)
    os.makedirs(d, exist_ok=True)
    if kind == "ckpt":
        agent.load(spec["path"])
        agent.enable_training_mode(False)             # eval：不建 memory、不算 value

    torch.manual_seed(args.seed)
    # wrapper 的 reset() 只在第一次真重置（skrl IsaacLabMultiAgentWrapper._reset_once），
    # 不置回 True 的话第二组会接着第一组的地图跑，各组起点不同、根本没法比。
    env._reset_once = True
    obs, _ = env.reset()
    states = env.state()
    trails = [[] for _ in names]
    r_hist, kr_hist, frames = [], [], []
    n_coll, n_reset, reason = 0, 0, f"跑满 {args.steps} 步"
    print(f"\n=== [{tag}] ===\n  step | known  | task | 奖励/步 | 备注", flush=True)

    for step in range(args.steps):
        with torch.no_grad():
            actions = act(kind, obs, states)
        obs, rew, term, trunc, info = env.step(actions)
        states = env.state()
        r_step = float(sum(rew[n].mean() for n in names) / len(names))
        truncated = bool(trunc[names[0]][0])
        terminated = bool(term[names[0]][0])
        done = terminated or truncated

        # known_ratio 必须从 info 取：环境在 step 内部已经自动重置过，
        # 这时读 omap 拿到的是清零后的新地图（_get_rewards 里写的 log 才是重置前的真值）。
        log = info.get("log", {}) if isinstance(info, dict) else {}
        kr = float(log["known_ratio"]) if "known_ratio" in log else float(base.omap.known_ratio()[0])
        nt = int(float(log["n_task_nodes"])) if "n_task_nodes" in log else int(base._task_valid[0].sum())
        r_hist.append(r_step)
        kr_hist.append(kr)

        if done:
            # 同理，_collided 已被 _reset_idx 清零，只能用 term/trunc + 覆盖率反推原因
            reason = ("episode 超时" if truncated
                      else f"探索达标 (known>={base.cfg.done_known_ratio:.2f})"
                      if kr >= base.cfg.done_known_ratio else "碰撞终止")
            print(f"  {step:4d} | {kr:.4f} | {nt:4d} | {r_step:+7.2f} | 终止：{reason}", flush=True)
            if not args.keep_going:
                break
            n_reset += 1
            trails = [[] for _ in names]
            continue                                  # 重置后的位姿/地图不入帧

        collided = bool(base._collided[0].any())
        n_coll += int(collided)
        pos = [(dr.data.root_pos_w - base.scene.env_origins)[0].detach().cpu().numpy() for dr in base.drones]
        for i, pp in enumerate(pos):
            trails[i].append((float(pp[0]), float(pp[1])))

        if step % args.every == 0 or step == args.steps - 1:
            fm = frontier_mask(base.omap, fcfg)
            nodes = {"pos": base._task_pos, "n_unknown": base._task_nunk, "valid": base._task_valid}
            f = os.path.join(d, f"frame_{step:04d}.png")
            save_frame(f, base.omap, 0, fmask=fm, task_nodes=nodes, drone_pos=pos, trails=trails,
                       title=f"[{tag}]  step {step}  known={kr:.3f}  "
                             f"task nodes={nt}  r={r_step:+.2f}" + ("  <碰撞>" if collided else ""))
            frames.append(f)
            print(f"  {step:4d} | {kr:.4f} | {nt:4d} | {r_step:+7.2f} |"
                  f"{' 碰撞' if collided else ''}", flush=True)

    gif = os.path.join(out_root, f"{tag}.gif")
    imageio.mimsave(gif, [imageio.imread(f) for f in frames], duration=0.35, loop=0)
    # 航程：轨迹折线长度（只算 xy，z 幅度很小）
    path_len = [float(np.linalg.norm(np.diff(np.array(t), axis=0), axis=1).sum()) if len(t) > 1 else 0.0
                for t in trails]
    st = {"tag": tag, "steps": len(r_hist), "known": max(kr_hist) if kr_hist else 0.0,
          "r_mean": float(np.mean(r_hist)) if r_hist else 0.0,
          "r_total": float(np.sum(r_hist)) if r_hist else 0.0,
          "coll_steps": n_coll, "path": float(np.mean(path_len)), "reason": reason,
          "frames": frames, "gif": gif}
    print(f"[OK] {tag}: {len(frames)} 帧, 最终 known={st['known']:.4f} -> {gif}", flush=True)
    return st


stats = [rollout(s) for s in specs]

# ---------------- 对比表 ----------------
hdr = f"\n{'策略':<22} {'步数':>5} {'最终known':>10} {'奖励/步':>9} {'总奖励':>9} {'碰撞步':>7} {'平均航程(m)':>12}  终止原因"
lines = [hdr, "-" * 104]
for s in stats:
    lines.append(f"{s['tag']:<22} {s['steps']:>5} {s['known']:>10.4f} {s['r_mean']:>9.3f} "
                 f"{s['r_total']:>9.1f} {s['coll_steps']:>7} {s['path']:>12.1f}  {s['reason']}")
table = "\n".join(lines)
print(table, flush=True)
with open(os.path.join(out_root, "summary.txt"), "w") as fh:
    fh.write(table + "\n")

# ---------------- 并排 GIF ----------------
if len(stats) > 1:
    n_max = max(len(s["frames"]) for s in stats)
    seqs = []
    for s in stats:                                   # 短的用最后一帧补齐，便于同步对比
        imgs = [imageio.imread(f)[:, :, :3] for f in s["frames"]]
        imgs += [imgs[-1]] * (n_max - len(imgs))
        seqs.append(imgs)
    cols = 2 if len(seqs) > 2 else len(seqs)
    rows = int(np.ceil(len(seqs) / cols))
    blank = np.full_like(seqs[0][0], 255)
    combo = []
    for k in range(n_max):
        grid = []
        for r in range(rows):
            row = [seqs[r * cols + c][k] if r * cols + c < len(seqs) else blank for c in range(cols)]
            grid.append(np.hstack(row))
        combo.append(np.vstack(grid))
    cg = os.path.join(out_root, "compare.gif")
    imageio.mimsave(cg, combo, duration=0.35, loop=0)
    print(f"[OK] 并排对比 GIF: {cg}  ({rows}x{cols}, {n_max} 帧)", flush=True)

print("[DONE] rollout 可视化完成", flush=True)
os._exit(0)
