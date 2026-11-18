"""TRACE-Explorer MAPPO (CTDE) 训练入口。

IsaacLab 自带的 train.py 只注册 isaaclab_tasks 里的任务，所以这里自己起一个
入口，先 import trace.tasks 完成 gym 注册，再走和官方脚本一样的
SkrlVecEnvWrapper + skrl Runner 流程。

用法:
    python scripts/train_trace_marl.py --headless --num_envs 4 --max_iterations 5
"""
import argparse, os, sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Train TRACE-Explorer with MAPPO.")
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--max_iterations", type=int, default=None, help="覆盖 trainer.timesteps")
parser.add_argument("--task", type=str, default="Trace-Explore-MARL-v0")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--task_graph_every", type=int, default=None,
                    help="任务图重算间隔(控制步)。CCL 约 150ms/env，env 多时调大可省吞吐")
parser.add_argument("--episode_length_s", type=float, default=None)
parser.add_argument("--group", type=str, default=None,
                    help="消融分组：base / belief / belief_comm / belief_marginal。"
                         "不给则用 env_cfg 的默认值")
parser.add_argument("--num_agents", type=int, default=None)
parser.add_argument("--run_dir", type=str, default=None,
                    help="experiment.directory，默认 trace_explore_marl")
parser.add_argument("--tag", type=str, default=None, help="experiment_name，便于区分分组/种子")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
from isaaclab_rl.skrl import SkrlVecEnvWrapper
from isaaclab_tasks.utils.hydra import load_cfg_from_registry
from skrl.utils.runner.torch import Runner

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import trace.tasks  # noqa: F401  —— 必须先 import 才完成 gym 注册

env_cfg = load_cfg_from_registry(args_cli.task, "env_cfg_entry_point")
agent_cfg = load_cfg_from_registry(args_cli.task, "skrl_mappo_cfg_entry_point")

env_cfg.scene.num_envs = args_cli.num_envs
env_cfg.sim.device = args_cli.device
env_cfg.seed = args_cli.seed
if args_cli.num_agents is not None:
    env_cfg.num_agents = args_cli.num_agents
    env_cfg.__post_init__()

group_summary = None
if args_cli.group is not None:
    from scripts.ablation_groups import apply_group
    group_summary = apply_group(env_cfg, args_cli.group)
    # 分组会改观测维度，必须把 agent_cfg 里的空间定义同步过去（skrl 从 env 读，
    # 所以这里不用动网络；但日志要记下来以便核对）
    print(f"[INFO] 分组 {args_cli.group}: obs_dim={group_summary['obs_dim']} "
          f"decentralized={group_summary['decentralized']} "
          f"reward_mode={group_summary['reward_mode']} comm={group_summary['comm']}", flush=True)
if args_cli.task_graph_every is not None:
    env_cfg.task_graph_every = args_cli.task_graph_every
if args_cli.episode_length_s is not None:
    env_cfg.episode_length_s = args_cli.episode_length_s
agent_cfg["seed"] = args_cli.seed
if args_cli.run_dir is not None:
    agent_cfg["agent"]["experiment"]["directory"] = args_cli.run_dir
if args_cli.tag is not None:
    agent_cfg["agent"]["experiment"]["experiment_name"] = args_cli.tag
if args_cli.max_iterations is not None:
    agent_cfg["trainer"]["timesteps"] = args_cli.max_iterations * agent_cfg["agent"]["rollouts"]

print(f"[INFO] 任务={args_cli.task} num_envs={args_cli.num_envs} "
      f"timesteps={agent_cfg['trainer']['timesteps']} "
      f"task_graph_every={env_cfg.task_graph_every} "
      f"episode_length_s={env_cfg.episode_length_s}", flush=True)
print(f"[INFO] 总 transition 数 ≈ {agent_cfg['trainer']['timesteps']*args_cli.num_envs*env_cfg.num_agents}", flush=True)

# 把解析后的完整配置落盘。第一阶段发现 skrl 什么都不存，导致 11-01 那次训练的
# num_envs 之类参数事后无从查证、复现不了。
import json
run_root = agent_cfg["agent"]["experiment"]["directory"]
os.makedirs(run_root, exist_ok=True)
meta = {
    "argv": sys.argv,
    "task": args_cli.task,
    "seed": args_cli.seed,
    "num_envs": args_cli.num_envs,
    "num_agents": int(env_cfg.num_agents),
    "timesteps": agent_cfg["trainer"]["timesteps"],
    "rollouts": agent_cfg["agent"]["rollouts"],
    "group": group_summary,
    "env": {
        "decentralized": bool(env_cfg.decentralized),
        "reward_mode": env_cfg.reward_mode,
        "use_true_velocity": bool(env_cfg.use_true_velocity),
        "task_graph_every": env_cfg.task_graph_every,
        "task_graph_min_new_voxels": env_cfg.task_graph_min_new_voxels,
        "task_graph_stagger": bool(env_cfg.task_graph_stagger),
        "episode_length_s": env_cfg.episode_length_s,
        "obs_dim": int(next(iter(env_cfg.observation_spaces.values()))),
        "state_dim": int(env_cfg.state_space),
        "comm": None if not env_cfg.decentralized else {
            "enabled": env_cfg.comm.enabled, "range_m": env_cfg.comm.range_m,
            "every": env_cfg.comm.every, "drop_prob": env_cfg.comm.drop_prob,
            "delay_steps": env_cfg.comm.delay_steps, "max_age": env_cfg.comm.max_age,
        },
    },
}
tag = args_cli.tag or "run"
with open(os.path.join(run_root, f"cfg_{tag}.json"), "w") as fh:
    json.dump(meta, fh, indent=2, ensure_ascii=False)
print(f"[INFO] 配置已落盘: {run_root}/cfg_{tag}.json", flush=True)

env = gym.make(args_cli.task, cfg=env_cfg)
env = SkrlVecEnvWrapper(env, ml_framework="torch")
runner = Runner(env, agent_cfg)
runner.run()

print("[DONE] 训练流程结束", flush=True)
env.close()
os._exit(0)
