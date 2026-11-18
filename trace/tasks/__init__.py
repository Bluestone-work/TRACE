"""把 TRACE 环境注册成 gym 任务，这样能直接用 IsaacLab 自带的 RL 脚本训练。

用法（在 upstream/IsaacLab 目录下）:

    python scripts/reinforcement_learning/skrl/train.py \
        --task Trace-Explore-MARL-v0 --algorithm MAPPO --headless --num_envs 4

注意：必须先 import trace.tasks 才会完成注册，训练脚本用 --task 时
要确保 PYTHONPATH 包含 TRACE_baseline 根目录，并在脚本里 import。
"""

import gymnasium as gym

from . import agents

gym.register(
    id="Trace-Explore-MARL-v0",
    entry_point="trace.envs.trace_explore_env:TraceExploreEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "trace.envs.trace_explore_env:TraceExploreEnvCfg",
        "skrl_mappo_cfg_entry_point": f"{agents.__name__}:skrl_mappo_cfg.yaml",
        "skrl_ippo_cfg_entry_point": f"{agents.__name__}:skrl_mappo_cfg.yaml",
    },
)
