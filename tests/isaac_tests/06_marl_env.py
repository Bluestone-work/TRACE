"""TraceExploreEnv 端到端验证：建图 -> 任务图 -> 观测 -> 动作 -> 奖励。"""
import argparse, os, sys
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--steps", type=int, default=120)
parser.add_argument("--num_envs", type=int, default=2)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from trace.envs.trace_explore_env import TraceExploreEnv, TraceExploreEnvCfg, _obs_dim, _state_dim

cfg = TraceExploreEnvCfg()
cfg.scene.num_envs = args.num_envs
cfg.sim.device = args.device
env = TraceExploreEnv(cfg)
print(f"[OK] 环境创建成功: num_envs={env.num_envs}, agents={env.cfg.possible_agents}", flush=True)
print(f"[OK] 控制频率 = {1/(cfg.sim.dt*cfg.decimation):.1f} Hz, "
      f"建图 {1/(cfg.sim.dt*cfg.decimation*cfg.map_update_every):.1f} Hz, "
      f"任务图 {1/(cfg.sim.dt*cfg.decimation*cfg.task_graph_every):.2f} Hz", flush=True)

obs, _ = env.reset()
for a in env.cfg.possible_agents:
    assert obs[a].shape == (env.num_envs, _obs_dim(cfg.num_agents, cfg.decentralized)), \
        f"{a} 观测维度 {tuple(obs[a].shape)} != {(env.num_envs, _obs_dim(cfg.num_agents, cfg.decentralized))}"
print(f"[OK] 观测维度 = {tuple(obs[env.cfg.possible_agents[0]].shape)} (期望 {_obs_dim(cfg.num_agents, cfg.decentralized)})", flush=True)
st = env.state()
assert st.shape == (env.num_envs, _state_dim(cfg.num_agents))
print(f"[OK] 特权状态维度 = {tuple(st.shape)} (期望 {_state_dim(cfg.num_agents)})", flush=True)

print("\n  step | known  | task节点 | 平均奖励 | 观测有限 | 终止", flush=True)
for step in range(args.steps):
    acts = {a: torch.rand(env.num_envs, 3, device=env.device) * 2 - 1
            for a in env.cfg.possible_agents}
    obs, rew, term, trunc, _ = env.step(acts)
    if step % 20 == 0 or step == args.steps - 1:
        r = torch.stack([rew[a] for a in env.cfg.possible_agents]).mean()
        fin = all(bool(torch.isfinite(obs[a]).all()) for a in env.cfg.possible_agents)
        nt = int(env._task_valid[0].sum())
        print(f"  {step:4d} | {float(env.omap.known_ratio()[0]):.4f} | {nt:7d} |"
              f" {float(r):+8.3f} | {str(fin):8s} | {int(term[env.cfg.possible_agents[0]].sum())}", flush=True)
        assert fin, "观测出现非有限值"
        assert torch.isfinite(r), "奖励非有限"

assert float(env.omap.known_ratio()[0]) > 0.05, "覆盖率过低，建图可能未生效"
print(f"\n[OK] 最终 known_ratio = {[round(float(x),4) for x in env.omap.known_ratio()]}", flush=True)
print(f"[OK] 各 env task 节点数 = {[int(x) for x in env._task_valid.sum(dim=1)]}", flush=True)
print("[DONE] MARL 环境端到端验证通过", flush=True)
os._exit(0)
