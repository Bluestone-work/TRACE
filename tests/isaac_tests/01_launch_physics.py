import argparse, os
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app
print("[OK] AppLauncher 启动成功（未崩溃）", flush=True)

import torch
import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationCfg, SimulationContext

print("[OK] torch:", torch.__version__, "cuda:", torch.cuda.is_available(),
      "ngpu:", torch.cuda.device_count(), flush=True)

sim = SimulationContext(SimulationCfg(dt=1/120, device="cuda:0"))
print("[OK] SimulationContext 创建完成, device =", sim.device, flush=True)

cfg_ground = sim_utils.GroundPlaneCfg()
cfg_ground.func("/World/defaultGroundPlane", cfg_ground)

cfg_light = sim_utils.DomeLightCfg(intensity=2000.0)
cfg_light.func("/World/Light", cfg_light)
print("[OK] 场景创建完成（地面 + 光照）", flush=True)

sim.reset()
print("[OK] sim.reset() 完成", flush=True)

for i in range(200):
    sim.step()
print("[OK] 物理步进 200 步完成, 仿真时间 = %.3f s" % (200/120), flush=True)

print("[DONE] Isaac Lab headless 全链路验证通过", flush=True)
os._exit(0)   # 不调 close()：Isaac Sim 的 close() 会无限挂起
