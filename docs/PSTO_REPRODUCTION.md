# PSTO reproduction notes

PSTO uses decentralized actors with a centralized critic. Each actor receives a 13-D
state (quaternion, body velocity, angular velocity, previous 3-D action), a downsampled
24x480 LiDAR image, and a heatmap encoding the evader/team intent. The heatmap uses
positive evader values and negative teammate values with temporal decay. Evader motion is
scripted using pursuer repulsion, obstacle repulsion, boundary bias and altitude control.

The actor samples two independent Beta variables. The first controls planar direction and
the second controls speed. The resulting body velocity is rotated by yaw and sent to the
Lee velocity controller. It is not a four-motor command.

The auxiliary LSTM consumes observed historical positions plus `agent_dt` and predicts
future trajectory points. Future ground truth is used only as a supervised target during
training; it is not exposed to the actor.

This repository separates the official snapshot from the TRACE adapter so dependency
changes for Isaac Sim 6 cannot alter the original implementation.

## P0 experiment entry points

The official 10 Hz time scale and the native 30 Hz adapter are separate configurations:

```bash
PYTHONPATH=/home/wj/isaacsim/extsDeprecated/omni.isaac.ml_archive/pip_prebundle:$PWD \
PYTHONEXE=/home/wj/isaacsim/kit/python/bin/python3 \
/home/wj/isaacsim/python.sh scripts/train_psto.py \
  --config configs/psto/official_reproduction.yaml --headless

PYTHONPATH=/home/wj/isaacsim/extsDeprecated/omni.isaac.ml_archive/pip_prebundle:$PWD \
PYTHONEXE=/home/wj/isaacsim/kit/python/bin/python3 \
/home/wj/isaacsim/python.sh scripts/train_psto.py \
  --config configs/psto/trace_native.yaml --headless
```

Run the physical checks before long training:

```bash
/home/wj/isaacsim/python.sh scripts/test_psto_physics.py --headless
/home/wj/isaacsim/python.sh scripts/profile_psto_lidar.py --headless
/home/wj/isaacsim/python.sh scripts/test_psto_episodes.py --episodes 10 --steps 280 --headless
```

The full P0 status and actual local evidence are recorded in
[`PSTO_P0_ACCEPTANCE_REPORT.md`](PSTO_P0_ACCEPTANCE_REPORT.md).
