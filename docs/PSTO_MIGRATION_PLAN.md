# PSTO-AAV-Pursuit -> TRACE migration plan

## Scope

The upstream snapshot in `upstream/psto-aav-pursuit` is pinned to
`764e2a156f4f74fc3d2a25dfaa6871c57ec7944`. TRACE keeps this reference untouched and
adds an Isaac Sim 6 adapter under `trace/pursuit`.

## Algorithm mapping

| Upstream component | TRACE port |
|---|---|
| 2 pursuers + 1 evader | `PSTOPursuitEnv` and `PSTOPursuitScene` |
| 24x480 LiDAR, 4x pooling | `PSTOLidarAdapter` |
| LiDAR + PSTO heatmap FullModel | `trace.pursuit.models` |
| Beta actor, centralized critic, PPO/GAE/ValueNorm | `trace.pursuit.algorithm.mappo` |
| LSTM historical trajectory prediction | `trajectory_prediction.py` |
| scripted repulsive evader | `evader.py` |
| capture/escape/collision reward | `rewards.py` |
| Hummingbird + QuadrotorActuator + RotorFlight | Isaac adapter in `scene.py` |

The policy action remains the upstream 2-D Beta action. It is converted to a body-frame
horizontal velocity target and then yaw-rotated into world coordinates. The target is
passed to TRACE's Lee controller; only rotor forces and torques advance PhysX.

## TRACE-specific adaptation

The upstream environment depends on OmniDrones/Isaac Sim 4.1. TRACE uses the existing
Isaac Sim 6.0.1 Hummingbird USD, `QuadrotorActuator`, `RotorFlight`, 240 Hz physics and
30 Hz policy clock. The upstream scan definition is retained; TRACE's old 36x4 scan is
never silently used as a PSTO scan. Raycast implementation and scene initialization are
isolated so the NavRL baseline defaults are unchanged.

## Status

- Imported and pinned upstream reference: complete.
- Pure Torch network/action/LSTM/heatmap/reward units: complete when tests pass.
- Isaac Sim 6 three-vehicle physical scene: in progress.
- PPO rollout/training and checkpoint compatibility: in progress.
- Paper-level training/metrics: not claimed until a real run completes.

## Second-stage acceptance results (2026-11-18)

A physical acceptance run was executed with `/home/wj/isaacsim/python.sh` and its kit
Python (`isaacsim.exp.base.python-6.0.1`), rather than the conda package that reports
Isaac 5.1. Command:

```bash
PYTHONEXE=/home/wj/isaacsim/kit/python/bin/python3 PYTHONPATH=$PWD \
  /home/wj/isaacsim/python.sh scripts/test_psto_scene.py --steps 2 --headless
```

The run passed for three independent Hummingbirds, one shared 240 Hz World, eight
physics substeps per 30 Hz policy step, finite state/rotor/LiDAR values, and independent
rotor control. Artifacts are in `results/psto_scene/`.

The pure Torch observation/MAPPO/LSTM suite passes 7 tests. `scripts/test_psto_update.py`
performed a real PPO and LSTM backward update, changed actor parameters, and restored all
module and optimizer states from `results/psto_update/checkpoint.pt`.

The RGB viewport file is produced by Isaac's render product but is black in headless mode
because no active viewport/display is available over SSH. The non-black, data-derived
three-dimensional physical trajectory plot is `results/psto_scene/psto_trajectory.png`.
