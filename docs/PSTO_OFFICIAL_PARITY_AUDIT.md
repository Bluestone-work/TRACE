# PSTO official parity audit

Audit date: 2026-11-20. The official source is the independent checkout
`upstream/psto-aav-pursuit-official` at HITSZ-MAS commit
`764e2a156f4f74fc3d24a25dfaa6871c57ec7944`. TRACE was audited at Git commit
`0d62c4f` before the audit-only files in this change. Isaac Sim is
`6.0.1` (build `110.1.2`), Python `3.12.13`, PyTorch `2.11.0+cu128`, CUDA
`12.8`; the conda unit-test environment is Python `3.11.18`, PyTorch
`2.7.0+cu128`.

`EXACT` means the same calculation was numerically tested. `APPROXIMATE`
means the dimensions or formula are similar but the runtime path differs.
`DIFFERENT` means the behavior or dependency is materially different.
`UNKNOWN` means the official runtime was not available for a direct test.

The audit artifacts and tests are delivered at final TRACE commit `c26a8a2`.

| Module | Official implementation | TRACE implementation | Status | Difference and impact | Evidence |
|---|---|---|---|---|---|
| Official source | OmniDrones/Isaac Sim paper code | Isaac Sim 6 adapter | DIFFERENT | The official release documents Isaac Sim 4.1.0, Python 3.10, OmniDrones, TorchRL 0.3.1. TRACE runs Python 3.12/Isaac 6.0.1. | `upstream/psto-aav-pursuit-official/ENVIRONMENT.md:5-19`; `trace/pursuit/scene.py:70-88` |
| Hummingbird asset | OmniDrones `MultirotorBase` and its registered Hummingbird | Hummingbird USD through `SingleArticulation` | UNKNOWN | The USD path is shared with the vendored OmniDrones asset, but mass/inertia, rotor joints, actuator wiring, and contact materials were not proven numerically equivalent. | `env.py:275-284`; `trace/pursuit/scene.py:23-36` |
| Controller and rotor dynamics | OmniDrones `LeePositionController`, `drone.apply_action` | NavRL-derived `QuadrotorActuator` and `RotorFlight` | DIFFERENT | Both use velocity targets, but controller implementation, motor interpolation, wrench application, and PhysX integration are different. Collision rate cannot be attributed to learning alone until this is calibrated. | `env.py:615-663`; `trace/pursuit/scene.py:112-120`; `upstream/NavRL/isaac6-training/{quadrotor_dynamics.py,rotor_flight.py}` |
| Physics/control rate | Official `dt=0.01`, `frame_skip=10`, 10 Hz policy | `physics_dt=1/240`, 8 substeps, 30 Hz policy | DIFFERENT | The policy sees and acts on a different time scale; the official 10-step LSTM time interval is 0.1 s while TRACE uses about 0.0333 s. | `configs/psto/official_reproduction.yaml`; `configs/psto/trace_native_long.yaml`; `env.py:136-137,980` |
| Initial positions | `(-3,2,2),(-3,-2,2),(0,0,2)` | Same default positions | EXACT | The fixed single-environment initial geometry is identical. | `env.py:283-285`; `trace/pursuit/scene.py:74-75` |
| Terrain/obstacles | Ground plane plus `TerrainImporter` and `HfDiscreteObstaclesTerrainCfg` (18 obstacles, generated terrain) | Ground `FixedCuboid` plus 18 independent `FixedCuboid`s with random sizes/centers | DIFFERENT | Terrain height field, obstacle dimensions/distribution, collision mesh, and reset layout differ. The TRACE 100k collision rate is therefore not an official-environment result. | `env.py:296-340`; `trace/pursuit/scene.py:79-87` |
| Reset semantics | Smart reset positions after early collisions and per-env progress handling | Fixed/randomized pose reset; no official smart-reset sequence | DIFFERENT | Official avoids repeatedly starting in bad collision states; TRACE resets from its own RNG/layout contract. | `env.py:555-614,1190-1214`; `trace/pursuit/scene.py:91-111` |
| LiDAR geometry | RayCaster over terrain mesh, 24 vertical beams, 0.75° horizontal resolution, 10 m range | 24x480 physics-scene raycasts, same nominal FOV/range, self-prim filtering | APPROXIMATE / UNKNOWN | Nominal shape and range match, but ray origin, beam ordering, hit representation, mesh filtering, and no-hit behavior were not directly compared in the official runtime. | `env.py:125-184,786-807`; `trace/pursuit/scene.py:40-53`; `tests/psto_parity/test_lidar_parity.py` |
| LiDAR downsampling | 4x4 `F.max_pool2d`, resulting 6x120 | Same 4x4 max pooling | EXACT | Pure tensor operation passes the parity test. | `env.py:33-38,797-807`; `trace/pursuit/lidar.py:7-10`; `tests/psto_parity/test_lidar_parity.py` |
| Centralized critic state | Local positions (9) + 10-D state per drone (30) + previous actions (9) + 18-bin pursuer LiDAR (36) = 84 | Same 84-D layout for the single environment | EXACT | Shape and pooling test pass. TRACE uses world positions, which equal local positions only because `num_envs=1` has zero offset. | `env.py:949-970`; `trace/pursuit/env.py:19-24,37-38` |
| Beta actor | `BetaActor`, alpha/beta `1+Softplus+1e-6`, independent Beta distribution | Same alpha/beta parameterization and independent Beta | EXACT | Pure model/action tests pass. TRACE uses a dependency-light distribution instead of TorchRL's `ProbabilisticActor`. | `nav_utils.py:87-115`; `trace/pursuit/algorithm/mappo.py:33-40,87-90`; `tests/psto_parity/test_action_parity.py` |
| Action conversion | Beta `[direction,magnitude]` → body planar velocity → yaw-only world transform | Same formula and yaw-only transform | EXACT | Numeric action parity passes. | `mappo.py:210-250`; `trace/pursuit/algorithm/mappo.py:91-96`; `tests/psto_parity/test_action_parity.py` |
| Physical action application | World velocity sent to Lee controller and rotor articulation | World velocity sent to custom RotorFlight/PhysX actuator | DIFFERENT | The abstract actor action is aligned, but the downstream flight dynamics are not. | `env.py:621-663`; `trace/pursuit/env.py:102-110` |
| Heatmap projection | World→body, azimuth/elevation bins, signed max/min pooling, `10/dist`, decay | Same projection implementation | EXACT for projection | The independent numeric projection test passes. The caller differs: official builds an unmasked true current evader channel and true-history training buffer, while TRACE masks actor heatmap channels when invisible. | `env.py:729-782,813-918`; `trace/pursuit/observations.py:7-28`; `tests/psto_parity/test_observation_parity.py` |
| Visibility | Distance gate plus ray-hit alignment and obstacle-clearance test | Same thresholds applied to proximity-converted raycasts | APPROXIMATE | No-hit and target-hit representation differ between RayCaster and native raycast adapter; exact runtime parity is UNKNOWN. | `env.py:680-727`; `trace/pursuit/visibility.py:5-13` |
| LSTM architecture | 3-D position plus optional feature input, one-layer LSTM, 64 hidden, 10×3 output | Same architecture; TRACE fixes device indexing and valid-last-sample handling | APPROXIMATE | Architecture matches for all-valid sequences. Masked sequence behavior intentionally differs from the official right-padding assumption. | `lstm.py:6-59`; `trace/pursuit/trajectory_prediction.py:6-63`; `tests/psto_parity/test_model_parity.py` |
| LSTM data | Official history is `evader_full_history` filled from true state every step; `agent_dt` repeated at 10 Hz | TRACE separates observation history from teacher labels and uses sparse elapsed intervals/occlusion augmentation | DIFFERENT | This prevents actor leakage but changes the training distribution and is not official algorithm parity. | `env.py:820-845,972-980`; `trace/pursuit/env.py:61-101`; `trace/pursuit/algorithm/mappo.py:97-177` |
| GAE | `terminated` masks both TD bootstrap and recursive GAE | `terminated` masks bootstrap; `episode_end` additionally stops recursion across truncation | DIFFERENT by design | TRACE preserves timeout bootstrap, which is semantically correct but differs from the current official helper. | `nav_utils.py:117-148`; `trace/pursuit/algorithm/mappo.py:42-52` |
| PPO/update | TensorDict/TorchRL actor, 4 epochs×12 minibatches, Huber value loss, separate optimizers | Plain PyTorch storage/update, configurable epochs/minibatches, separate optimizers | APPROXIMATE | Default formulas and dimensions are close, but TensorDict batching, value-loss coefficient, scheduler construction, and persistent LSTM minibatch counter differ. | `mappo.py:254-410`; `trace/pursuit/algorithm/mappo.py:114-180` |
| ValueNorm | Running mean/variance with beta .995 and variance floor .01 | Same equations and buffers | EXACT | Numerically covered by the existing PSTO unit tests. | `nav_utils.py:11-74`; `trace/pursuit/algorithm/mappo.py:18-31` |
| Evader policy | Pursuer repulsion + LiDAR obstacle repulsion + boundary projection + altitude control | Same forces, but native hit points supplied by TRACE | APPROXIMATE | TRACE currently omits the official `hit_dist_from_center < arena_radius-1.5` internal-obstacle filter. Local force parity only covers hits inside the arena. | `env.py:451-545`; `trace/pursuit/evader.py:7-23`; `tests/psto_parity/test_evader_policy_parity.py` |
| Reward formula | Pursuit, coordination, obstacle log/quantile, formation, terminal bonuses/penalties | Same numeric terms for supplied inputs | EXACT for formula | Pure reward tests pass. | `env.py:996-1107`; `trace/pursuit/rewards.py:5-26`; `tests/psto_parity/test_reward_parity.py` |
| Collision source | Pursuer collision from downsampled LiDAR proximity; evader collision from evader LiDAR hit distance | Geometric AABB/pose thresholds for vehicles and cuboid obstacles | DIFFERENT | This is the leading environment/measurement discrepancy behind the 70% collision result; it must be diagnosed with per-step PhysX contact logging. | `env.py:1034-1038,1108-1124`; `trace/pursuit/env.py:39-59` |
| Termination | Capture, collision, below-bound, escape; official suppresses initial-step termination and separates terminated/truncated | Same event categories, but no official initial-step suppression and local collision fallback | DIFFERENT | Early reset behavior and collision classification can alter episode length and GAE targets. | `env.py:1123-1138`; `trace/pursuit/env.py:102-110` |

## Tests executed

```bash
conda run --no-capture-output -n trace_isaac \
  python -m pytest -q -ra tests/psto tests/psto_parity
```

Result: **30 passed, 1 warning, 14.39 s**. These tests cover pure tensor
operations and formula references; they do not replace the unavailable
official OmniDrones/Isaac Sim 4.1 runtime.

## Audit conclusions

1. TRACE is not a complete official-environment reproduction. The actor,
   centralized-state layout, Beta action transform, heatmap projection, reward
   arithmetic, and ValueNorm are the strongest aligned parts.
2. The 100k collision rate cannot currently be attributed mainly to training
   insufficiency. Terrain geometry, controller/rotor dynamics, LiDAR backend,
   reset semantics, and collision source all differ materially.
3. The saved 20-episode CSV confirms 14 reported pursuer-obstacle collisions,
   but it lacks PhysX contact impulses, per-step LiDAR minima, obstacle scales,
   and reward breakdown. The geometric analysis is therefore evidence, not a
   final contact-causality proof.
4. A paper-scale training run is **NO-GO** until the independent official
   smoke is unblocked or the TRACE-vs-official differences are explicitly
   calibrated and reported.
