# PSTO stage 3 P0 report

Baseline: `feaadef50e94fc9debbfcec63f07c99b9e1de978`.

## P0-1 centralized state

`trace/pursuit/env.py::build_global_state_84` now follows upstream `global_state_full`:

```text
all_drone_positions_local_flat  3*3 = 9
all_drone_other_states          3*(quat4 + linear_vel3 + angular_vel3) = 30
all_prev_actions                3*3 = 9
_get_1d_scan_from_2d_image      2*18 = 36
                                             84
```

The one-world TRACE adapter has no environment offset, so world and local positions are
identical. Previous actions are body-frame 3-D velocity targets. A shape assertion rejects
anything except 84 dimensions.

## P0-2 action loop

The environment accepts normalized actor actions with shape `[2, 2]`. It calls
`PSTOMAPPO.action_to_velocity`, which performs the official Beta direction/magnitude
mapping, body planar velocity construction, and yaw-only world transform. The resulting
world velocity goes through `RotorFlight._apply_rotor_control` and `QuadrotorActuator`;
no motor command or pose write is made during a step. `prev_actions` stores the body-frame
3-D targets for all three vehicles.

## P0-3 visibility and leakage

`visibility.py` implements max-distance, beam alignment and target-distance clearance.
Evader current/predicted heatmap signs are multiplied by each pursuer's independent mask;
teammate current/future heatmap entries remain present. The actor receives only observed
history and the predicted trajectory. Ground-truth history is retained separately as an
auxiliary label buffer. Teammate future points use `p + v*t`, as in upstream.

## P0-4 rewards

`rewards.py` ports the upstream reward terms: relative velocity pursuit, pursuer pair
coordination, LiDAR range-minus-hit-distance logarithmic quantile penalty, capture bonus,
formation angle term, collision terms, altitude bound and time penalty. Escape uses the
upstream OR condition (`min distance > escape_min OR max distance > escape_max`).

## Executed checks

Pure Torch/NumPy checks:

```bash
PYTHONPATH=$PWD conda run --no-capture-output -n trace_isaac pytest -q tests/psto
```

Result: `12 passed`.

Real Isaac Sim 6.0.1 integration:

```bash
PYTHONPATH=/home/wj/isaacsim/extsDeprecated/omni.isaac.ml_archive/pip_prebundle:$PWD \
PYTHONEXE=/home/wj/isaacsim/kit/python/bin/python3 \
/home/wj/isaacsim/python.sh scripts/test_psto_integration.py --steps 20 --headless
```

Result: 20/20 policy steps passed with centralized state 84-D, actor action 2-D, physical
rotor stepping, fresh LiDAR/heatmap observations, finite values and no future label passed
to the actor. Report: `results/psto_integration/integration.json`.

Duration checks:

```bash
PYTHONPATH=/home/wj/isaacsim/extsDeprecated/omni.isaac.ml_archive/pip_prebundle:$PWD \
PYTHONEXE=/home/wj/isaacsim/kit/python/bin/python3 \
/home/wj/isaacsim/python.sh scripts/test_psto_long.py --headless
```

Result: 10 s hover and 30 s independent control passed. Drone 0 displacement was
14.776 m, stationary drone displacement was `2.3e-5 m`, and peak speed was `0.505 m/s`.
Report: `results/psto_long/long.json`.

The physical scene report now records screenshot failure/black-headless status separately;
`results/psto_scene/psto_scene.json` reports `screenshot_status` rather than claiming that
visual rendering succeeded.

## Remaining scope

P0 is complete. Formal MAPPO long training, independent episode evaluation and MP4
recording remain deliberately unstarted.
