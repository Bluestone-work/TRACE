# PSTO long-training readiness

**Decision: NO-GO for a paper-scale long run.**  The training loop is now a real
Isaac Sim 6.0.1 MAPPO/LSTM loop and the 100,000-frame exploratory continuation
completed without numerical failure.  The final 20-episode physical audit
captured 5/20 episodes, but collided in 14/20 episodes.  The policy therefore
has not demonstrated reliable collision-safe pursuit.  These results are
reported as task performance, not hidden behind a smoke-test `pass` status.

The earlier validation was run from Git `aed43a95064b6726e086778c819859ff973163d2`.
The final continuation and evaluation were run from Git
`a34c2ad05888f95db282dffa30cac769ecfdb004`.  The Isaac command used the local
6.0.1 installation and its compatible Torch bundle:

```bash
PYTHONPATH=/home/wj/isaacsim/extsDeprecated/omni.isaac.ml_archive/pip_prebundle:$PWD \
  /home/wj/isaacsim/python.sh <script> ...
```

Putting the standalone `torch_site` wheel first makes Isaac Sim exit during
startup with `libtorch_cuda.so: undefined symbol: ncclCommShrink`; this is why
the command above is part of the reproducibility record.

## P0 results

The unit workflow now pins a CPU-compatible dependency set (`numpy<2`,
`torch==2.5.1`) and runs with `pytest -q -ra --maxfail=1`.  The local equivalent
was executed with the project Isaac environment:

```bash
conda run --no-capture-output -n trace_isaac \
  python -m pytest -q -ra tests/psto
```

Result: **20 passed, 1 warning, 22.66 s**.  No GitHub Actions run was available
from this checkout (`gh` is not installed), so the remote workflow result is
still unverified.

The real MAPPO checkpoint audit used 20 independently seeded randomized
episodes, a 280-step limit, random vehicle resets and random obstacle layouts:

```bash
PYTHONPATH=/home/wj/isaacsim/extsDeprecated/omni.isaac.ml_archive/pip_prebundle:$PWD \
  /home/wj/isaacsim/python.sh scripts/test_psto_episodes.py \
  --episodes 20 --steps 280 --seed 4348 --controller mappo \
  --checkpoint results/psto_p0/final_acceptance/mappo_seed4348_deprecated.pt \
  --headless --out results/psto_p0/final_acceptance/mappo20
```

The run completed 20 physical episodes in Isaac Sim 6.0.1.  The CSV was read
back and collision classes were checked independently:

| metric | result |
| --- | ---: |
| Capture success rate | 0.05 (1/20) |
| Collision rate | 0.95 (19/20) |
| Escape rate | 0.00 (0/20) |
| Timeout rate | 0.00 (0/20) |
| Average episode reward | -892.4406 |
| Average episode length | 143.25 policy steps |
| Target visibility ratio | 0.49194 |
| Collision class | 19 pursuer-obstacle, 0 pursuer-pursuer, 0 pursuer-evader, 0 evader-obstacle |

The complete records are [episodes.json](../results/psto_p0/final_acceptance/mappo20_run1/episodes.json),
[episodes.csv](../results/psto_p0/final_acceptance/mappo20_run1/episodes.csv),
[trajectories.csv](../results/psto_p0/final_acceptance/mappo20_run1/trajectories.csv),
and [run_metadata.json](../results/psto_p0/final_acceptance/mappo20_run1/run_metadata.json).

Scene and environment checkpoint state now include the independent NumPy RNG,
obstacle layout, vehicle state, rotor speeds, histories, previous actions and
episode counters.  The pending post-rollout observation is also stored.  That
last field fixes a subtle resume bug where calling `observe()` after restore
duplicated the current LSTM history sample.

The real PhysX continuous/resume comparison was run with seed 9201:

```bash
PYTHONPATH=/home/wj/isaacsim/extsDeprecated/omni.isaac.ml_archive/pip_prebundle:$PWD \
  /home/wj/isaacsim/python.sh scripts/test_psto_resume_determinism.py \
  --config configs/psto/trace_native_pilot.yaml --seed 9201 \
  --out results/psto_p0/final_acceptance/resume_determinism_tolerance --headless
```

Result: **pass**, 2 frames and 2 updates in both paths, no mismatches, maximum
model tensor absolute difference `5.7975e-08`.  A small `allclose` tolerance is
used because two independent PhysX Kit processes differ by a few floating
point ulps; counters and metadata remain exact.  The result is in
[determinism.json](../results/psto_p0/final_acceptance/resume_determinism_tolerance/determinism.json).

`checkpoint_interval`, `evaluation_interval` and `num_eval_episodes` are read
from the YAML and can be overridden on the command line.  Each checkpoint
stores cumulative frames, update counters, environment state, pending
observation, optimizer/scheduler state, ValueNorm and Python/NumPy/Torch RNG
states.  `trace_native_pilot.yaml` is a bounded single-environment pilot
configuration; it is separate from both the 32-frame smoke configuration and
the upstream 9.8-billion-frame reference configuration.

## P1 10,000-frame pilot

The following was a real PhysX run, with one environment, 64-step rollouts,
one PPO epoch and LSTM supervision on every minibatch:

```bash
PYTHONPATH=/home/wj/isaacsim/extsDeprecated/omni.isaac.ml_archive/pip_prebundle:$PWD \
  /home/wj/isaacsim/python.sh scripts/train_psto.py \
  --config configs/psto/trace_native_pilot.yaml --headless --seed 4348 \
  --total_frames 10000 --rollout_steps 64 --training_epoch_num 1 \
  --num_minibatches 1 --checkpoint_interval 25 --evaluation_interval 50 \
  --num_eval_episodes 1 \
  --checkpoint results/psto_p0/final_acceptance/pilot_10000/checkpoint.pt
```

The run ended with `status=pass`, **10,000 frames**, **157 updates**, 2,015.02
seconds and 4.9627 frames/s.  All recorded values were finite:

| series | minimum | maximum | final |
| --- | ---: | ---: | ---: |
| policy loss | -0.00364 | 0.07657 | 0.00143 |
| value loss | 0.00220 | 5.00319 | 0.00398 |
| LSTM supervised loss | 0.00650 | 0.03557 | 0.01537 |
| encoder gradient norm | 0.533 | 2.496 | 1.444 |
| actor gradient norm | 0.144 | 0.702 | 0.533 |
| critic gradient norm | 0.121 | 19.399 | 0.612 |
| LSTM gradient norm | 0.0179 | 0.0704 | 0.0444 |

Independent evaluations were executed at frames 3200, 6400 and 9600:

| frame | steps | outcome | reward | visibility |
| ---: | ---: | --- | ---: | ---: |
| 3200 | 280 | timeout | -293.95 | 0.5643 |
| 6400 | 253 | escape | -1219.91 | 0.3834 |
| 9600 | 104 | collision | -969.77 | 0.4375 |

The complete training metadata, loss history and evaluation records are in
[checkpoint.json](../results/psto_p0/final_acceptance/pilot_10000/checkpoint.json),
[train.log](../results/psto_p0/final_acceptance/pilot_10000/train.log), and
[config_snapshot.yaml](../results/psto_p0/final_acceptance/pilot_10000/config_snapshot.yaml).
The final checkpoint was reloaded with Isaac Sim's Python 3.12 stack; all
encoder, actor, critic and LSTM tensors were finite and its progress counter
was `frames=10000, updates=157`.

GPU telemetry was sampled every five seconds for 297 samples.  GPU 0 averaged
10.20% utilization (4–40%) and GPU 1 averaged 19.91% (0–100%); the latter was
the active training device during high-utilization intervals.  Raw samples
and aggregates are in [gpu_telemetry.csv](../results/psto_p0/final_acceptance/pilot_10000/gpu_telemetry.csv)
and [gpu_summary.json](../results/psto_p0/final_acceptance/pilot_10000/gpu_summary.json).

After the pilot, the instrumented entry point was exercised once to verify the
new per-update telemetry fields.  That real Isaac Sim run recorded GPU
snapshots and a `checkpoint_seconds` value of **0.01515 s** for its checkpoint;
the record is [checkpoint.json](../results/psto_p0/final_acceptance/checkpoint_timing_smoke/checkpoint.json).
This one-frame run is instrumentation evidence only and is not used as a
training or convergence result.

## Exploratory 100,000-frame continuation

After the pilot, the policy was continued from the 10,000-frame checkpoint in
Isaac Sim 6.0.1. This was a bounded single-environment exploratory run, not a
claim of equivalence to the upstream 1,024-environment training setup:

```bash
PYTHONPATH=/home/wj/isaacsim/extsDeprecated/omni.isaac.ml_archive/pip_prebundle:$PWD \
  /home/wj/isaacsim/python.sh scripts/train_psto.py \
  --config configs/psto/trace_native_long.yaml --headless --seed 4348 \
  --total_frames 100000 --rollout_steps 64 --training_epoch_num 1 \
  --num_minibatches 1 --checkpoint_interval 25 --evaluation_interval 50 \
  --num_eval_episodes 1 --checkpoint results/psto_p0/long_train_100k/checkpoint.pt \
  --resume
```

The continuation ended with `status=pass`, **100,000 cumulative frames**,
**1,564 updates**, **17,807.92 s** elapsed and **5.6155 cumulative
frames/s**. The final checkpoint records 67 genuinely terminated steps and
269 truncated steps. Actor, critic, encoder and LSTM losses and gradient
norms remained finite throughout. The complete records are in
[checkpoint.json](../results/psto_p0/long_train_100k/checkpoint.json),
[train.log](../results/psto_p0/long_train_100k/train.log),
[config_snapshot.yaml](../results/psto_p0/long_train_100k/config_snapshot.yaml),
and [run_manifest.json](../results/psto_p0/long_train_100k/run_manifest.json).

The final checkpoint was evaluated with 20 independently seeded randomized
PhysX episodes (280 policy steps maximum):

```bash
PYTHONPATH=/home/wj/isaacsim/extsDeprecated/omni.isaac.ml_archive/pip_prebundle:$PWD \
  /home/wj/isaacsim/python.sh scripts/test_psto_episodes.py \
  --episodes 20 --steps 280 --seed 50000 --controller mappo \\
  --checkpoint results/psto_p0/long_train_100k/checkpoint.pt \
  --config configs/psto/trace_native_long.yaml --headless \
  --out results/psto_p0/long_train_100k/eval20
```

All 20 episodes completed with no script failures:

| metric | result |
| --- | ---: |
| Capture success rate | 0.25 (5/20) |
| Collision rate | 0.70 (14/20) |
| Escape rate | 0.00 (0/20) |
| Timeout flag | 0.10 (2/20; one coincides with a terminal collision) |
| Genuine truncation | 0.05 (1/20) |
| Average episode reward | -233.3886 |
| Average episode length | 149.95 policy steps |
| Target visibility ratio | 0.54005 |

The machine-readable result is [episodes.json](../results/psto_p0/long_train_100k/eval20/episodes.json),
with raw trajectories in [trajectories.csv](../results/psto_p0/long_train_100k/eval20/trajectories.csv)
and episode rows in [episodes.csv](../results/psto_p0/long_train_100k/eval20/episodes.csv).
The CSV also records `terminated` and `truncated` separately; the genuine
timeout count above uses `truncated`, so the collision at the 280-step boundary
is not counted as a timeout termination.

## Remaining blockers

The code and physical training path are ready for controlled experiments, but
the acceptance gate for a long research run is not met:

1. The final 100,000-frame continuation still has a 70% collision rate and
   only 25% capture success across 20 randomized episodes.
2. The 10,000-frame pilot had only one evaluation episode per checkpoint and
   showed timeout, escape and collision outcomes without a capture; the longer
   continuation improved the aggregate result but did not establish robustness.
3. The adapter remains one physical environment (`num_envs=1`), so these
   numbers cannot be described as an upstream 1024-environment reproduction.
4. The GitHub Actions workflow was fixed and the local 20-test suite passes,
   but a remote Actions run still needs to be triggered and observed.

The next gate should be a collision-safe curriculum or reward/scene diagnosis,
followed by a fresh multi-seed evaluation.  Starting a paper-scale long run
before that would turn the current failure modes into training time rather
than evidence of convergence.
