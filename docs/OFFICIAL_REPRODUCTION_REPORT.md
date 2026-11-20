# Official PSTO reproduction report

## Scope

This is an environment-availability report for the pinned HITSZ-MAS source,
not a claim that the official Isaac Sim run succeeded. The source commit is
`764e2a156f4f74fc3d24a25dfaa6871c57ec7944`.

## Required smoke sequence

Once the documented Python 3.10 / Isaac Sim 4.1.0 / OmniDrones environment is
installed, the bounded smoke sequence is:

1. Import `env.py` and construct one `NavigationEnv`.
2. Reset one environment with the fixed seed.
3. Confirm three Hummingbirds, 24x480 LiDAR, 6x120 downsampled input, and an
   84-dimensional centralized state.
4. Run one actor forward pass and one Beta action conversion.
5. Advance ten policy steps and record reward, termination, and finite state.
6. Run one MAPPO update and save a smoke checkpoint.

The requested runtime is represented by
`configs/psto/official_smoke.yaml`. It is intentionally independent from the
Isaac Sim 6 TRACE configurations.

## Current execution evidence

The probe command was executed on the current host:

```bash
PYTHONPATH=/home/wj/isaacsim/extsDeprecated/omni.isaac.ml_archive/pip_prebundle:$PWD \
  /home/wj/isaacsim/python.sh scripts/test_official_psto_smoke.py \
  --out results/psto_official_smoke/official_smoke.json
```

It returned **BLOCKED** because the Isaac Sim 6.0.1 Python 3.12 runtime does
not contain OmniDrones, TorchRL, TensorDict, or Isaac Lab 4.x. No official
environment reset, LiDAR rollout, actor pass, or MAPPO update was claimed.

## Consequence

The official-vs-TRACE numerical tests that do not require OmniDrones can run
locally. Runtime-dependent parity remains **UNKNOWN** until the independent
official environment is installed. This is an explicit dependency blocker,
not a substituted mock result.
