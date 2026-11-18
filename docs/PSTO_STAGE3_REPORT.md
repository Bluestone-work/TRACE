# PSTO Stage 3 verification

## Implemented fixes

- `trace/pursuit/scene.py` now uses the upstream initial arrangement `(-3, 2, 2)`, `(-3, -2, 2)`, `(0, 0, 2)` and creates a ground plane plus deterministic fixed cuboid obstacles. The three Hummingbirds remain separate articulations in one 240 Hz PhysX world; commands are applied every eight physics ticks (30 Hz).
- `trace/pursuit/env.py` keeps the actor's previous action as body-frame velocity and builds the official 84-D centralized state. The actor receives only current observations; delayed LSTM history/target data are exposed under separate keys.
- `trace/pursuit/algorithm/mappo.py` now runs configured PPO epochs and shuffled minibatches, with GAE, value normalization/clipping, independent optimizers, schedulers, supervised LSTM updates, and scheduler/optimizer checkpoint state.
- `scripts/train_psto.py` is a real physical rollout/update/checkpoint entry point (`num_envs=1` currently). `scripts/eval_psto.py` performs gradient-free physical evaluation and writes the requested metrics.
- Rollout collection now evaluates the critic on every nonterminal next observation and passes the resulting per-step `next_value` to GAE; terminal transitions receive zero bootstrap. Both train and eval connect `policy.lstm` through `env.set_policy_lstm()` before reset. The evader now converts its own physical LiDAR hits to world points and passes them to the official obstacle-repulsion policy.

## Commands and observed results

All commands below were run with Isaac Sim 6.0.1 through `/home/wj/isaacsim/python.sh`.

```bash
PYTHONPATH=/home/wj/isaacsim/extsDeprecated/omni.isaac.ml_archive/pip_prebundle:$PWD \
PYTHONEXE=/home/wj/isaacsim/kit/python/bin/python3 \
/home/wj/isaacsim/python.sh scripts/test_psto_integration.py --steps 20 --headless
```

Result: `pass`, 20/20 contiguous policy steps, `done=False` for every row, centralized state 84-D, actor action 2-D, no future label passed to actor. The CSV distance changes continuously from 3.60555 m to 3.89591 m.

```bash
PYTHONPATH=/home/wj/isaacsim/extsDeprecated/omni.isaac.ml_archive/pip_prebundle:$PWD \
PYTHONEXE=/home/wj/isaacsim/kit/python/bin/python3 \
/home/wj/isaacsim/python.sh scripts/test_psto_long.py --headless
```

Result: `pass`; 10 s hover and 30 s independent control, 240 Hz physics, 30 Hz policy, 8 substeps, 12.6700 m commanded position response, 0 m stationary-drone displacement, 1.5598 m/s peak speed, no non-finite or altitude failures. LiDAR scans were finite and obstacles were present in the scene.

```bash
PYTHONPATH=/home/wj/isaacsim/extsDeprecated/omni.isaac.ml_archive/pip_prebundle:$PWD \
PYTHONEXE=/home/wj/isaacsim/kit/python/bin/python3 \
/home/wj/isaacsim/python.sh scripts/train_psto.py --headless --total_frames 4 --rollout_steps 4 --training_epoch_num 1 --num_minibatches 1 --checkpoint results/psto_train/final_smoke.pt
```

Result: `pass`; one real rollout/update, finite policy/value/LSTM losses and nonzero gradient norms, checkpoint written. A subsequent `--resume` run loaded the checkpoint and performed another real update. One-transition smoke runs are finite after using population standard deviation for advantage normalization.

After the next-value/LSTM/evader fixes, a 20-frame real Isaac Sim rollout completed with one PPO update (`policy_loss=0.0255`, `value_loss=2.7385`, `sl_loss=0.0122`, finite gradients). The nonzero supervised loss confirms that delayed LSTM labels are being used while the environment prediction path is active.

The physical evaluation entry point was launched against this checkpoint with one episode and two steps. It returned `pass`, visibility ratio `0.5`, episode reward `0.6681`, and no capture/escape/collision; the result is in `results/psto_eval/smoke.json`. This is an entry-point smoke evaluation, not a trained-policy claim.

## Remaining limitations

- The fixed cuboid obstacle field is an Isaac Sim 6 adapter, not a byte-for-byte OmniDrones `TerrainImporter` reproduction. It supplies real collision geometry and LiDAR occlusion, but terrain-generation parity remains future work.
- The physical adapter is single-world (`--num_envs 1`); vectorized multi-environment training is not claimed.
- Long-horizon trained-policy metrics are not reported here because only a short parameter-update smoke run was requested. No fabricated success rate is included.
