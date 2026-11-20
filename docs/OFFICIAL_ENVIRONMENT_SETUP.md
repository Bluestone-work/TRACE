# Official PSTO environment setup

The official source is kept separately at `upstream/psto-aav-pursuit-official`.
It is pinned to HITSZ-MAS commit
`764e2a156f4f74fc3d24a25dfaa6871c57ec7944` and is not modified by TRACE.

The official README and `ENVIRONMENT.md` require:

- Linux and Python 3.10.
- Isaac Sim 4.1.0.
- OmniDrones installed separately in editable mode.
- `torchrl==0.3.1` and its compatible `tensordict` dependency.

The current machine has Isaac Sim 6.0.1 with Python 3.12.13. The following
probe was run without substituting the TRACE adapter:

```bash
PYTHONPATH=/home/wj/isaacsim/extsDeprecated/omni.isaac.ml_archive/pip_prebundle:$PWD \
  /home/wj/isaacsim/python.sh scripts/test_official_psto_smoke.py \
  --out results/psto_official_smoke/official_smoke.json
```

Observed result: **BLOCKED**. `omni_drones`, `torchrl`, `tensordict`, and the
Isaac Lab 4.x module are unavailable in the Isaac Sim 6.0.1 runtime. The
machine-readable probe result is
`results/psto_official_smoke/official_smoke.json`.

The independent environment must be installed before the official smoke can
be claimed. After installation, run the probe again and then use the official
`train.py` with a bounded Hydra override. Do not point the official script at
the TRACE Isaac Sim 6.0.1 adapter and do not run the 9.8B-frame configuration.
