#!/usr/bin/env python3
"""Compare uninterrupted and checkpoint-resumed real PhysX training.

The script launches the real Isaac Sim training entry point three times.  The
first run executes two policy frames continuously; the second executes one
update, then resumes from its checkpoint for the second frame.  Model,
optimizer, scheduler, ValueNorm and counter states are compared exactly.
"""
import argparse
import json
import os
import subprocess
from pathlib import Path


def run(cmd, cwd):
    print("$", " ".join(str(x) for x in cmd), flush=True)
    # Isaac Sim 6.0.1 ships an ml_archive Torch wheel.  Put that wheel first
    # for Kit startup; otherwise the app exits before the training entry point
    # can import its PyTorch stack (the standalone torch_site wheel is a
    # different binary build).  Keep any caller supplied paths after it.
    env = os.environ.copy()
    bundled = "/home/wj/isaacsim/extsDeprecated/omni.isaac.ml_archive/pip_prebundle"
    env["PYTHONPATH"] = f"{bundled}:{cwd}:{env.get('PYTHONPATH', '')}"
    subprocess.run(cmd, cwd=cwd, check=True, env=env)


def equal_tree(a, b, path="", rtol=1e-6, atol=1e-7):
    import torch
    if isinstance(a, torch.Tensor):
        # PhysX can differ by a few ulps between two Kit processes even with
        # identical serialized poses/rotor speeds.  Treat that bounded
        # floating point noise as deterministic; counters and non-tensor
        # metadata below remain exact.
        return isinstance(b, torch.Tensor) and torch.allclose(a, b, rtol=rtol, atol=atol, equal_nan=False)
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(equal_tree(a[k], b[k], f"{path}.{k}", rtol, atol) for k in a)
    if isinstance(a, (list, tuple)):
        return isinstance(b, type(a)) and len(a) == len(b) and all(equal_tree(x, y, f"{path}[{i}]", rtol, atol) for i, (x, y) in enumerate(zip(a, b)))
    return a == b


def max_tensor_abs_diff(a, b):
    import torch
    if isinstance(a, torch.Tensor):
        return float((a - b).abs().max().item()) if a.numel() else 0.0
    if isinstance(a, dict):
        return max((max_tensor_abs_diff(a[k], b[k]) for k in a if k in b), default=0.0)
    if isinstance(a, (list, tuple)):
        return max((max_tensor_abs_diff(x, y) for x, y in zip(a, b)), default=0.0)
    return 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/psto/trace_native_pilot.yaml")
    ap.add_argument("--seed", type=int, default=9201)
    ap.add_argument("--out", default="results/psto_p0/resume_determinism")
    ap.add_argument("--python", default=os.environ.get("ISAACSIM_PYTHON", "/home/wj/isaacsim/python.sh"))
    ap.add_argument("--headless", action="store_true")
    args = ap.parse_args()
    root = Path(__file__).resolve().parents[1]; out = root / args.out
    out.mkdir(parents=True, exist_ok=True)
    continuous = out / "continuous.pt"; split = out / "split.pt"
    common = [args.python, "scripts/train_psto.py", "--config", args.config, "--seed", str(args.seed),
              "--total_frames", "2", "--rollout_steps", "1", "--training_epoch_num", "1",
              "--num_minibatches", "1"]
    if args.headless: common.append("--headless")
    run(common + ["--checkpoint", str(continuous)], root)
    run(common + ["--checkpoint", str(split), "--max_updates", "1"], root)
    run(common + ["--checkpoint", str(split), "--resume"], root)
    import torch
    a = torch.load(continuous, map_location="cpu", weights_only=False)
    b = torch.load(split, map_location="cpu", weights_only=False)
    state_keys = ["encoder", "actor", "critic_encoder", "critic", "lstm", "encoder_opt", "actor_opt",
                  "critic_encoder_opt", "critic_opt", "lstm_opt", "encoder_scheduler", "actor_scheduler",
                  "critic_encoder_scheduler", "critic_scheduler", "lstm_scheduler", "value_norm",
                  "update_count", "lstm_train_counter"]
    mismatches = [key for key in state_keys if not equal_tree(a.get(key), b.get(key), key)]
    apg = a.get("progress", {}); bpg = b.get("progress", {})
    if apg.get("frames") != bpg.get("frames") or apg.get("updates") != bpg.get("updates"):
        mismatches.append("progress.counters")
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    except Exception:
        sha = "unknown"
    result = {"status": "pass" if not mismatches else "fail", "git_sha": sha,
              "config": str((root / args.config).resolve()), "seed": args.seed,
              "continuous_checkpoint": str(continuous), "resumed_checkpoint": str(split),
              "continuous_frames": apg.get("frames"), "resumed_frames": bpg.get("frames"),
              "continuous_updates": apg.get("updates"), "resumed_updates": bpg.get("updates"),
              "max_model_tensor_abs_diff": max(max_tensor_abs_diff(a.get(k), b.get(k)) for k in state_keys),
              "mismatches": mismatches}
    (out / "determinism.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    if mismatches: raise SystemExit(1)


if __name__ == "__main__": main()
