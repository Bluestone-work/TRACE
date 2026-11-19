#!/usr/bin/env python3
"""Measure the unmodified PSTO LiDAR and environment-step cost in Isaac Sim 6."""
import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np


def unique_dir(path):
    path = Path(path)
    if not path.exists():
        path.mkdir(parents=True)
        return path
    i = 1
    while path.with_name(f"{path.name}_run{i}").exists():
        i += 1
    out = path.with_name(f"{path.name}_run{i}")
    out.mkdir(parents=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--out", default="results/psto_lidar_profile")
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--seed", type=int, default=4348)
    args = ap.parse_args()
    out = unique_dir(args.out)
    from isaacsim import SimulationApp
    app = SimulationApp({"headless": bool(args.headless), "multi_gpu": False})
    env = None
    rows, failures = [], []
    try:
        from trace.pursuit.env import PSTOPursuitEnv
        env = PSTOPursuitEnv(seed=args.seed)
        env.reset()
        for step in range(args.steps):
            scan_times = []
            for drone in env.scene.vehicles:
                # Force a fresh sensor sample for the benchmark; normal operation
                # intentionally reuses the cache within one PhysX state.
                drone._scene_lidar_epoch += 1
                drone._lidar_cache_epoch = -1
                start = time.perf_counter()
                scan = drone._raycast_lidar()
                elapsed = time.perf_counter() - start
                scan_times.append(elapsed)
                if scan.shape != (24, 480) or not np.isfinite(scan).all():
                    failures.append(f"invalid_scan_{step}_{len(scan_times)-1}")
            start = time.perf_counter()
            for drone in env.scene.vehicles:
                drone._scene_lidar_epoch += 1
                drone._lidar_cache_epoch = -1
            env.observe()
            observe_time = time.perf_counter() - start
            start = time.perf_counter()
            env.step(np.full((2, 2), .5, dtype=np.float32), actions_are_normalized=True)
            step_time = time.perf_counter() - start
            rows.append({"step": step, "pursuer0_scan_s": scan_times[0],
                "pursuer1_scan_s": scan_times[1], "evader_scan_s": scan_times[2],
                "scan_total_s": sum(scan_times), "observe_s": observe_time,
                "env_step_s": step_time, "env_step_fps": 1. / max(step_time, 1e-12)})
        status = "pass" if not failures else "fail"
    except Exception as exc:
        import traceback
        traceback.print_exc()
        failures.append(f"{type(exc).__name__}: {exc}")
        status = "blocked"
    result = {"status": status, "steps": len(rows), "requested_steps": args.steps,
        "seed": args.seed, "raw_lidar_shape": [24, 480], "downsampled_shape": [6, 120],
        "failures": failures, "semantic_note": "raw proximity = range - hit_distance; zero means no hit",
        "csv": str(out / "lidar_profile.csv")}
    with (out / "lidar_profile.csv").open("w", newline="") as handle:
        fields = rows[0].keys() if rows else ["step"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    if rows:
        result["mean_scan_s"] = float(np.mean([r["scan_total_s"] for r in rows]))
        result["mean_observe_s"] = float(np.mean([r["observe_s"] for r in rows]))
        result["mean_env_step_s"] = float(np.mean([r["env_step_s"] for r in rows]))
        result["mean_env_step_fps"] = float(1. / max(result["mean_env_step_s"], 1e-12))
    (out / "lidar_profile.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)
    if env is not None:
        env.close()
    app.close()
    if status != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
