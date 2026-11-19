#!/usr/bin/env python3
"""Longer real-PhysX hover and independent-command acceptance checks."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hover-seconds", type=float, default=10.)
    ap.add_argument("--independent-seconds", type=float, default=30.)
    ap.add_argument("--out", default="results/psto_p0/physics_long")
    ap.add_argument("--headless", action="store_true")
    args = ap.parse_args(); out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    from isaacsim import SimulationApp
    app = SimulationApp({"headless": bool(args.headless), "multi_gpu": False})
    scene = None; rows = []; failures = []
    try:
        from trace.pursuit.scene import PSTOPursuitScene
        scene = PSTOPursuitScene(); scene.reset()
        policy_dt = scene.policy_dt
        phases = [("hover", int(round(args.hover_seconds / policy_dt))),
                  ("independent", int(round(args.independent_seconds / policy_dt)))]
        for phase, steps in phases:
            scene.reset()
            initial = np.stack([v.position for v in scene.vehicles]).copy()
            for step in range(steps):
                if phase == "hover":
                    command = np.zeros((3, 3), np.float32)
                else:
                    command = np.array([[.45, 0., 0.], [-.35, 0., 0.], [0., .25, 0.]], np.float32)
                scene.step(command)
                pos = np.stack([v.position for v in scene.vehicles]); vel = np.stack([v.velocity for v in scene.vehicles])
                quat = np.stack([v.orientation for v in scene.vehicles]); ang = np.stack([v.angular_velocity for v in scene.vehicles])
                finite = np.isfinite(np.r_[pos.ravel(), vel.ravel(), quat.ravel(), ang.ravel()]).all()
                if not finite: failures.append(f"nonfinite_{phase}_{step}")
                if step % max(int(round(1. / policy_dt)), 1) == 0 or step == steps - 1:
                    rows.append({"phase": phase, "step": step, "sim_time_s": step * policy_dt,
                        "max_altitude_error": float(np.max(np.abs(pos[:, 2] - 2.))),
                        "max_speed": float(np.max(np.linalg.norm(vel, axis=1))),
                        "max_position_change": float(np.max(np.linalg.norm(pos - initial, axis=1))),
                        "finite": bool(finite)})
        status = "pass" if not failures else "fail"
    except Exception as exc:
        import traceback; traceback.print_exc(); failures.append(f"{type(exc).__name__}: {exc}"); status = "blocked"
    with (out / "physics.csv").open("w", newline="") as handle:
        fields = rows[0].keys() if rows else ["phase"]; writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    result = {"status": status, "physics_dt": 1/240, "policy_dt": 1/30, "hover_seconds": args.hover_seconds,
        "independent_seconds": args.independent_seconds, "failures": failures, "csv": str(out / "physics.csv")}
    (out / "physics.json").write_text(json.dumps(result, indent=2)); print(json.dumps(result, indent=2), flush=True)
    if scene is not None: scene.close()
    app.close()
    if status != "pass": raise SystemExit(1)


if __name__ == "__main__": main()
