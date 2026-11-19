#!/usr/bin/env python3
"""Run complete multi-episode PSTO rollouts in real Isaac Sim PhysX."""
import argparse
import csv
import json
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
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--steps", type=int, default=280)
    ap.add_argument("--seed", type=int, default=4348)
    ap.add_argument("--out", default="results/psto_episodes")
    ap.add_argument("--headless", action="store_true")
    args = ap.parse_args()
    out = unique_dir(args.out)
    from isaacsim import SimulationApp
    app = SimulationApp({"headless": bool(args.headless), "multi_gpu": False})
    env = None
    episode_rows, trajectory_rows, failures = [], [], []
    try:
        from trace.pursuit.env import PSTOPursuitEnv
        env = PSTOPursuitEnv(max_episode_steps=args.steps, seed=args.seed,
            randomize_resets=True, randomize_obstacles=True)
        for episode in range(args.episodes):
            episode_seed = args.seed + episode
            obs = env.reset(seed=episode_seed)
            reward_sum = 0.
            visibility = []
            reacquisition_steps = []
            missing_start = None
            terminal_info = {"capture": False, "escaped": False, "collision": False,
                "timeout": False, "terminated": False, "truncated": False}
            completed = 0
            for step in range(args.steps):
                if not np.isfinite(obs["state"]).all() or not np.isfinite(obs["central_state"]).all():
                    failures.append(f"nonfinite_observation_episode_{episode}_step_{step}")
                actions = np.full((2, 2), .5, dtype=np.float32)
                next_obs, reward, done, info = env.step(actions, actions_are_normalized=True)
                if not np.isfinite(reward):
                    failures.append(f"nonfinite_reward_episode_{episode}_step_{step}")
                visibility.append(float(info["visibility_ratio"]))
                if info["visibility_ratio"] < 1e-6 and missing_start is None:
                    missing_start = step
                if info["visibility_ratio"] > 1e-6 and missing_start is not None:
                    reacquisition_steps.append(step - missing_start)
                    missing_start = None
                pos, quat, vel, ang = env._root()
                for drone in range(3):
                    trajectory_rows.append({"episode": episode, "seed": episode_seed, "step": step,
                        "drone": drone, "x": float(pos[drone, 0]), "y": float(pos[drone, 1]),
                        "z": float(pos[drone, 2]), "vx": float(vel[drone, 0]),
                        "vy": float(vel[drone, 1]), "vz": float(vel[drone, 2]),
                        "qw": float(quat[drone, 0]), "qx": float(quat[drone, 1]),
                        "qy": float(quat[drone, 2]), "qz": float(quat[drone, 3]),
                        "wx": float(ang[drone, 0]), "wy": float(ang[drone, 1]), "wz": float(ang[drone, 2])})
                reward_sum += float(reward); completed += 1; terminal_info = info
                obs = next_obs
                if done:
                    break
            episode_rows.append({"episode": episode, "seed": episode_seed, "steps": completed,
                "reward": reward_sum, "capture": bool(terminal_info["capture"]),
                "escaped": bool(terminal_info["escaped"]), "collision": bool(terminal_info["collision"]),
                "timeout": bool(terminal_info["timeout"]), "terminated": bool(terminal_info["terminated"]),
                "truncated": bool(terminal_info["truncated"]),
                "visibility_ratio": float(np.mean(visibility)) if visibility else 0.,
                "reacquisition_time_steps": float(np.mean(reacquisition_steps)) if reacquisition_steps else None})
        status = "pass" if not failures and len(episode_rows) == args.episodes else "fail"
    except Exception as exc:
        import traceback
        traceback.print_exc()
        failures.append(f"{type(exc).__name__}: {exc}")
        status = "blocked"
    with (out / "episodes.csv").open("w", newline="") as handle:
        fields = episode_rows[0].keys() if episode_rows else ["episode"]
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(episode_rows)
    with (out / "trajectories.csv").open("w", newline="") as handle:
        fields = trajectory_rows[0].keys() if trajectory_rows else ["episode"]
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(trajectory_rows)
    result = {"status": status, "episodes_requested": args.episodes, "episodes_completed": len(episode_rows),
        "steps_per_episode_limit": args.steps, "policy_hz": 30., "episode_duration_s": args.steps / 30.,
        "failures": failures, "episodes_csv": str(out / "episodes.csv"),
        "trajectories_csv": str(out / "trajectories.csv"),
        "capture_count": int(sum(r["capture"] for r in episode_rows)),
        "escape_count": int(sum(r["escaped"] for r in episode_rows)),
        "collision_count": int(sum(r["collision"] for r in episode_rows)),
        "timeout_count": int(sum(r["timeout"] for r in episode_rows)),
        "mean_steps": float(np.mean([r["steps"] for r in episode_rows])) if episode_rows else 0.}
    (out / "episodes.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)
    if env is not None:
        env.close()
    app.close()
    if status != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
