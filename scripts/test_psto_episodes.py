#!/usr/bin/env python3
"""Run complete multi-episode PSTO rollouts in real Isaac Sim PhysX."""
import argparse
import csv
import json
import hashlib
import subprocess
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


def git_sha():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def file_sha256(path):
    if not path:
        return None
    p = Path(path)
    if not p.is_file():
        return None
    digest = hashlib.sha256()
    with p.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--steps", type=int, default=280)
    ap.add_argument("--seed", type=int, default=4348)
    ap.add_argument("--out", default="results/psto_episodes")
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--controller", choices=("fixed", "mappo"), default="fixed")
    ap.add_argument("--checkpoint", help="MAPPO checkpoint; required with --controller mappo")
    ap.add_argument("--config", default="configs/psto/trace_native_pilot.yaml")
    args = ap.parse_args()
    out = unique_dir(args.out)
    config_path = Path(args.config)
    if config_path.is_file():
        (out / "config_snapshot.yaml").write_text(config_path.read_text(encoding="utf-8"), encoding="utf-8")
    from isaacsim import SimulationApp
    app = SimulationApp({"headless": bool(args.headless), "multi_gpu": False})
    env = None
    episode_rows, trajectory_rows, failures = [], [], []
    try:
        import torch
        from trace.pursuit.env import PSTOPursuitEnv
        env = PSTOPursuitEnv(max_episode_steps=args.steps, seed=args.seed,
            randomize_resets=True, randomize_obstacles=True)
        policy = None
        if args.controller == "mappo":
            if not args.checkpoint:
                raise ValueError("--checkpoint is required with --controller mappo")
            from trace.pursuit.algorithm.mappo import PSTOMAPPO
            policy = PSTOMAPPO(device="cuda" if torch.cuda.is_available() else "cpu")
            policy.load_model_weights(args.checkpoint, load_optimizers=False); policy.eval(); env.set_policy_lstm(policy.lstm)
        # Verify the seed contract before collecting episodes: the same seed must
        # reconstruct reset poses, while a different seed must change them.
        env.reset(seed=args.seed)
        same_a = np.stack([v.position for v in env.scene.vehicles]).copy()
        env.reset(seed=args.seed)
        same_b = np.stack([v.position for v in env.scene.vehicles]).copy()
        layout_a = np.asarray(env.scene.obstacle_positions, np.float32).copy()
        env.reset(seed=args.seed)
        layout_b = np.asarray(env.scene.obstacle_positions, np.float32).copy()
        env.reset(seed=args.seed + 1)
        different = np.stack([v.position for v in env.scene.vehicles]).copy()
        layout_different = np.asarray(env.scene.obstacle_positions, np.float32).copy()
        if not np.allclose(same_a, same_b): failures.append("same_seed_reset_not_reproducible")
        if not np.allclose(layout_a, layout_b): failures.append("same_seed_obstacle_layout_not_reproducible")
        if np.allclose(same_a, different): failures.append("different_seed_reset_not_randomized")
        if np.allclose(layout_a, layout_different): failures.append("different_seed_obstacle_layout_not_randomized")
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
            episode_layout = np.asarray(env.scene.obstacle_positions, np.float32).tolist()
            for step in range(args.steps):
                if not np.isfinite(obs["state"]).all() or not np.isfinite(obs["central_state"]).all():
                    failures.append(f"nonfinite_observation_episode_{episode}_step_{step}")
                if policy is None:
                    actions = np.full((2, 2), .5, dtype=np.float32)
                else:
                    with torch.no_grad():
                        action, _, _ = policy.act(torch.as_tensor(obs["fused_image"], device=policy.device),
                            torch.as_tensor(obs["state"], device=policy.device),
                            torch.as_tensor(obs["central_state"], device=policy.device), deterministic=True)
                    actions = action.cpu().numpy()
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
                "reacquisition_time_steps": float(np.mean(reacquisition_steps)) if reacquisition_steps else None,
                "pursuer_pursuer_collision": bool(terminal_info.get("pursuer_pursuer_collision", False)),
                "pursuer_evader_collision": bool(terminal_info.get("pursuer_evader_collision", False)),
                "pursuer_obstacle_collision": bool(terminal_info.get("pursuer_obstacle_collision", False)),
                "evader_obstacle_collision": bool(terminal_info.get("evader_obstacle_collision", False)),
                "obstacle_layout": json.dumps(episode_layout)})
        execution_status = "pass" if not failures and len(episode_rows) == args.episodes else "fail"
    except Exception as exc:
        import traceback
        traceback.print_exc()
        failures.append(f"{type(exc).__name__}: {exc}")
        execution_status = "blocked"
    with (out / "episodes.csv").open("w", newline="") as handle:
        fields = episode_rows[0].keys() if episode_rows else ["episode"]
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(episode_rows)
    with (out / "trajectories.csv").open("w", newline="") as handle:
        fields = trajectory_rows[0].keys() if trajectory_rows else ["episode"]
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(trajectory_rows)
    result = {"status": execution_status, "git_sha": git_sha(), "config": str(config_path.resolve()),
        "test_scope": "MAPPO policy rollout" if args.controller == "mappo" else "fixed-action environment rollout",
        "task_performance_status": "evaluated" if args.controller == "mappo" else "not_evaluated",
        "controller": args.controller, "checkpoint": args.checkpoint, "checkpoint_sha256": file_sha256(args.checkpoint),
        "episodes_requested": args.episodes, "episodes_completed": len(episode_rows),
        "steps_per_episode_limit": args.steps, "policy_hz": 30., "episode_duration_s": args.steps / 30.,
        "failures": failures, "episodes_csv": str(out / "episodes.csv"),
        "trajectories_csv": str(out / "trajectories.csv"),
        "capture_count": int(sum(r["capture"] for r in episode_rows)),
        "escape_count": int(sum(r["escaped"] for r in episode_rows)),
        "collision_count": int(sum(r["collision"] for r in episode_rows)),
        "timeout_count": int(sum(r["timeout"] for r in episode_rows)),
        "mean_steps": float(np.mean([r["steps"] for r in episode_rows])) if episode_rows else 0.,
        "capture_success_rate": float(np.mean([r["capture"] for r in episode_rows])) if episode_rows else 0.,
        "collision_rate": float(np.mean([r["collision"] for r in episode_rows])) if episode_rows else 0.,
        "escape_rate": float(np.mean([r["escaped"] for r in episode_rows])) if episode_rows else 0.,
        "average_episode_reward": float(np.mean([r["reward"] for r in episode_rows])) if episode_rows else 0.,
        "average_episode_length": float(np.mean([r["steps"] for r in episode_rows])) if episode_rows else 0.,
        "target_visibility_ratio": float(np.mean([r["visibility_ratio"] for r in episode_rows])) if episode_rows else 0.}
    (out / "episodes.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)
    if env is not None:
        env.close()
    app.close()
    if execution_status != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
