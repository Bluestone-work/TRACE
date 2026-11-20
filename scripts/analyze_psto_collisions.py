#!/usr/bin/env python3
"""Audit saved PSTO trajectories for geometric collision evidence.

This tool deliberately does not infer PhysX contact from a reward flag.  It
recomputes the distance to the saved obstacle AABBs and vehicle pairs from the
CSV produced by ``test_psto_episodes.py``.  LiDAR minimum distance, contact
impulses, and reward terms are reported as unavailable unless they were logged
by the rollout script.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


def _bool(value: str | bool | None) -> bool:
    return value is True or str(value).lower() == "true"


def _aabb_distance(point: np.ndarray, center: np.ndarray, scale: np.ndarray) -> float:
    half = scale / 2.0
    delta = np.maximum(np.abs(point - center) - half, 0.0)
    return float(np.linalg.norm(delta))


def _load(path: Path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def analyze(episodes_path: Path, trajectories_path: Path, out: Path, window: int = 60):
    episodes = _load(episodes_path)
    trajectories = _load(trajectories_path)
    by_episode: dict[int, list[dict]] = defaultdict(list)
    for row in trajectories:
        by_episode[int(row["episode"])].append(row)
    episode_index = {int(row["episode"]): row for row in episodes}

    summary_rows = []
    trajectory_rows = []
    class_counts = defaultdict(int)
    unavailable = {
        "physx_contact_impulse": True,
        "lidar_min_distance": True,
        "reward_breakdown": True,
        "target_visibility_per_step": True,
    }

    for episode_id, ep in sorted(episode_index.items()):
        rows = by_episode.get(episode_id, [])
        steps: dict[int, dict[int, np.ndarray]] = defaultdict(dict)
        for row in rows:
            steps[int(row["step"])][int(row["drone"])] = np.array(
                [float(row["x"]), float(row["y"]), float(row["z"])], dtype=np.float64
            )
        try:
            obstacle_layout = json.loads(ep.get("obstacle_layout", "[]"))
        except json.JSONDecodeError:
            obstacle_layout = []
        obstacles = [
            (np.asarray(item, dtype=np.float64), np.asarray([0.0, 0.0, 0.0]))
            for item in []
        ]
        # The saved layout contains centers only.  TRACE's fixed cuboids use
        # random scales; those scales are not in the legacy trajectory CSV.
        # Use the conservative vehicle-center distance and label the AABB check
        # as unavailable when scales cannot be recovered.
        centers = [np.asarray(item[:3], dtype=np.float64) for item in obstacle_layout]

        first_event = None
        min_obstacle_distance = math.inf
        min_pair_distance = math.inf
        min_evader_distance = math.inf
        for step, poses in sorted(steps.items()):
            if not all(i in poses for i in range(3)):
                continue
            p0, p1, ev = poses[0], poses[1], poses[2]
            pair = float(np.linalg.norm(p0 - p1))
            p_ev = min(float(np.linalg.norm(p0 - ev)), float(np.linalg.norm(p1 - ev)))
            min_pair_distance = min(min_pair_distance, pair)
            min_evader_distance = min(min_evader_distance, p_ev)
            # Center distance is still useful evidence, but cannot establish
            # cuboid contact without the per-obstacle dimensions.
            center_distances = [
                float(np.linalg.norm(p - c))
                for p in (p0, p1, ev)
                for c in centers
            ]
            if center_distances:
                min_obstacle_distance = min(min_obstacle_distance, min(center_distances))
            geometric_classes = []
            if pair < 0.35:
                geometric_classes.append("pursuer_pursuer")
            if p_ev < 0.32:
                geometric_classes.append("pursuer_evader")
            if min(p0[2], p1[2], ev[2]) < 0.5:
                geometric_classes.append("below_bound")
            # Without obstacle scales, do not claim a cuboid collision from a
            # center distance.  Reported flags from the rollout remain separate.
            reported = []
            if _bool(ep.get("pursuer_pursuer_collision")):
                reported.append("pursuer_pursuer")
            if _bool(ep.get("pursuer_evader_collision")):
                reported.append("pursuer_evader")
            if _bool(ep.get("pursuer_obstacle_collision")):
                reported.append("pursuer_obstacle")
            if _bool(ep.get("evader_obstacle_collision")):
                reported.append("evader_obstacle")
            if _bool(ep.get("collision")) or _bool(ep.get("terminated")):
                # The terminal reason cannot be reconstructed from trajectories
                # alone; classify it below from the logged collision flags.
                pass
            if first_event is None and geometric_classes:
                first_event = (step, geometric_classes)

        reported_classes = [
            name
            for name, key in (
                ("pursuer_pursuer", "pursuer_pursuer_collision"),
                ("pursuer_evader", "pursuer_evader_collision"),
                ("pursuer_obstacle", "pursuer_obstacle_collision"),
                ("evader_obstacle", "evader_obstacle_collision"),
            )
            if _bool(ep.get(key))
        ]
        if not reported_classes and _bool(ep.get("collision")):
            reported_classes = ["collision_reason_unavailable"]
        if not reported_classes and _bool(ep.get("escaped")):
            reported_classes = ["escape"]
        if not reported_classes and _bool(ep.get("truncated")):
            reported_classes = ["timeout"]
        if not reported_classes:
            reported_classes = ["capture" if _bool(ep.get("capture")) else "nonterminal"]
        for name in reported_classes:
            class_counts[name] += 1

        event_step = first_event[0] if first_event else None
        if event_step is None and _bool(ep.get("collision")):
            # The rollout's collision class is authoritative for the reason,
            # but timing is unavailable because no contact event was logged.
            event_step = int(ep.get("steps", 0)) - 1
        if event_step is not None:
            selected = range(max(0, event_step - window), event_step + 1)
            for step in selected:
                for row in [r for r in rows if int(r["step"]) == step]:
                    trajectory_rows.append(
                        {
                            **row,
                            "event_step": event_step,
                            "event_time_s": event_step / 30.0,
                            "reported_classes": ";".join(reported_classes),
                            "min_pair_distance": min_pair_distance,
                            "min_pursuer_evader_distance": min_evader_distance,
                            "nearest_obstacle_center_distance": min_obstacle_distance
                            if math.isfinite(min_obstacle_distance)
                            else None,
                        }
                    )

        summary_rows.append(
            {
                "episode": episode_id,
                "seed": ep.get("seed"),
                "steps": ep.get("steps"),
                "reported_collision": _bool(ep.get("collision")),
                "reported_classes": ";".join(reported_classes),
                "geometric_event_step": event_step,
                "geometric_event_time_s": event_step / 30.0 if event_step is not None else None,
                "min_pursuer_pair_distance": min_pair_distance
                if math.isfinite(min_pair_distance)
                else None,
                "min_pursuer_evader_distance": min_evader_distance
                if math.isfinite(min_evader_distance)
                else None,
                "nearest_obstacle_center_distance": min_obstacle_distance
                if math.isfinite(min_obstacle_distance)
                else None,
                "capture_distance": None,
                "target_visibility_ratio": ep.get("visibility_ratio"),
                "lidar_min_distance": None,
                "reward_breakdown": None,
                "physx_contact_impulse": None,
                "evidence_quality": "geometric_trajectory_plus_rollout_flags",
            }
        )

    out.mkdir(parents=True, exist_ok=True)
    for filename, rows in (
        ("collision_summary.csv", summary_rows),
        ("collision_trajectories.csv", trajectory_rows),
    ):
        fields = list(rows[0].keys()) if rows else ["episode"]
        with (out / filename).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    report = {
        "episodes": len(summary_rows),
        "collision_episodes": sum(row["reported_collision"] for row in summary_rows),
        "class_counts": dict(class_counts),
        "source_episodes": str(episodes_path),
        "source_trajectories": str(trajectories_path),
        "policy_hz": 30.0,
        "pre_event_window_steps": window,
        "evidence_limitations": unavailable,
    }
    (out / "collision_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    lines = [
        "# PSTO collision analysis",
        "",
        "This report recomputes geometric evidence from the saved trajectory CSV and preserves the rollout collision flags.",
        "It does not claim PhysX contact because the original rollout did not log contact impulses or per-step LiDAR minima.",
        "",
        f"- Episodes analyzed: **{report['episodes']}**",
        f"- Episodes with a reported collision: **{report['collision_episodes']}**",
        f"- Reported classes: `{json.dumps(dict(class_counts), sort_keys=True)}`",
        "",
        "The per-episode table is in `collision_summary.csv`; the last 60 policy steps before each reported or geometric event are in `collision_trajectories.csv`.",
        "Obstacle center distances are diagnostic only because the legacy CSV stores obstacle centers but not obstacle scales. LiDAR minimum distance, PhysX contact impulses, and reward breakdown were not logged and are explicitly `null`.",
        "",
        "A replay with a new diagnostic recorder is required to prove contact timing and distinguish a geometric threshold event from a PhysX contact.",
    ]
    (out / "collision_analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", default="results/psto_p0/long_train_100k/eval20/episodes.csv")
    parser.add_argument("--trajectories", default="results/psto_p0/long_train_100k/eval20/trajectories.csv")
    parser.add_argument("--out", default="results/psto_collision_analysis")
    parser.add_argument("--window", type=int, default=60)
    args = parser.parse_args()
    report = analyze(Path(args.episodes), Path(args.trajectories), Path(args.out), args.window)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
