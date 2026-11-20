# PSTO collision analysis

This report recomputes geometric evidence from the saved trajectory CSV and preserves the rollout collision flags.
It does not claim PhysX contact because the original rollout did not log contact impulses or per-step LiDAR minima.

- Episodes analyzed: **20**
- Episodes with a reported collision: **14**
- Reported classes: `{"capture": 5, "pursuer_obstacle": 14, "timeout": 1}`

The per-episode table is in `collision_summary.csv`; the last 60 policy steps before each reported or geometric event are in `collision_trajectories.csv`.
Obstacle center distances are diagnostic only because the legacy CSV stores obstacle centers but not obstacle scales. LiDAR minimum distance, PhysX contact impulses, and reward breakdown were not logged and are explicitly `null`.

A replay with a new diagnostic recorder is required to prove contact timing and distinguish a geometric threshold event from a PhysX contact.
