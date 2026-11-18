"""Isaac Sim 6.0.1 smoke test for the first NavRL migration stage."""

from __future__ import annotations

import argparse
import numpy as np

from isaacsim import SimulationApp


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--policy", action="store_true", help="run the exported NavRL checkpoint")
    parser.add_argument("--headless", action="store_true", default=True)
    args = parser.parse_args()

    # Isaac Sim must be launched before importing omni/isaacsim runtime modules.
    app = SimulationApp({"headless": True})
    try:
        from navrl6_env import Isaac6NavEnv

        env = Isaac6NavEnv()
        policy = None
        if args.policy:
            from navrl_policy_numpy import NavRLPolicy

            policy = NavRLPolicy()
        obs = env.reset()
        assert obs["state"].shape == (8,), obs["state"].shape
        assert obs["lidar"].shape == (1, 36, 4), obs["lidar"].shape

        for step in range(args.steps):
            # Same high-level action contract as NavRL: world-frame velocity in m/s.
            action = np.array([0.8, 0.0, 0.0], dtype=np.float32)
            if policy is not None:
                action = policy(obs["state"], obs["lidar"], obs["direction"])
            obs, reward, terminated, info = env.step(action)
            if not np.isfinite(obs["lidar"]).all():
                raise RuntimeError("LiDAR observation contains NaN/Inf")
            if step == 0:
                print(
                    "ISAAC6_NAVRL_SMOKE_OK"
                    f" state={obs['state'].shape} lidar={obs['lidar'].shape}"
                    f" reward={reward:.3f}"
                )
            if terminated:
                obs = env.reset()

        pos = env.position.copy()
        print(f"ISAAC6_NAVRL_SMOKE_DONE steps={args.steps} position={pos.tolist()}")
        return 0
    finally:
        try:
            env.close()
        except Exception:
            pass
        app.close()


if __name__ == "__main__":
    raise SystemExit(main())
