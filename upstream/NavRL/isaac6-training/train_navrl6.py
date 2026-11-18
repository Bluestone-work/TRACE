"""Train the migrated NavRL policy with the native Isaac Sim 6 adapter.

The default settings are intentionally small for migration verification.
Rotor mode trains through the velocity controller, motor model and PhysX;
increase ``--iterations`` and ``--steps`` for longer training runs.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from isaacsim import SimulationApp


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--dynamics", choices=("rotor", "kinematic"), default="rotor")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="optional checkpoint to resume from; defaults to released NavRL policy",
    )
    parser.add_argument("--output", type=Path, default=Path("navrl6_checkpoint.pt"))
    args = parser.parse_args()

    app = SimulationApp({"headless": True})
    env = None
    try:
        root = Path(__file__).resolve().parent
        sys.path.insert(0, str(root))
        # Importing train_smoke gives us the tested rollout and PPO update;
        # it does not start an app at import time.
        from train_smoke import _collect_rollout, _disable_isaac_fast_importer, _ppo_update

        quick = root.parents[0] / "quick-demos"
        sys.path.insert(0, str(quick))
        previous = _disable_isaac_fast_importer()
        try:
            from agent import Agent
        finally:
            sys.meta_path[:] = previous

        from navrl6_env import Isaac6NavEnv

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        env = Isaac6NavEnv(dynamics=args.dynamics)
        agent = Agent(device)
        if args.checkpoint is not None:
            checkpoint = args.checkpoint
            if not checkpoint.is_absolute():
                checkpoint = root / checkpoint
            agent.policy.load_state_dict(torch.load(checkpoint, map_location=device))
            print(f"ISAAC6_TRAINING_RESUME checkpoint={checkpoint}", flush=True)
        print(
            f"ISAAC6_TRAINING_START device={device} iterations={args.iterations} steps={args.steps}",
            flush=True,
        )
        for iteration in range(args.iterations):
            data = _collect_rollout(agent.policy, env, args.steps, device)
            stats = _ppo_update(agent.policy, data)
            print(
                f"ISAAC6_TRAINING_UPDATE iteration={iteration + 1}"
                f" actor_loss={stats['actor_loss']:.5f}"
                f" critic_loss={stats['critic_loss']:.5f}",
                flush=True,
            )

        output = args.output if args.output.is_absolute() else root / args.output
        torch.save(agent.policy.state_dict(), output)
        print(f"ISAAC6_TRAINING_DONE checkpoint={output}", flush=True)
        return 0
    finally:
        if env is not None:
            env.close()
        app.close()


if __name__ == "__main__":
    raise SystemExit(main())
