"""Run the original NavRL neural policy inside Isaac Sim 6.0.1."""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from isaacsim import SimulationApp


def _disable_isaac_fast_importer() -> list[object]:
    """Temporarily let Python resolve torch submodules from torch_site.

    Isaac Sim installs ``omni.ext._impl.fast_importer.FastFinder`` on
    ``sys.meta_path``.  In Isaac Sim 6 it still exposes the deprecated
    ``omni.isaac.ml_archive`` Torch submodules.  The top-level ``torch``
    package is supplied by ``torch_site`` through PYTHONPATH, so allowing the
    FastFinder to resolve ``torch._dynamo`` mixes two incompatible Torch
    versions.  We only remove it while importing the NavRL Torch stack and
    restore it before importing the Isaac environment.
    """
    previous = list(sys.meta_path)
    sys.meta_path[:] = [
        finder
        for finder in sys.meta_path
        if not (
            getattr(finder, "__name__", "") == "FastFinder"
            or getattr(finder, "__module__", "").startswith(
                "omni.ext._impl.fast_importer"
            )
        )
    ]
    return previous


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--gui", action="store_true", help="open the Isaac Sim viewport")
    parser.add_argument("--loop", action="store_true", help="repeat until the window is closed")
    parser.add_argument("--realtime", action="store_true", help="limit playback to 30 simulation steps/s")
    parser.add_argument("--screenshot", type=Path, help="save the viewport after --steps steps")
    parser.add_argument("--record-dir", type=Path, help="save viewport PNG frames for replay")
    parser.add_argument("--record-every", type=int, default=6, help="steps between recorded frames")
    parser.add_argument("--style", choices=("debug", "realistic"), default="debug")
    parser.add_argument("--dynamics", choices=("rotor", "kinematic"), default="rotor")
    parser.add_argument("--report", type=Path, help="write aggregate evaluation metrics as JSON")
    parser.add_argument("--camera", choices=("overview", "follow"), default=None)
    parser.add_argument("--show-sensors", action="store_true", help="show LiDAR and trajectory in realistic mode")
    parser.add_argument("--renderer", choices=("RealTimePathTracing", "PathTracing"),
                        default="RealTimePathTracing")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="optional NavRL state_dict checkpoint to evaluate",
    )
    args = parser.parse_args()
    if args.steps < 1 or args.record_every < 1:
        parser.error("--steps and --record-every must be positive")
    if (args.loop or args.screenshot or args.record_dir) and not args.gui:
        parser.error("--loop, --screenshot and --record-dir require --gui")
    print("TORCH_POLICY_START", flush=True)
    app = SimulationApp({"headless": not args.gui, "width": 1280, "height": 720,
                         "multi_gpu": False, "renderer": args.renderer,
                         "samples_per_pixel_per_frame": 32})
    print("TORCH_POLICY_APP", flush=True)
    env = None
    try:
        # quick-demos contains the released network definition and checkpoint.
        quick = Path(__file__).resolve().parents[1] / "quick-demos"
        sys.path.insert(0, str(quick))
        previous_meta_path = _disable_isaac_fast_importer()
        try:
            from agent import Agent
            from tensordict import TensorDict
            from torchrl.envs.utils import ExplorationType, set_exploration_type
        finally:
            sys.meta_path[:] = previous_meta_path
        print("TORCH_POLICY_IMPORTS_OK", flush=True)

        from navrl6_env import Isaac6NavEnv
        print("TORCH_POLICY_NAVRL_IMPORT_OK", flush=True)

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        agent = Agent(device)
        if args.checkpoint is not None:
            checkpoint = args.checkpoint
            if not checkpoint.is_absolute():
                checkpoint = Path(__file__).resolve().parent / checkpoint
            agent.policy.load_state_dict(torch.load(checkpoint, map_location=device))
            print(f"TORCH_POLICY_CHECKPOINT={checkpoint}", flush=True)
        else:
            checkpoint = quick / "ckpts/navrl_checkpoint.pt"
        print(f"ISAAC6_LEARNED_POLICY checkpoint={checkpoint} evaluation=mean_action", flush=True)
        env = Isaac6NavEnv(dynamics=args.dynamics)
        obs = env.reset()
        viewer = None
        if args.gui:
            from navrl6_viewer import NavRLViewer
            viewer = NavRLViewer(env, style=args.style, camera=args.camera,
                                 show_sensors=args.show_sensors)
            print("ISAAC6_VIEWER_READY", flush=True)
        episode = 1
        step = 0
        metrics = {"dynamics": args.dynamics, "steps": 0, "reached_goal": 0,
                   "collisions": 0, "failed": 0, "timeouts": 0, "peak_tilt_degrees": 0.0}
        with torch.inference_mode(), set_exploration_type(ExplorationType.MEAN):
            while app.is_running() and (args.loop or step < args.steps):
                started = time.monotonic()
                if not env.world.is_playing():
                    env.world.render()
                    time.sleep(0.01)
                    continue
                td = TensorDict(
                    {
                        "agents": {
                            "observation": {
                                "state": torch.as_tensor(obs["state"], device=device).reshape(1, 8),
                                "lidar": torch.as_tensor(obs["lidar"], device=device).reshape(1, 1, 36, 4),
                                "direction": torch.as_tensor(obs["direction"], device=device).reshape(1, 1, 3),
                                "dynamic_obstacle": torch.as_tensor(
                                    obs["dynamic_obstacle"], device=device
                                ).reshape(1, 1, 5, 10),
                            }
                        }
                    },
                    batch_size=[1],
                    device=device,
                )
                out = agent.policy(td)
                action = out["agents", "action"].detach().cpu().numpy().reshape(3)
                obs, reward, terminated, info = env.step(action)
                metrics["peak_tilt_degrees"] = max(metrics["peak_tilt_degrees"], info["tilt_degrees"])
                if viewer is not None:
                    viewer.update(obs, reward, episode)
                if step == 0:
                    print(f"ISAAC6_TORCH_POLICY_OK action={action.tolist()} reward={reward:.3f}")
                step += 1
                if args.record_dir and step % args.record_every == 0:
                    viewer.screenshot(args.record_dir / f"frame_{step:06d}.png")
                if args.screenshot and step == args.steps:
                    viewer.screenshot(args.screenshot)
                if terminated:
                    metrics["reached_goal"] += int(info["distance"] < 0.5 and not info["failed"])
                    metrics["collisions"] += int(info["collision"])
                    metrics["failed"] += int(info["failed"])
                    metrics["timeouts"] += int(env.step_count >= 900)
                    obs = env.reset()
                    episode += 1
                if args.realtime:
                    time.sleep(max(0.0, env.dt - (time.monotonic() - started)))
        print(f"ISAAC6_TORCH_POLICY_DONE steps={step} position={env.position.tolist()}")
        metrics["steps"] = step
        print("ISAAC6_POLICY_EVALUATION " + json.dumps(metrics), flush=True)
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps(metrics, indent=2) + "\n")
        return 0
    except Exception:
        # Print before app.close(): Kit fast shutdown may otherwise hide errors.
        traceback.print_exc()
        raise
    finally:
        if env is not None:
            env.close()
        app.close()


if __name__ == "__main__":
    raise SystemExit(main())
