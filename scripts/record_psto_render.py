#!/usr/bin/env python3
"""Record a real Isaac Sim render of the three-drone PSTO scene.

This script uses the Hummingbird USD articulations and the shared PhysX world from
``trace.pursuit.scene``.  The controller is deliberately a small visual-playback
controller: it commands velocity vectors toward/away from the evader so that the
render shows physical motion.  It is not a MAPPO evaluation script.

Run with Isaac Sim 6.0.1, for example:

    /home/wj/isaacsim/python.sh scripts/record_psto_render.py \
        --headless --steps 60 --out results/psto_p0/isaac_render/psto_demo
"""
import argparse
import os
from pathlib import Path

import numpy as np


def _look_at(stage, path, eye, target):
    from pxr import Gf, UsdGeom

    cam = UsdGeom.Camera.Define(stage, path)
    # A close, cinematic view keeps the Hummingbird geometry readable.  The
    # global validation camera is intentionally wider; this recorder is for a
    # human looking at the actual rotorcraft in the PhysX scene.
    cam.CreateFocalLengthAttr(45.0)
    cam.CreateClippingRangeAttr(Gf.Vec2f(0.05, 200.0))
    UsdGeom.Xformable(cam).AddTransformOp().Set(
        Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eye), Gf.Vec3d(*target), Gf.Vec3d(0.0, 0.0, 1.0)).GetInverse()
    )
    return cam


def _visual_commands(positions, step):
    """Generate bounded world-frame velocity targets for a readable playback."""
    pursuers = positions[:2]
    evader = positions[2]
    commands = np.zeros((3, 3), dtype=np.float32)
    for i in range(2):
        direction = evader[:2] - pursuers[i, :2]
        norm = float(np.linalg.norm(direction))
        if norm > 1e-5:
            commands[i, :2] = direction / norm * 0.55
    away = evader[:2] - pursuers[:, :2].mean(axis=0)
    norm = float(np.linalg.norm(away))
    if norm > 1e-5:
        commands[2, :2] = away / norm * 0.35
    commands[2, :2] += np.array([0.0, 0.16 * np.sin(step / 8.0)], dtype=np.float32)
    return commands


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=4348)
    parser.add_argument("--out", default="results/psto_p0/isaac_render/psto_demo")
    parser.add_argument("--resolution", type=int, nargs=2, default=(960, 540))
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()
    if args.steps < 1:
        raise SystemExit("--steps must be positive")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    frame_dir = out.with_name(out.name + "_frames")
    frame_dir.mkdir(parents=True, exist_ok=True)

    from isaacsim import SimulationApp

    app = SimulationApp(
        {
            "headless": bool(args.headless),
            "enable_cameras": True,
            "multi_gpu": False,
            "width": int(args.resolution[0]),
            "height": int(args.resolution[1]),
        }
    )
    scene = None
    product = annotator = None
    try:
        import imageio.v2 as imageio
        import omni.replicator.core as rep
        from PIL import Image
        from trace.pursuit.scene import PSTOPursuitScene

        scene = PSTOPursuitScene(
            seed=args.seed,
            randomize_resets=True,
            randomize_obstacles=True,
        )
        scene.reset(randomize=True, seed=args.seed, randomize_obstacles=True)
        stage = scene.world.stage
        _look_at(stage, "/World/PSTORenderCamera", (5.5, -8.0, 5.0), (-1.0, 0.0, 2.0))
        product = rep.create.render_product(
            "/World/PSTORenderCamera", tuple(int(x) for x in args.resolution)
        )
        annotator = rep.AnnotatorRegistry.get_annotator("rgb")
        annotator.attach([product])
        scene.world.render()
        for _ in range(12):
            rep.orchestrator.step(delta_time=0.0, rt_subframes=2, pause_timeline=False)

        frames = []
        for step in range(args.steps):
            scene.step(_visual_commands(np.stack([v.position for v in scene.vehicles]), step))
            scene.world.render()
            rep.orchestrator.step(delta_time=0.0, rt_subframes=2, pause_timeline=False)
            data = np.asarray(annotator.get_data())
            if data.ndim != 3 or data.shape[-1] < 3:
                raise RuntimeError(f"unexpected RGB render shape: {data.shape}")
            frame = np.asarray(data[..., :3], dtype=np.uint8)
            frames.append(frame)
            if step in (0, args.steps - 1):
                Image.fromarray(frame).save(frame_dir / f"frame_{step:04d}.png")
            print(f"[{step + 1}/{args.steps}] rendered", flush=True)
        gif_path = out.with_suffix(".gif")
        imageio.mimsave(gif_path, frames, duration=1.0 / max(args.fps, 1), loop=0)
        print(f"[OK] Isaac Sim render GIF: {gif_path} ({gif_path.stat().st_size / 1e6:.2f} MB)", flush=True)
        print(f"[OK] first/last PNG frames: {frame_dir}", flush=True)
    finally:
        if annotator is not None and product is not None:
            try:
                annotator.detach([product])
                product.destroy()
            except Exception:
                pass
        if scene is not None:
            scene.close()
        app.close()


if __name__ == "__main__":
    main()
