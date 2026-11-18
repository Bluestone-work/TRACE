"""A: released NavRL random-goal navigation. B: RGB-only search and follow."""

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from isaacsim import SimulationApp


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--realtime", action="store_true")
    parser.add_argument("--seconds", type=float, default=60.)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--speed-limit", type=float, default=2.)
    parser.add_argument("--initial-distance", type=float, default=10.)
    parser.add_argument("--follow-distance", type=float, default=3.5)
    parser.add_argument("--debug-scene", action="store_true")
    parser.add_argument("--scene", choices=("hall", "forest"), default="hall",
                        help="presentation scene; forest uses procedural tree trunks and foliage")
    parser.add_argument("--hide-target", action="store_true", help="negative control: hide visual marker")
    parser.add_argument("--camera-dropout-start", type=float, default=-1.)
    parser.add_argument("--camera-dropout-duration", type=float, default=2.)
    parser.add_argument("--report", type=Path, default=Path("artifacts/tracking_demo_report.json"))
    parser.add_argument("--record-dir", type=Path)
    parser.add_argument("--record-every", type=int, default=30)
    args = parser.parse_args()
    if args.seconds <= 0 or not 0 < args.speed_limit <= 2. or not 6 <= args.initial_distance <= 10:
        parser.error("require seconds>0, 0<speed-limit<=2, 6<=initial-distance<=10")
    if not 2.5 <= args.follow_distance <= 6. or args.record_every < 1:
        parser.error("require 2.5<=follow-distance<=6 and record-every>=1")
    if (args.loop or args.record_dir) and not args.gui:
        parser.error("--loop and --record-dir require --gui")
    root = Path(__file__).resolve().parent
    args.report = args.report if args.report.is_absolute() else root / args.report
    args.report.parent.mkdir(parents=True, exist_ok=True)
    if args.record_dir:
        args.record_dir = args.record_dir if args.record_dir.is_absolute() else root / args.record_dir
    app = SimulationApp({"headless": not args.gui, "width": 1280, "height": 720,
                         "multi_gpu": False, "renderer": "RealTimePathTracing",
                         "samples_per_pixel_per_frame": 16})
    scene = camera = None
    report = {"status": "running", "seed": args.seed, "speed_limit_A_mps": args.speed_limit,
              "speed_limit_B_mps": args.speed_limit, "equal_dynamics": True,
              "A_algorithm": "released NavRL PPO / random goals / mean action",
              "B_algorithm": "rendered RGB marker / alpha-beta filter / search-track / LiDAR avoidance",
              "B_privileged_target_state": False, "physics_hz": 240, "policy_hz": 30,
              "camera_hz": 10, "camera_resolution": [640, 360],
              "initial_separation_m": args.initial_distance,
              "frames": 0, "detections": 0, "mode_counts": {}, "failures": [],
              "max_command_speed_A": 0., "max_command_speed_B": 0.,
              "peak_actual_speed_A": 0., "peak_actual_speed_B": 0.,
              "scene": args.scene}
    log_path = args.report.with_suffix(".jsonl")
    log = log_path.open("w", buffering=1)
    try:
        from run_torch_policy import _disable_isaac_fast_importer
        sys.path.insert(0, str(root.parent / "quick-demos"))
        previous = _disable_isaac_fast_importer()
        try:
            from agent import Agent
            from tensordict import TensorDict
            from torchrl.envs.utils import ExplorationType, set_exploration_type
        finally:
            sys.meta_path[:] = previous
        from tracking_scene import TrackingRGBCamera, TrackingScene
        from visual_tracker import VisualSearchTracker, avoid_obstacles, detect_marker

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        agent = Agent(device)
        scene = TrackingScene(seed=args.seed, speed_limit=args.speed_limit,
                              initial_distance=args.initial_distance, realistic=not args.debug_scene,
                              scene_type=args.scene)
        tracker = VisualSearchTracker(scene.follower.position, args.speed_limit, args.follow_distance)
        camera = TrackingRGBCamera(scene.world)
        if args.hide_target:
            from pxr import UsdGeom
            UsdGeom.Imageable(scene.marker).MakeInvisible()
        camera.set_pose(scene.follower.position, tracker.camera_yaw)
        for _ in range(12):
            scene.world.render()  # Warm up sensor without advancing simulation.
        if camera.capture() is None:
            raise RuntimeError("RGB camera produced no pixels after warmup")
        viewer = None
        if args.gui:
            from tracking_viewer import TrackingViewer
            viewer = TrackingViewer(scene)
        print("TRACKING_DEMO_READY A=NavRL B=RGB_only equal_speed_limits=True", flush=True)
        command_b = np.zeros(3)
        steps = int(np.ceil(args.seconds / scene.dt))
        last_mode = None
        separations, errors = [], []
        physics_start = scene.world.current_time
        report["initial_separation_m"] = float(np.linalg.norm(scene.leader.position - scene.follower.position))
        with torch.inference_mode(), set_exploration_type(ExplorationType.MEAN):
            while app.is_running() and (args.loop or scene.steps < steps):
                started = time.monotonic()
                if not scene.world.is_playing():
                    scene.world.render()
                    time.sleep(0.01)
                    continue
                now = scene.steps * scene.dt
                scene.maybe_change_goal()
                # Original NavRL starts facing its segment goal. Re-bin the
                # 360-degree scan into that frame when changing goals, without
                # teleporting yaw or changing the physical vehicle state.
                goal_yaw = np.arctan2(scene.leader.goal_direction[1], scene.leader.goal_direction[0])
                obs = scene.leader._observation(lidar_yaw=goal_yaw)
                td = TensorDict({"agents": {"observation": {
                    key: torch.as_tensor(value, device=device).reshape(shape)
                    for key, value, shape in (
                        ("state", obs["state"], (1, 8)),
                        ("lidar", obs["lidar"], (1, 1, 36, 4)),
                        ("direction", obs["direction"], (1, 1, 3)),
                        ("dynamic_obstacle", obs["dynamic_obstacle"], (1, 1, 5, 10)))
                }}}, batch_size=[1], device=device)
                command_a = agent.policy(td)["agents", "action"].cpu().numpy().reshape(3)
                if scene.steps % 3 == 0:
                    b = scene.follower
                    camera.set_pose(b.position, tracker.camera_yaw)
                    rgb = camera.capture()
                    if rgb is None:
                        raise RuntimeError("RGB stream became empty")
                    dropout = (args.camera_dropout_start >= 0 and args.camera_dropout_start <= now
                               < args.camera_dropout_start + args.camera_dropout_duration)
                    if dropout:
                        rgb = np.zeros_like(rgb)
                    detection = detect_marker(rgb, camera.position, camera.yaw,
                                               horizontal_fov=camera.horizontal_fov)
                    command_b = tracker.update(detection, b.position, now, scene.dt * 3)
                    if scene.steps == 0 or (detection is not None and report["detections"] == 0):
                        from tracking_viewer import annotated_rgb
                        suffix = "first_detection" if detection is not None else "initial_search"
                        annotated_rgb(rgb, detection, tracker.mode).save(
                            args.report.with_name(args.report.stem + "_" + suffix + ".png"))
                    command_b = avoid_obstacles(command_b, b._raycast_lidar(), b.lidar_yaw,
                                                args.speed_limit)
                    # Everything below is evaluation/display; it never feeds the tracker.
                    a = scene.leader
                    separation = float(np.linalg.norm(a.position - b.position))
                    separations.append(separation)
                    report["frames"] += 1
                    report["detections"] += int(detection is not None)
                    report["mode_counts"][tracker.mode] = report["mode_counts"].get(tracker.mode, 0) + 1
                    error = None
                    if detection is not None:
                        error = float(np.linalg.norm(detection.position - a.position))
                        errors.append(error)
                    row = {"time_s": now, "mode": tracker.mode, "detected": detection is not None,
                           "camera_dropout": dropout, "bbox": detection.bbox if detection else None,
                           "visual_range_m": detection.range_m if detection else None,
                           "visual_position_error_m_evaluation_only": error,
                           "separation_m_evaluation_only": separation,
                           "A_position_evaluation_only": a.position.tolist(), "B_position": b.position.tolist(),
                           "A_speed": float(np.linalg.norm(a.velocity)), "B_speed": float(np.linalg.norm(b.velocity))}
                    log.write(json.dumps(row) + "\n")
                    if tracker.mode != last_mode or scene.steps % 150 == 0:
                        print(f"TRACKING_STATE t={now:.1f} mode={tracker.mode}"
                              f" detected={detection is not None} separation_eval={separation:.2f}", flush=True)
                        last_mode = tracker.mode
                    if viewer:
                        viewer.update(rgb, detection, tracker, now)
                        if args.record_dir and scene.steps % args.record_every == 0:
                            viewer.record(args.record_dir, scene.steps, tracker)
                ca, cb = scene.step(command_a, command_b)
                for name, drone, command in (("A", scene.leader, ca), ("B", scene.follower, cb)):
                    report["max_command_speed_" + name] = max(report["max_command_speed_" + name],
                                                              float(np.linalg.norm(command)))
                    report["peak_actual_speed_" + name] = max(report["peak_actual_speed_" + name],
                                                              float(np.linalg.norm(drone.velocity)))
                if scene.steps % 3 == 0:
                    failures = scene.failures()
                    if failures:
                        report["failures"].extend(failures)
                        raise RuntimeError(f"flight acceptance failed at t={now:.1f}: {failures}")
                if args.realtime:
                    time.sleep(max(0., scene.dt - (time.monotonic() - started)))
        report.update(status="passed", simulation_seconds=scene.steps * scene.dt,
                      measured_physics_seconds=scene.world.current_time - physics_start,
                      first_acquisition_s=tracker.first_seen, acquisitions=tracker.acquisitions,
                      A_waypoints_reached=scene.waypoints_reached, A_goal_history=scene.goals,
                      final_separation_m=separations[-1], minimum_separation_m=min(separations),
                      detection_fraction=report["detections"] / max(report["frames"], 1),
                      median_visual_position_error_m=float(np.median(errors)) if errors else None,
                      final_10s_mean_separation_m=float(np.mean(separations[-100:])),
                      trajectory_log=str(log_path))
        checks = {"equal_command_limits": max(report["max_command_speed_A"],
                                               report["max_command_speed_B"]) <= args.speed_limit + 1e-6,
                  "no_flight_failures": not report["failures"],
                  "one_shared_physics_clock": abs(report["measured_physics_seconds"]
                                                    - report["simulation_seconds"]) < 1e-5}
        if args.hide_target:
            checks["hidden_marker_never_acquired"] = report["detections"] == 0
        elif report["simulation_seconds"] >= 20.:
            checks.update(visual_acquisition=tracker.first_seen is not None,
                          visual_tracking_majority=report["detection_fraction"] > 0.5,
                          following_distance=report["final_10s_mean_separation_m"] < 6.,
                          A_random_navigation=scene.waypoints_reached >= 2)
        report["acceptance_checks"] = checks
        if not all(checks.values()):
            raise RuntimeError(f"demo acceptance failed: {checks}")
        print("TRACKING_DEMO_DONE " + json.dumps(report), flush=True)
        return 0
    except Exception as exc:
        report.update(status="failed", error=str(exc))
        traceback.print_exc()
        return 1
    finally:
        log.close()
        args.report.write_text(json.dumps(report, indent=2) + "\n")
        if camera:
            camera.close()
        if scene:
            scene.close()
        app.close()


if __name__ == "__main__":
    raise SystemExit(main())
