"""Acceptance flights for gravity, hover, velocity tracking and motor-driven motion."""

import json
import traceback
from pathlib import Path

import numpy as np
from isaacsim import SimulationApp


def main():
    app = SimulationApp({"headless": True, "multi_gpu": False})
    env = None
    output = Path(__file__).resolve().parent / "artifacts/dynamics_checks.json"
    output.parent.mkdir(exist_ok=True)
    report = {"status": "running", "checks": {}}
    try:
        from navrl6_env import Isaac6NavEnv
        from quadrotor_dynamics import rotation_matrix

        env = Isaac6NavEnv(dynamics="rotor", obstacle=False)

        def passed(name, metrics):
            report["checks"][name] = metrics
            print(f"DYNAMICS_CHECK_OK {name} {json.dumps(metrics)}", flush=True)

        env.reset()
        for _ in range(48):
            env.world.step(render=False)  # NO forces: must fall under gravity
        env._read_physics_state()
        time = 48 * env.physics_dt
        expected_z = 2.0 - 0.5 * 9.81 * time**2
        assert abs(float(env.position[2]) - expected_z) < 0.015, env.position
        assert abs(float(env.velocity[2]) + 9.81 * time) < 0.05, env.velocity
        passed("free_fall", {"seconds": time, "z_m": float(env.position[2]),
                               "expected_z_m": expected_z, "vz_mps": float(env.velocity[2])})

        env.reset()
        start = env.position.copy()
        peak_tilt = 0.0
        for _ in range(180):
            _, _, _, info = env.step(np.zeros(3))
            peak_tilt = max(peak_tilt, info["tilt_degrees"])
        drift = float(np.linalg.norm(env.position - start))
        assert drift < 0.03 and np.linalg.norm(env.velocity) < 0.03, (drift, env.velocity)
        assert peak_tilt < 1.0, peak_tilt
        passed("hover", {"seconds": 180 * env.dt, "drift_m": drift,
                          "peak_tilt_deg": peak_tilt, "rotor_force_N": env.actuator.forces.tolist()})

        env.reset()
        peak_tilt = 0.0
        for i in range(90):
            _, _, _, info = env.step(np.array([1., 0., 0.]))
            peak_tilt = max(peak_tilt, info["tilt_degrees"])
            if i == 0:
                first_vx = float(env.velocity[0])
        assert abs(float(env.velocity[0]) - 1) < 0.12, env.velocity
        assert abs(float(env.position[2]) - 2) < 0.15, env.position
        assert 2 < peak_tilt < 35, peak_tilt
        assert 0 < first_vx < 0.25, first_vx
        passed("velocity_x", {"vx_after_one_action_mps": first_vx,
                               "velocity_after_3s_mps": env.velocity.tolist(),
                               "peak_tilt_deg": peak_tilt})

        brake_start = env.position.copy()
        minimum_pitch = 0.0
        for i in range(60):
            env.step(np.zeros(3))
            R = rotation_matrix(env.orientation)
            minimum_pitch = min(minimum_pitch, float(np.rad2deg(np.arctan2(R[0, 2], R[2, 2]))))
            if i == 0:
                brake_first_vx = float(env.velocity[0])
        distance = float(env.position[0] - brake_start[0])
        assert abs(float(env.velocity[0])) < 0.1, env.velocity
        assert 0.1 < distance < 1.0 and brake_first_vx > 0.5, (distance, brake_first_vx)
        assert minimum_pitch < -2.0, minimum_pitch
        passed("braking", {"distance_m": distance, "vx_after_one_action_mps": brake_first_vx,
                            "final_velocity_mps": env.velocity.tolist(), "minimum_pitch_deg": minimum_pitch})

        env.reset()
        for _ in range(90):
            env.step(np.array([0., 1., 0.]))
        assert abs(float(env.velocity[1]) - 1) < 0.12 and abs(float(env.velocity[0])) < 0.1, env.velocity
        passed("velocity_y", {"velocity_mps": env.velocity.tolist()})

        env.reset()
        for _ in range(60):
            env.step(np.array([0., 0., .5]))
        assert abs(float(env.velocity[2]) - .5) < .05 and env.position[2] > 2.6, (env.position, env.velocity)
        passed("climb", {"z_m": float(env.position[2]), "vz_mps": float(env.velocity[2])})

        env.reset()
        roll = np.deg2rad(12)
        env.drone.set_world_pose(orientation=np.array([np.cos(roll/2), np.sin(roll/2), 0., 0.]))
        for _ in range(120):
            _, _, _, info = env.step(np.zeros(3))
        assert info["tilt_degrees"] < 1.0 and np.linalg.norm(env.velocity) < .05, info
        passed("attitude_recovery", {"initial_roll_deg": 12., "final_tilt_deg": info["tilt_degrees"],
                                      "final_velocity_mps": env.velocity.tolist()})

        report["status"] = "passed"
        report["physics_hz"] = 1 / env.physics_dt
        report["policy_hz"] = 1 / env.dt
        print("ISAAC6_FULL_DYNAMICS_CHECKS_OK", flush=True)
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
        traceback.print_exc()
        print("ISAAC6_FULL_DYNAMICS_CHECKS_FAILED", report["error"], flush=True)
    finally:
        output.write_text(json.dumps(report, indent=2) + "\n")
        if env is not None:
            env.close()
        app.close()
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
