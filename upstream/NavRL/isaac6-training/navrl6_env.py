"""Native Isaac Sim 6 NavRL environment, with rotor dynamics or a kinematic adapter."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

from rotor_flight import RotorFlight


class Isaac6NavEnv(RotorFlight):
    """Single-agent NavRL-compatible environment running on Isaac Sim 6."""

    lidar_range = 4.0
    lidar_vfov = (-10.0, 20.0)
    lidar_vbeams = 4
    lidar_hres = 10.0
    lidar_hbeams = 36
    action_limit = 2.0
    dt = 1.0 / 30.0

    def __init__(self, dynamics="rotor", obstacle=True) -> None:
        from isaacsim.core.api import World
        from isaacsim.core.api.objects import FixedCuboid
        from isaacsim.core.prims import SingleArticulation, RigidPrim
        from isaacsim.core.utils.stage import add_reference_to_stage
        from pxr import PhysicsSchemaTools, PhysxSchema, Usd, UsdPhysics

        if dynamics not in ("rotor", "kinematic"):
            raise ValueError(f"unknown dynamics mode: {dynamics}")
        self.dynamics = dynamics
        self.physics_substeps = 8 if dynamics == "rotor" else 1
        self.physics_dt = self.dt / self.physics_substeps
        self._World = World
        self._FixedCuboid = FixedCuboid
        self._SingleArticulation = SingleArticulation
        self._add_reference_to_stage = add_reference_to_stage
        self.world = World(physics_dt=self.physics_dt, rendering_dt=self.dt, stage_units_in_meters=1.0)

        hummingbird_usd = (
            Path(__file__).resolve().parents[1]
            / "isaac-training/third_party/OmniDrones/omni_drones/robots/assets/usd/hummingbird.usd"
        )
        if not hummingbird_usd.exists():
            raise FileNotFoundError(hummingbird_usd)
        add_reference_to_stage(str(hummingbird_usd), "/World/Hummingbird")
        # Rotor flight integrates forces under gravity; the comparison-only
        # kinematic adapter disables gravity while setting poses directly.
        self._self_query_ids = set()
        for prim in Usd.PrimRange(self.world.stage.GetPrimAtPath("/World/Hummingbird")):
            self._self_query_ids.add(PhysicsSchemaTools.sdfPathToInt(prim.GetPath()))
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                body = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
                body.CreateDisableGravityAttr(dynamics == "kinematic")
                body.CreateLinearDampingAttr(0.0)
                body.CreateAngularDampingAttr(0.0)
        self.drone = SingleArticulation("/World/Hummingbird", name="navrl_hummingbird")
        self.world.scene.add(self.drone)
        self.base_view = self.rotor_view = self.actuator = None
        if dynamics == "rotor":
            from quadrotor_dynamics import QuadrotorActuator
            self.actuator = QuadrotorActuator(self.physics_dt)
            self.base_view = RigidPrim("/World/Hummingbird/base_link", name="navrl_body")
            self.rotor_view = RigidPrim("/World/Hummingbird/rotor_[0-3]", name="navrl_rotors")
            self.world.scene.add(self.base_view)
            self.world.scene.add(self.rotor_view)
        self.ground = FixedCuboid(
            prim_path="/World/Ground",
            name="ground",
            position=np.array([0.0, 0.0, -0.05]),
            scale=np.array([100.0, 100.0, 0.1]),
            size=1.0,
            color=np.array([0.15, 0.15, 0.15]),
        )
        self.world.scene.add(self.ground)
        self.obstacle = FixedCuboid(
            prim_path="/World/Obstacle",
            name="obstacle",
            position=np.array([2.0, 0.0, 1.5]),
            scale=np.array([1.0, 1.0, 3.0]),
            size=1.0,
            color=np.array([0.9, 0.15, 0.1]),
        )
        self.world.scene.add(self.obstacle)
        if not obstacle:
            self.world.scene.remove_object("obstacle")

        self.position = np.zeros(3, dtype=np.float32)
        self.velocity = np.zeros(3, dtype=np.float32)
        self.orientation = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        self.angular_velocity = np.zeros(3, dtype=np.float32)
        self.goal_direction = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        self.lidar_yaw = 0.0
        self.target = np.array([8.0, 0.0, 2.0], dtype=np.float32)
        self.step_count = 0
        self.world.reset()
        if dynamics == "rotor":
            self.rotor_joint_indices = np.array([self.drone.get_dof_index(f"rotor_{i}_joint")
                                                 for i in range(4)])
            print(f"ISAAC6_ROTOR_DYNAMICS physics_dt={self.physics_dt:.6f}"
                  f" base_mass={self.base_view.get_masses().tolist()}"
                  f" rotor_masses={self.rotor_view.get_masses().tolist()}"
                  f" joints={self.drone.dof_names}", flush=True)

        # The query interface is available after the physics scene is created.
        import omni.physics.core

        self._query = omni.physics.core.get_physics_scene_query_interface()

    def reset(self) -> dict[str, np.ndarray]:
        self.world.reset()
        self.drone.set_world_pose(position=np.array([0.0, 0.0, 2.0]),
                                  orientation=np.array([1.0, 0.0, 0.0, 0.0]))
        self.drone.set_linear_velocity(np.zeros(3))
        self.drone.set_angular_velocity(np.zeros(3))
        self.position = np.array([0.0, 0.0, 2.0], dtype=np.float32)
        self.velocity = np.zeros(3, dtype=np.float32)
        self.orientation = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        self.angular_velocity = np.zeros(3, dtype=np.float32)
        self.goal_direction = self.target - self.position
        self.goal_direction[2] = 0.0
        self.goal_direction /= max(float(np.linalg.norm(self.goal_direction)), 1e-6)
        if self.actuator is not None:
            self.actuator.reset()
            self.drone.set_joint_positions(np.zeros(4), joint_indices=self.rotor_joint_indices)
            self.drone.set_joint_velocities(self.actuator.rotor_speeds.astype(np.float32),
                                            joint_indices=self.rotor_joint_indices)
        self.step_count = 0
        self.lidar_yaw = 0.0
        # Flush the teleport at reset to USD/query geometry without a physics step.
        self.world.render()
        return self._observation()

    def close(self) -> None:
        self.world.clear_instance()

    def step(self, action: np.ndarray) -> tuple[dict[str, np.ndarray], float, bool, dict[str, Any]]:
        action = np.asarray(action, dtype=np.float32).reshape(3)
        if not np.isfinite(action).all():
            raise ValueError("nonfinite velocity action")
        action = np.clip(action, -self.action_limit, self.action_limit)
        previous_velocity = self.velocity.copy()
        if self.dynamics == "rotor":
            self._step_rotor_dynamics(action)
        else:
            self._step_kinematic(action)
        self.step_count += 1
        obs = self._observation()
        distance = float(np.linalg.norm(self.target - self.position))
        clearance = self.lidar_range - obs["lidar"]
        collision = bool(np.min(clearance) < 0.3)
        from quadrotor_dynamics import rotation_matrix
        tilt = float(np.arccos(np.clip(rotation_matrix(self.orientation)[2, 2], -1.0, 1.0)))
        failed = collision or self.position[2] < 0.2 or self.position[2] > 6.0 or tilt > np.deg2rad(80)
        # Static NavRL reward: goalward speed, log clearance, velocity smoothness,
        # and height penalty (the original dynamic-obstacle reward is absent).
        direction = (self.target - self.position) / max(distance, 1e-6)
        height_error = max(0.0, abs(float(self.position[2]) - float(self.target[2])) - 0.2)
        reward = float(np.dot(self.velocity, direction) + 1.0
                       + np.log(np.clip(clearance, 1e-6, self.lidar_range)).mean()
                       - 0.1 * np.linalg.norm(self.velocity - previous_velocity)
                       - 8.0 * height_error**2)
        terminated = bool(failed or distance < 0.5 or self.step_count >= 900)
        info = {"distance": distance, "collision": collision, "failed": bool(failed),
                "tilt_degrees": float(np.rad2deg(tilt)), "velocity_command": action.copy(),
                "actual_velocity": self.velocity.copy(), "dynamics": self.dynamics}
        if self.actuator is not None:
            info.update(rotor_forces=self.actuator.forces.copy(),
                        rotor_speeds=self.actuator.rotor_speeds.copy(),
                        actuator_saturated=self.actuator.saturated)
        return obs, reward, terminated, info

    def _step_kinematic(self, action):
        self.velocity = action.copy()
        # Comparison mode only: write the pose before the rendering update.
        next_position = self.position.astype(np.float64) + self.velocity.astype(np.float64) * self.dt
        next_position[2] = max(0.4, min(6.0, next_position[2]))
        self.drone.set_world_pose(position=next_position,
                                  orientation=np.array([1.0, 0.0, 0.0, 0.0]))
        self.drone.set_linear_velocity(np.zeros(3))
        self.drone.set_angular_velocity(np.zeros(3))
        self.world.step(render=False)
        # Keep the sensor origin aligned with the actual articulation root,
        # including any contact correction applied by PhysX.
        self.position = np.asarray(self.drone.get_world_pose()[0], dtype=np.float32)

    def _step_rotor_dynamics(self, action):
        for _ in range(self.physics_substeps):
            self._apply_rotor_control(action)
            self.world.step(render=False)
        self._read_physics_state()

    def _observation(self, lidar_yaw=None) -> dict[str, np.ndarray]:
        lidar = self._raycast_lidar(yaw=lidar_yaw)
        direction = self.target - self.position
        norm = np.linalg.norm(direction)
        direction = direction / max(norm, 1e-6)
        ex = self.goal_direction
        ey = np.array([-ex[1], ex[0], 0.0], dtype=np.float32)
        ez = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        relative_local = np.array([np.dot(direction, ex), np.dot(direction, ey), np.dot(direction, ez)])
        velocity_local = np.array([np.dot(self.velocity, ex), np.dot(self.velocity, ey), np.dot(self.velocity, ez)])
        state = np.concatenate([relative_local, [np.linalg.norm(self.target[:2] - self.position[:2]),
                                                   self.target[2] - self.position[2]], velocity_local]).astype(np.float32)
        return {
            "state": state,
            "lidar": lidar[None, ...].astype(np.float32),
            "direction": self.goal_direction[None, ...].astype(np.float32),
            "dynamic_obstacle": np.zeros((1, 5, 10), dtype=np.float32),
        }

    def _raycast_lidar(self, yaw=None) -> np.ndarray:
        origin = self.position.astype(np.float64)
        scan_yaw = self.lidar_yaw if yaw is None else float(yaw)
        distances = np.full((self.lidar_hbeams, self.lidar_vbeams), self.lidar_range, dtype=np.float32)
        vertical = np.deg2rad(np.linspace(self.lidar_vfov[0], self.lidar_vfov[1], self.lidar_vbeams))
        for hi in range(self.lidar_hbeams):
            ray_yaw = 2.0 * math.pi * hi / self.lidar_hbeams + scan_yaw
            for vi, pitch in enumerate(vertical):
                direction = np.array(
                    [math.cos(pitch) * math.cos(ray_yaw), math.cos(pitch) * math.sin(ray_yaw), math.sin(pitch)]
                )
                def report_hit(hit):
                    # Ignore drone links, while continuing to look for the
                    # nearest external surface behind any self hit.
                    if hit.rigid_body not in self._self_query_ids and hit.collision not in self._self_query_ids:
                        distances[hi, vi] = min(distances[hi, vi], float(hit.distance))
                    return True

                self._query.raycast_all(
                    tuple(float(x) for x in origin), tuple(float(x) for x in direction),
                    self.lidar_range, report_hit, True
                )
        # NavRL's original quick demo represents clearance as max_range-distance.
        return self.lidar_range - distances
