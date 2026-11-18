"""Shared force application for vehicles stepped in one PhysX world."""

import numpy as np

from quadrotor_dynamics import rotation_matrix


class RotorFlight:
    def _read_physics_state(self):
        pos, quat = self.base_view.get_world_poses()
        self.position = np.asarray(pos[0], dtype=np.float32).copy()
        self.orientation = np.asarray(quat[0], dtype=np.float32).copy()
        self.velocity = np.asarray(self.base_view.get_linear_velocities()[0], dtype=np.float32).copy()
        self.angular_velocity = np.asarray(self.base_view.get_angular_velocities()[0], dtype=np.float32).copy()
        R = rotation_matrix(self.orientation)
        self.lidar_yaw = float(np.arctan2(R[1, 0], R[0, 0]))
        if not np.isfinite(np.r_[self.position, self.orientation, self.velocity, self.angular_velocity]).all():
            raise RuntimeError("nonfinite PhysX drone state")

    def _apply_rotor_control(self, action):
        """Apply one force update WITHOUT stepping the shared world."""
        self._read_physics_state()
        requested = self.actuator.command(self.orientation, self.velocity, self.angular_velocity, action)
        forces, moments = self.actuator.motor_step(requested)
        local_forces = np.zeros((4, 3), dtype=np.float32)
        local_forces[:, 2] = forces
        self.rotor_view.apply_forces_and_torques_at_pos(forces=local_forces, is_global=False)
        reaction_world = rotation_matrix(self.orientation) @ np.array([0., 0., moments.sum()])
        self.base_view.apply_forces_and_torques_at_pos(
            torques=reaction_world.astype(np.float32).reshape(1, 3), is_global=True)
        self.drone.set_joint_velocities(self.actuator.rotor_speeds.astype(np.float32),
                                        joint_indices=self.rotor_joint_indices)
