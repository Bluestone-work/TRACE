# MIT License
# 
# Copyright (c) 2023 Botian Xu, Tsinghua University
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""NumPy port of OmniDrones' Lee velocity controller and RotorGroup.

Controller/rotor equations derive from the MIT-licensed OmniDrones sources by
Botian Xu (2023); see ../isaac-training/third_party/OmniDrones/LICENSE.
The simulator integrates the rigid-body dynamics; this module only computes
actuator forces from measured state and a world-frame velocity setpoint.
"""

from pathlib import Path

import numpy as np
import yaml


def rotation_matrix(quaternion):
    """Body-to-world rotation, scalar-first quaternion (w, x, y, z)."""
    q = np.asarray(quaternion, dtype=np.float64)
    q = q / np.linalg.norm(q)
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
        [2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)],
    ])


class QuadrotorActuator:
    """Lee control, plus-layout allocation, actuator saturation and motor lag."""

    def __init__(self, physics_dt):
        package = (Path(__file__).resolve().parents[1]
                   / "isaac-training/third_party/OmniDrones/omni_drones")
        self.params = yaml.safe_load((package / "robots/assets/usd/hummingbird.yaml").read_text())
        gains = yaml.safe_load((package / "controllers/cfg/lee_controller_hummingbird.yaml").read_text())
        self.mass = float(self.params["mass"])
        self.inertia = np.array([self.params["inertia"][k] for k in ("xx", "yy", "zz")])
        self.velocity_gain = np.array(gains["velocity_gain"], dtype=np.float64)
        self.attitude_gain = np.array(gains["attitude_gain"], dtype=np.float64)
        self.rate_gain = np.array(gains["angular_rate_gain"], dtype=np.float64)
        rotor = self.params["rotor_configuration"]
        self.kf = np.array(rotor["force_constants"])
        self.km = np.array(rotor["moment_constants"])
        self.omega_max = np.array(rotor["max_rotation_velocities"])
        self.directions = np.array(rotor["directions"])
        angles, arms = np.array(rotor["rotor_angles"]), np.array(rotor["arm_lengths"])
        # Wrench order: body torque x/y/z [Nm], total thrust [N].
        self.allocation = np.vstack([
            np.sin(angles) * arms, -np.cos(angles) * arms,
            -self.directions * self.km / self.kf, np.ones(4),
        ])
        self.max_forces = self.kf * self.omega_max**2
        # Upstream 0.43 is a per-step interpolation factor, NOT 0.43 seconds.
        # Preserve its effective response at the upstream 16 ms reference dt.
        self.motor_alpha = 1.0 - (1.0 - 0.43)**(physics_dt / 0.016)
        self.throttle = np.zeros(4)
        self.requested_forces = np.zeros(4)
        self.forces = np.zeros(4)
        self.moments = np.zeros(4)
        self.saturated = False

    def reset(self):
        self.throttle[:] = np.sqrt(self.mass * 9.81 / self.max_forces.sum())
        self.requested_forces[:] = self.mass * 9.81 / 4
        self.forces[:] = self.requested_forces
        self.moments[:] = -self.directions * (self.km / self.kf) * self.forces

    def command(self, quaternion, velocity_world, angular_velocity_world, target_velocity):
        R = rotation_matrix(quaternion)
        # Original velocity mode sets target_pos=current_pos and target_acc=0.
        desired_acc = self.velocity_gain * (np.asarray(target_velocity) - velocity_world)
        desired_acc += np.array([0.0, 0.0, 9.81])
        b3 = desired_acc / max(np.linalg.norm(desired_acc), 1e-9)
        yaw = np.arctan2(R[1, 0], R[0, 0])  # same yaw_control=False semantics
        b1_yaw = np.array([np.cos(yaw), np.sin(yaw), 0.0])
        b2 = np.cross(b3, b1_yaw)
        b2 /= max(np.linalg.norm(b2), 1e-9)
        desired_R = np.column_stack([np.cross(b2, b3), b2, b3])
        error_matrix = 0.5 * (desired_R.T @ R - R.T @ desired_R)
        attitude_error = error_matrix[[2, 0, 1], [1, 2, 0]]
        body_rate = R.T @ np.asarray(angular_velocity_world)
        torque = -self.attitude_gain * attitude_error - self.rate_gain * body_rate
        thrust = self.mass * np.dot(desired_acc, R[:, 2])
        self.requested_forces = np.linalg.solve(self.allocation, np.r_[torque, thrust])
        self.saturated = bool(np.any(self.requested_forces < 0)
                              or np.any(self.requested_forces > self.max_forces))
        return np.clip(self.requested_forces, 0.0, self.max_forces)

    def motor_step(self, target_forces):
        target_throttle = np.sqrt(np.clip(target_forces / self.max_forces, 0.0, 1.0))
        self.throttle += self.motor_alpha * (target_throttle - self.throttle)
        self.forces = self.max_forces * self.throttle**2
        self.moments = -self.directions * self.km * (self.omega_max * self.throttle)**2
        return self.forces, self.moments

    @property
    def rotor_speeds(self):
        return self.throttle * self.omega_max * self.directions
