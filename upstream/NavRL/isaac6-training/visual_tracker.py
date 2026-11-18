"""RGB-only target measurements and a local search/track baseline.

This module deliberately has no simulator/stage/target-state imports. The only
external pose it receives is the observer's own camera and vehicle pose.
"""

from dataclasses import dataclass

import numpy as np


def limit_speed(velocity, maximum=2.0):
    velocity = np.asarray(velocity, dtype=np.float64)
    if not np.isfinite(velocity).all():
        raise ValueError("nonfinite velocity command")
    return velocity * min(1.0, maximum / max(float(np.linalg.norm(velocity)), 1e-9))


def wrap_angle(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


@dataclass
class VisualDetection:
    bbox: tuple[int, int, int, int]
    position: np.ndarray
    range_m: float
    pixels: int


def detect_marker(rgb, camera_position, camera_yaw, horizontal_fov=90.0, radius=0.18):
    """Estimate bearing and range from a known-size magenta sphere in RGB.

    There is no depth, segmentation-ID or privileged target pose input.
    Range uses the angular diameter of the known marker (monocular scale cue).
    """
    if rgb is None or np.asarray(rgb).ndim != 3:
        return None
    rgb = np.asarray(rgb)[..., :3].astype(np.float32)
    red, green, blue = np.moveaxis(rgb, -1, 0)
    mask = (red > 115) & (blue > 85) & (green < red * 0.62) & (green < blue * 0.75)
    ys, xs = np.nonzero(mask)
    if len(xs) < 8:
        return None
    x0, x1, y0, y1 = int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())
    width, height = x1 - x0 + 1, y1 - y0 + 1
    if width < 3 or height < 3 or not 0.4 < width / height < 2.5:
        return None
    # A clipped target has unreliable size/range; reacquire it during the sweep.
    h, w = rgb.shape[:2]
    if x0 == 0 or y0 == 0 or x1 == w - 1 or y1 == h - 1:
        return None
    focal = 0.5 * w / np.tan(np.deg2rad(horizontal_fov) / 2)
    cx, cy = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
    right = (cx - (w - 1) / 2) / focal
    up = ((h - 1) / 2 - cy) / focal
    left_ray = np.array([1., (x0 - 0.5 - (w - 1) / 2) / focal, up])
    right_ray = np.array([1., (x1 + 0.5 - (w - 1) / 2) / focal, up])
    cosine = np.dot(left_ray, right_ray) / (np.linalg.norm(left_ray) * np.linalg.norm(right_ray))
    half_angle = 0.5 * np.arccos(np.clip(cosine, -1., 1.))
    distance = radius / max(np.sin(half_angle), 1e-5)
    if not 0.5 < distance < 35.0:
        return None
    forward = np.array([np.cos(camera_yaw), np.sin(camera_yaw), 0.])
    camera_right = np.array([np.sin(camera_yaw), -np.cos(camera_yaw), 0.])
    direction = forward + right * camera_right + np.array([0., 0., up])
    direction /= np.linalg.norm(direction)
    position = np.asarray(camera_position) + distance * direction - np.array([0., 0., 0.25])
    return VisualDetection((x0, y0, x1, y1), position, float(distance), len(xs))


class VisualSearchTracker:
    """Gimballed-camera sweep, alpha-beta motion filter and standoff pursuit."""

    def __init__(self, own_position, speed_limit=2.0, follow_distance=3.5):
        self.speed_limit = speed_limit
        self.follow_distance = follow_distance
        self.camera_yaw = np.pi  # Deliberately starts facing away from the arena.
        self.mode = "SEARCH"
        self.anchor = np.asarray(own_position, dtype=np.float64).copy()
        self.estimate = None
        self.target_velocity = np.zeros(3)
        self.last_seen = -np.inf
        self.first_seen = None
        self.acquisitions = 0
        self.search_started = 0.
        self.search_center = self.camera_yaw

    def update(self, detection, own_position, now, dt):
        own = np.asarray(own_position)
        if self.estimate is not None and now - self.last_seen <= 1.2:
            self.estimate += self.target_velocity * dt
        if detection is not None:
            gap = now - self.last_seen
            if gap > 1.2 or self.estimate is None:
                self.estimate = detection.position.copy()
                self.target_velocity[:] = 0.
                self.acquisitions += 1
                if self.first_seen is None:
                    self.first_seen = now
            else:
                residual = detection.position - self.estimate
                self.estimate += 0.45 * residual
                self.target_velocity = limit_speed(self.target_velocity + 0.06 / dt * residual,
                                                   self.speed_limit)
            self.last_seen = now
            self.mode = "TRACK"
        elif now - self.last_seen <= 1.2:
            self.mode = "COAST"
        else:
            if self.mode != "SEARCH":
                self.anchor = own.copy()
                self.search_started = now
                self.search_center = self.camera_yaw
            self.mode = "SEARCH"

        if self.mode == "SEARCH":
            elapsed = now - self.search_started
            if self.first_seen is not None and elapsed < 3.:
                # Search near the last bearing first; global sweep follows.
                desired = self.search_center + 1.1 * np.sin(2.1 * elapsed)
                self.camera_yaw = wrap_angle(self.camera_yaw + np.clip(
                    wrap_angle(desired - self.camera_yaw), -1.8 * dt, 1.8 * dt))
            else:
                self.camera_yaw = wrap_angle(self.camera_yaw + (0.8 if self.first_seen is None else 1.2) * dt)
            command = 0.8 * (self.anchor - own)
        else:
            relative = self.estimate - own
            distance = np.linalg.norm(relative[:2])
            direction = relative / max(distance, 0.1)
            position_error = relative - self.follow_distance * direction
            position_error[2] = relative[2]
            command = self.target_velocity + 0.9 * position_error
            desired_yaw = np.arctan2(relative[1], relative[0])
            self.camera_yaw = wrap_angle(self.camera_yaw + np.clip(
                wrap_angle(desired_yaw - self.camera_yaw), -1.8 * dt, 1.8 * dt))
        return limit_speed(command, self.speed_limit)


def avoid_obstacles(command, lidar, yaw, speed_limit=2.0):
    """Local horizontal LiDAR repulsion; target coordinates are not needed."""
    distances = 4.0 - np.asarray(lidar).reshape(36, 4)
    nearest = distances[:, :3].min(axis=1)
    angles = yaw + np.arange(len(nearest)) * 2 * np.pi / len(nearest)
    directions = np.column_stack([np.cos(angles), np.sin(angles), np.zeros(len(angles))])
    weights = np.maximum(1.7 - nearest, 0.) ** 2
    result = np.asarray(command).copy() - 1.4 * (weights[:, None] * directions).sum(axis=0)
    for distance, direction in zip(nearest, directions):
        if distance < 0.9:
            approach = np.dot(result, direction)
            if approach > 0:
                result -= approach * direction
    return limit_speed(result, speed_limit)
