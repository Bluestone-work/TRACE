"""Check pixel-based range/bearing, missing detections and speed fairness."""

import json
from pathlib import Path

import numpy as np

from visual_tracker import VisualSearchTracker, avoid_obstacles, detect_marker, limit_speed


def main():
    image = np.zeros((360, 640, 3), dtype=np.uint8)
    yy, xx = np.indices(image.shape[:2])
    radius_pixels = 320 * 0.18 / 10.
    image[(xx - 319.5)**2 + (yy - 179.5)**2 <= radius_pixels**2] = [240, 0, 220]
    own = np.array([-5., -3., 2.])
    camera = own + [0., 0., 0.25]
    detection = detect_marker(image, camera, 0.)
    assert detection is not None
    assert abs(detection.range_m - 10.) < 0.7, detection
    assert np.linalg.norm(detection.position - (own + [10., 0., 0.])) < 0.7
    assert detect_marker(np.zeros_like(image), camera, 0.) is None
    tracker = VisualSearchTracker(own)
    for i in range(20):
        command = tracker.update(None, own, i * 0.1, 0.1)
        assert tracker.mode == "SEARCH" and np.linalg.norm(command) < 1e-8
    tracker.update(detection, own, 2., 0.1)
    assert tracker.mode == "TRACK"
    tracker.update(None, own, 2.1, 0.1)
    assert tracker.mode == "COAST"
    tracker.update(None, own, 3.3, 0.1)
    assert tracker.mode == "SEARCH"
    tracker.update(detection, own, 3.4, 0.1)
    assert tracker.mode == "TRACK" and tracker.acquisitions == 2
    rng = np.random.default_rng(1)
    for velocity in rng.normal(0, 5, size=(100, 3)):
        assert np.linalg.norm(limit_speed(velocity, 2.)) <= 2. + 1e-12
    lidar = np.zeros((36, 4))
    lidar[0] = 3.4  # Obstacle 0.6 m directly in front.
    avoided = avoid_obstacles(np.array([2., 0., 0.]), lidar, 0.)
    assert avoided[0] <= 0. and np.linalg.norm(avoided) <= 2. + 1e-12
    report = {"status": "passed", "synthetic_visual_range_m": detection.range_m,
              "blank_rgb_no_detection": True, "search_coast_reacquire": True,
              "norm_speed_limit": True, "lidar_close_obstacle": True}
    output = Path(__file__).resolve().parent / "artifacts/visual_tracker_checks.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
