"""Native Isaac Sim viewport overlays for the migrated NavRL environment."""

from collections import deque
from pathlib import Path

import numpy as np


class NavRLViewer:
    def __init__(self, env, style="debug", camera=None, show_sensors=False):
        import omni.ui as ui
        from isaacsim.core.utils.viewports import set_camera_view
        from isaacsim.util.debug_draw import _debug_draw
        from pxr import Gf, UsdGeom, UsdLux

        self.env = env
        self.style = style
        self.camera = camera or ("follow" if style == "realistic" else "overview")
        self.show_debug = style == "debug" or show_sensors
        self.set_camera_view = set_camera_view
        self.draw = _debug_draw.acquire_debug_draw_interface()
        self.trail = deque(maxlen=1800)
        stage = env.world.stage
        light = UsdLux.DomeLight.Define(stage, "/World/ViewerLight")
        light.CreateIntensityAttr(900.0)
        if style == "realistic":
            from realistic_scene import build_hall
            build_hall(env)
        else:
            marker = UsdGeom.Sphere.Define(stage, "/World/GoalMarker")
            marker.CreateRadiusAttr(0.22)
            marker.CreateDisplayColorAttr([Gf.Vec3f(0.1, 0.9, 0.3)])
            UsdGeom.Xformable(marker).AddTranslateOp().Set(Gf.Vec3d(*map(float, env.target)))
        set_camera_view(eye=np.array([8.5, -12.0, 10.0]), target=np.array([3.3, 0.0, 1.5]))
        self._update_camera()

        self.window = ui.Window("NavRL | Isaac Sim 6", width=370, height=250)
        with self.window.frame:
            with ui.VStack(spacing=8):
                ui.Label("NavRL checkpoint / mean action", height=24)
                if style == "debug":
                    ui.Label("Green: goal   Red: obstacle   Blue: trajectory", height=24)
                else:
                    ui.Label("Goal: hover 2 m above the H landing pad", height=24)
                with ui.HStack(height=24):
                    checkbox = ui.CheckBox(width=24)
                    checkbox.model.set_value(self.show_debug)
                    checkbox.model.add_value_changed_fn(self._toggle_debug)
                    ui.Label("Show LiDAR / trajectory")
                self.status = ui.Label("Starting...", word_wrap=True)
                ui.Label("Lee controller / rotor forces / PhysX" if env.dynamics == "rotor"
                         else "Kinematic velocity adapter", height=24)
        env.world.render()

    def _toggle_debug(self, model):
        self.show_debug = model.as_bool

    def _update_camera(self):
        if self.camera == "follow":
            pos = self.env.position.astype(np.float64)
            self.set_camera_view(eye=pos + np.array([2.3, -4.6, 1.8]),
                                 target=pos + np.array([0.8, 0.0, -0.1]))

    def update(self, obs, reward, episode):
        pos = self.env.position
        if self.env.step_count <= 1:
            self.trail.clear()
        self.trail.append(tuple(map(float, pos)))
        self.draw.clear_lines()
        self.draw.clear_points()
        if self.show_debug:
            self._draw_debug(obs, pos)
        distance = float(np.linalg.norm(self.env.target - pos))
        self.status.text = (
            f"Episode {episode}  |  Step {self.env.step_count}\n"
            f"Position: {pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f} m\n"
            f"Speed: {np.linalg.norm(self.env.velocity):.2f} m/s\n"
            f"Goal distance: {distance:.2f} m  |  Reward: {reward:.3f}"
        )
        if self.env.actuator is not None:
            from quadrotor_dynamics import rotation_matrix
            tilt = np.rad2deg(np.arccos(np.clip(rotation_matrix(self.env.orientation)[2, 2], -1, 1)))
            self.status.text += f"\nTilt: {tilt:.1f} deg | Thrust: {self.env.actuator.forces.sum():.2f} N"
        self._update_camera()
        self.env.world.render()

    def _draw_debug(self, obs, pos):
        trail = list(self.trail)
        if len(trail) > 1:
            n = len(trail) - 1
            self.draw.draw_lines(trail[:-1], trail[1:], [(0.05, 0.6, 1.0, 1.0)] * n, [4.0] * n)

        # These are the same world-frame rays and ranges used by the policy.
        ranges = self.env.lidar_range - obs["lidar"][0]
        horizontal = (np.arange(self.env.lidar_hbeams) * 2 * np.pi / self.env.lidar_hbeams
                      + self.env.lidar_yaw)
        pitches = np.deg2rad(np.linspace(*self.env.lidar_vfov, self.env.lidar_vbeams))
        directions = np.stack(np.broadcast_arrays(
            np.cos(horizontal[:, None]) * np.cos(pitches),
            np.sin(horizontal[:, None]) * np.cos(pitches),
            np.sin(pitches)[None, :],
        ), axis=-1)
        endpoints = pos + ranges[..., None] * directions
        # Show one horizontal ring, and all measured hit points.
        ends = [tuple(map(float, p)) for p in endpoints[:, 1]]
        self.draw.draw_lines([tuple(map(float, pos))] * len(ends), ends,
                             [(1.0, 0.8, 0.15, 0.45)] * len(ends), [1.0] * len(ends))
        hits = [tuple(map(float, p)) for p in endpoints[ranges < self.env.lidar_range - 0.01]]
        if hits:
            self.draw.draw_points(hits, [(1.0, 0.65, 0.1, 1.0)] * len(hits), [4.0] * len(hits))
        self.draw.draw_points([tuple(map(float, pos))], [(0.1, 0.7, 1.0, 1.0)], [10.0])

    def screenshot(self, path):
        import asyncio
        from omni.kit.viewport.utility import capture_viewport_to_file, get_active_viewport

        path = Path(path).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        capture = capture_viewport_to_file(get_active_viewport(), str(path))
        future = asyncio.ensure_future(capture.wait_for_result(completion_frames=30))
        for _ in range(120):
            self.env.world.render()
            if future.done():
                future.result()
                break
        else:
            raise RuntimeError(f"Viewport capture timed out: {path}")
        if not path.is_file():
            raise RuntimeError(f"Viewport capture was not written: {path}")
        print(f"ISAAC6_VIEWER_SCREENSHOT={path}", flush=True)
