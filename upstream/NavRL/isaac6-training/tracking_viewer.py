"""Overview, B's actual RGB pixels, and clearly labelled evaluation metrics."""

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def annotated_rgb(rgb, detection, mode):
    if rgb is None:
        rgb = np.zeros((360, 640, 3), dtype=np.uint8)
    image = Image.fromarray(rgb)
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, image.width, 25), fill=(15, 20, 30))
    draw.text((10, 7), f"B RGB camera | {mode} | magenta = A", fill=(255, 255, 255))
    if detection is not None:
        draw.rectangle(detection.bbox, outline=(40, 255, 80), width=2)
        x, y, _, _ = detection.bbox
        draw.text((max(0, x - 20), max(30, y - 18)),
                  f"visual range {detection.range_m:.1f} m", fill=(40, 255, 80))
    return image


class TrackingViewer:
    def __init__(self, scene):
        import omni.ui as ui
        from isaacsim.core.utils.viewports import set_camera_view
        from pxr import Gf, UsdGeom

        self.scene = scene
        self.set_camera_view = set_camera_view
        self.provider = ui.ByteImageProvider()
        self.window = ui.Window("A NavRL / B visual search & follow", width=480, height=405,
                                position_x=950, position_y=70)
        with self.window.frame:
            with ui.VStack(spacing=5):
                ui.Label("A: NavRL random goals | B: RGB marker search / tracking", height=24)
                ui.ImageWithProvider(self.provider, height=270)
                self.status = ui.Label("Starting RGB camera...", height=85, word_wrap=True)
        marker = UsdGeom.Sphere.Define(scene.world.stage, "/World/TrackingGoal")
        marker.CreateRadiusAttr(0.12)
        marker.CreateDisplayColorAttr([Gf.Vec3f(0.1, 1., 0.2)])
        self.goal_translate = UsdGeom.Xformable(marker).AddTranslateOp()
        self.set_camera_view(eye=np.array([1.5, -23., 9.]), target=np.array([1.5, 0., 2.]))

    def update(self, rgb, detection, tracker, now):
        from pxr import Gf

        scene = self.scene
        self.last_rgb = annotated_rgb(rgb, detection, tracker.mode)
        rgba = np.asarray(self.last_rgb.convert("RGBA"))
        self.provider.set_bytes_data(list(rgba.tobytes()), [rgba.shape[1], rgba.shape[0]])
        a, b = scene.leader, scene.follower
        distance = np.linalg.norm(a.position - b.position)
        self.status.text = (
            f"t={now:.1f} s | {tracker.mode} | equal speed limit={scene.speed_limit:.1f} m/s\n"
            f"Actual speed A={np.linalg.norm(a.velocity):.2f}, B={np.linalg.norm(b.velocity):.2f} m/s\n"
            f"Evaluation only: separation={distance:.2f} m | A goals reached={scene.waypoints_reached}")
        self.goal_translate.Set(Gf.Vec3d(*map(float, a.target)))
        # No debug draw: its global overlays would contaminate B's RGB sensor.

    def record(self, directory, frame, tracker):
        import asyncio
        import time
        from omni.kit.viewport.utility import capture_viewport_to_file, get_active_viewport

        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        overview = directory / f"overview_{frame:06d}.png"
        overview.unlink(missing_ok=True)
        capture = capture_viewport_to_file(get_active_viewport(), str(overview))
        future = asyncio.ensure_future(capture.wait_for_result(completion_frames=30))
        for _ in range(180):
            self.scene.world.render()
            if future.done() and overview.is_file():
                future.result()
                break
            time.sleep(0.002)  # Let the asynchronous image writer finish.
        else:
            raise RuntimeError("tracking viewport capture timed out")
        with Image.open(overview) as source:
            source = source.convert("RGB")
            source.thumbnail((960, 540))
            canvas = Image.new("RGB", (1440, 540), (16, 22, 30))
            canvas.paste(source, (0, 0))
        camera = self.last_rgb.resize((480, 270))
        canvas.paste(camera, (960, 0))
        draw = ImageDraw.Draw(canvas)
        a, b = self.scene.leader, self.scene.follower
        lines = ["TWO DRONES / FULL ROTOR DYNAMICS", "", f"B mode: {tracker.mode}",
                 f"Time: {self.scene.steps * self.scene.dt:.1f} s",
                 f"A speed: {np.linalg.norm(a.velocity):.2f} m/s",
                 f"B speed: {np.linalg.norm(b.velocity):.2f} m/s",
                 f"Both limits: {self.scene.speed_limit:.1f} m/s", "",
                 f"Separation (evaluation): {np.linalg.norm(a.position - b.position):.2f} m",
                 f"A goals reached: {self.scene.waypoints_reached}"]
        for i, line in enumerate(lines):
            draw.text((978, 288 + i * 20), line, fill=(220, 230, 240))
        canvas.save(directory / f"frame_{frame:06d}.png")
        self.last_rgb.save(directory / f"rgb_{frame:06d}.png")
