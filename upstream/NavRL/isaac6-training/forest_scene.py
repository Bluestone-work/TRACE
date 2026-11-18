"""Procedural forest presentation scene for the two-drone tracking demo.

The forest is made from native USD primitives so it does not require an external
asset pack. Tree trunks receive collision geometry for the LiDAR and flight
physics; foliage is visual only, which keeps the scene responsive while still
providing camera occlusion.
"""

from pathlib import Path

import numpy as np


def build_forest(env, seed: int = 17) -> None:
    """Add a deterministic forest around the existing NavRL ground plane."""
    from pxr import Gf, Sdf, UsdGeom, UsdLux, UsdPhysics, UsdShade

    stage = env.world.stage
    root = "/World/Forest"
    UsdGeom.Xform.Define(stage, root)
    rng = np.random.default_rng(seed)

    def material(name, color, roughness=0.8, emission=None):
        path = f"{root}/Materials/{name}"
        mat = UsdShade.Material.Define(stage, path)
        shader = UsdShade.Shader.Define(stage, path + "/Surface")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(float(roughness))
        if emission is not None:
            shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*emission))
        mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        return mat

    bark = material("Bark", (0.16, 0.075, 0.028), 0.96)
    bark_light = material("BarkLight", (0.27, 0.13, 0.045), 0.94)
    leaves = material("Leaves", (0.035, 0.19, 0.055), 0.92)
    leaves_light = material("LeavesLight", (0.10, 0.34, 0.08), 0.88)
    grass = material("ForestFloor", (0.10, 0.19, 0.07), 0.98)
    path_mat = material("DirtPath", (0.28, 0.18, 0.08), 0.99)

    def bind(obj, mat):
        UsdShade.MaterialBindingAPI.Apply(obj.GetPrim()).Bind(mat)

    def transform(obj, position, scale=None):
        xform = UsdGeom.Xformable(obj)
        xform.AddTranslateOp().Set(Gf.Vec3d(*map(float, position)))
        if scale is not None:
            xform.AddScaleOp().Set(Gf.Vec3f(*map(float, scale)))
        return xform

    def cylinder(name, position, radius, height, mat, collision=False):
        obj = UsdGeom.Cylinder.Define(stage, f"{root}/{name}")
        obj.CreateRadiusAttr(float(radius))
        obj.CreateHeightAttr(float(height))
        transform(obj, (position[0], position[1], position[2] + height * 0.5))
        bind(obj, mat)
        if collision:
            UsdPhysics.CollisionAPI.Apply(obj.GetPrim())
        return obj

    def sphere(name, position, radius, scale, mat):
        obj = UsdGeom.Sphere.Define(stage, f"{root}/{name}")
        obj.CreateRadiusAttr(float(radius))
        transform(obj, position, scale)
        bind(obj, mat)
        return obj

    # Visual grass floor over the existing physical ground.
    floor = UsdGeom.Cube.Define(stage, root + "/GrassFloor")
    floor.CreateSizeAttr(1.0)
    transform(floor, (4.0, -1.0, -0.015), (30.0, 22.0, 0.03))
    bind(floor, grass)

    # A winding dirt strip gives the camera a readable foreground and leaves a
    # few open flight corridors through the trees.
    for i, x in enumerate(np.linspace(-10.0, 18.0, 28)):
        y = -1.0 + 1.2 * np.sin((x + 3.0) * 0.32)
        patch = UsdGeom.Cube.Define(stage, f"{root}/Path_{i:02d}")
        patch.CreateSizeAttr(1.0)
        transform(patch, (x, y, 0.004), (1.25, 2.2, 0.008))
        bind(patch, path_mat)

    # Candidate positions use a jittered grid, which looks natural but avoids
    # accidental clumps that make the first tracking test impossible.
    candidates = []
    for x in np.arange(-10.0, 18.1, 3.2):
        for y in np.arange(-9.0, 9.1, 3.2):
            if rng.random() < 0.62:
                candidates.append((x + rng.uniform(-0.55, 0.55),
                                   y + rng.uniform(-0.55, 0.55)))
    rng.shuffle(candidates)

    kept = []
    protected = np.asarray([[0.0, 0.0], [-9.2, -3.9], [9.0, -1.0]], dtype=np.float32)

    def distance_to_segment(point, start, end):
        segment = end - start
        alpha = np.dot(point - start, segment) / max(float(np.dot(segment, segment)), 1e-6)
        alpha = np.clip(alpha, 0.0, 1.0)
        return float(np.linalg.norm(point - (start + alpha * segment)))

    initial_a = protected[0]
    initial_b = protected[1]
    first_goal = protected[2]
    for x, y in candidates:
        p = np.array([x, y], dtype=np.float32)
        # Keep the initial launch area, B's initial position and the first A
        # goal clear. Other trees may still make later paths challenging.
        if np.min(np.linalg.norm(protected - p, axis=1)) < 1.8:
            continue
        # Keep the first visual acquisition corridor open. This makes the
        # forest a meaningful occlusion test after acquisition rather than a
        # permanently blind-start test.
        if (distance_to_segment(p, initial_b, initial_a) < 1.65
                or distance_to_segment(p, initial_a, first_goal) < 1.35):
            continue
        if any(np.linalg.norm(p - q) < 2.05 for q in kept):
            continue
        kept.append(p)
        if len(kept) >= 38:
            break

    for i, p in enumerate(kept):
        height = float(rng.uniform(4.2, 7.2))
        radius = float(rng.uniform(0.13, 0.24))
        cylinder(f"Trunk_{i:03d}", (p[0], p[1], 0.0), radius, height,
                 bark if i % 3 else bark_light, collision=True)
        crown_z = height * 0.70
        crown_scale = float(rng.uniform(1.0, 1.45))
        sphere(f"Crown_{i:03d}_0", (p[0], p[1], crown_z), 1.0,
               (crown_scale, crown_scale * 0.86, crown_scale * 1.15), leaves)
        sphere(f"Crown_{i:03d}_1", (p[0] - 0.35, p[1] + 0.12, crown_z - 0.35), 0.78,
               (crown_scale * 0.9, crown_scale * 0.8, crown_scale), leaves_light)
        if height > 5.4:
            sphere(f"Crown_{i:03d}_2", (p[0] + 0.28, p[1] - 0.10, height * 0.88), 0.68,
                   (crown_scale * 0.8, crown_scale * 0.72, crown_scale * 0.9), leaves)

    # Warm sun plus a cool low-intensity dome gives depth between trunks.
    dome = UsdLux.DomeLight.Define(stage, root + "/ForestDome")
    dome.CreateIntensityAttr(420.0)
    dome.CreateColorAttr(Gf.Vec3f(0.58, 0.70, 0.82))
    sun = UsdLux.DistantLight.Define(stage, root + "/ForestSun")
    sun.CreateIntensityAttr(1800.0)
    sun.CreateAngleAttr(1.8)
    sun.CreateColorAttr(Gf.Vec3f(1.0, 0.78, 0.52))
    UsdGeom.Xformable(sun).AddRotateXYZOp().Set(Gf.Vec3f(-38.0, -28.0, -25.0))

    print(f"ISAAC6_FOREST_SCENE_READY trees={len(kept)}", flush=True)
