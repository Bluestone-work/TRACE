"""An industrial flight-test hall built with native USD/PBR scene primitives.

Background walls and props have static colliders. Floor markings and lamps are
visual decoration. The NavRL actor and the original obstacle dimensions are
not changed by the presentation style.
"""

from pathlib import Path

import numpy as np


def prepare_textures():
    root = Path(__file__).resolve().parent / "assets"
    root.mkdir(exist_ok=True)
    diffuse, roughness = root / "concrete_color.png", root / "concrete_roughness.png"
    if diffuse.exists() and roughness.exists():
        return diffuse, roughness
    from PIL import Image, ImageFilter
    rng = np.random.default_rng(42)
    size = 1024
    field = np.zeros((size, size), dtype=np.float32)
    for scale, weight in ((8, 15), (32, 8), (128, 4), (1024, 2)):
        noise = rng.integers(0, 256, (scale, scale), dtype=np.uint8)
        expanded = Image.fromarray(noise).resize((size, size), Image.Resampling.BICUBIC)
        field += (np.asarray(expanded).astype(np.float32) / 255 - 0.5) * weight
    # Fine aggregate, pores and saw-cut joints in a two-metre slab.
    field[rng.random((size, size)) < 0.009] -= 16
    field[:3, :] -= 24
    field[:, :3] -= 24
    base = np.array([133, 132, 124], dtype=np.float32)
    rgb = np.clip(base + field[..., None], 0, 255).astype(np.uint8)
    Image.fromarray(rgb).save(diffuse)
    Image.fromarray(np.clip(183 + field * 1.5, 0, 255).astype(np.uint8)).filter(
        ImageFilter.GaussianBlur(0.5)
    ).save(roughness)
    return diffuse, roughness


def build_hall(env):
    from pxr import Gf, Sdf, UsdGeom, UsdLux, UsdPhysics, UsdShade

    stage = env.world.stage
    root = "/World/Presentation"
    UsdGeom.Xform.Define(stage, root)
    diffuse, rough = prepare_textures()

    def material(name, color, roughness=0.6, metal=0.0, emission=None):
        path = f"{root}/Materials/{name}"
        mat = UsdShade.Material.Define(stage, path)
        shader = UsdShade.Shader.Define(stage, path + "/Surface")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
        shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metal)
        if emission:
            shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*emission))
        mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        return mat, shader

    floor, shader = material("Concrete", (0.45, 0.44, 0.41), 0.78)
    reader = UsdShade.Shader.Define(stage, str(floor.GetPath()) + "/UV")
    reader.CreateIdAttr("UsdPrimvarReader_float2")
    reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
    reader.CreateOutput("result", Sdf.ValueTypeNames.Float2)
    for name, file, target, output, space in (
        ("Color", diffuse, "diffuseColor", "rgb", "sRGB"),
        ("Roughness", rough, "roughness", "r", "raw"),
    ):
        texture = UsdShade.Shader.Define(stage, str(floor.GetPath()) + "/" + name)
        texture.CreateIdAttr("UsdUVTexture")
        texture.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(str(file)))
        texture.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set(space)
        texture.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
        texture.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("repeat")
        texture.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(reader.ConnectableAPI(), "result")
        texture.CreateOutput(output, Sdf.ValueTypeNames.Float3 if output == "rgb" else Sdf.ValueTypeNames.Float)
        shader.GetInput(target).ConnectToSource(texture.ConnectableAPI(), output)

    steel, _ = material("PaintedSteel", (0.085, 0.115, 0.14), 0.3, 0.65)
    panel, _ = material("WallPanels", (0.51, 0.55, 0.56), 0.55, 0.2)
    wood, _ = material("Plywood", (0.27, 0.15, 0.065), 0.85)
    yellow, _ = material("SafetyYellow", (0.68, 0.44, 0.035), 0.65)
    black, _ = material("Rubber", (0.018, 0.022, 0.025), 0.8)
    white, _ = material("Paint", (0.72, 0.72, 0.66), 0.7)
    concrete, _ = material("PillarConcrete", (0.37, 0.36, 0.32), 0.92)
    luminous, _ = material("LampDiffuser", (0.8, 0.85, 0.88), 0.3,
                           emission=(8.0, 8.4, 9.0))

    def box(name, position, scale, mat, collision=False):
        name = name.replace("-", "m").replace(".", "_")
        obj = UsdGeom.Cube.Define(stage, f"{root}/{name}")
        obj.CreateSizeAttr(1.0)
        xform = UsdGeom.Xformable(obj)
        xform.AddTranslateOp().Set(Gf.Vec3d(*position))
        xform.AddScaleOp().Set(Gf.Vec3f(*scale))
        UsdShade.MaterialBindingAPI.Apply(obj.GetPrim()).Bind(mat)
        if collision:
            UsdPhysics.CollisionAPI.Apply(obj.GetPrim())
        return obj

    # A textured mesh above the existing physical floor; UVs tile every 2 m.
    mesh = UsdGeom.Mesh.Define(stage, root + "/ConcreteFloor")
    mesh.CreatePointsAttr([(-10, -12, 0.003), (18, -12, 0.003), (18, 8, 0.003), (-10, 8, 0.003)])
    mesh.CreateFaceVertexCountsAttr([4])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    mesh.CreateSubdivisionSchemeAttr("none")
    UsdGeom.PrimvarsAPI(mesh).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray,
        UsdGeom.Tokens.faceVarying).Set([(0, 0), (14, 0), (14, 10), (0, 10)])
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(floor)
    UsdShade.MaterialBindingAPI.Apply(env.obstacle.prim).Bind(concrete)

    box("BackWall", (4, 7, 3.8), (28, 0.18, 7.6), panel, True)
    box("LeftWall", (-10, -2, 3.8), (0.18, 18, 7.6), panel, True)
    box("RightWall", (18, -2, 3.8), (0.18, 18, 7.6), panel, True)
    for i, x in enumerate(range(-8, 18, 4)):
        box(f"Column_{i}", (x, 6.7, 3.8), (0.24, 0.3, 7.6), steel, True)
        box(f"RoofBeam_{i}", (x, -2, 7.2), (0.17, 18, 0.32), steel)
        for z in (0.15, 3.8, 7.2):
            box(f"WallRail_{i}_{int(z*100)}", (x + 1.8, 6.84, z), (3.7, 0.1, 0.1), steel)
    # Rack shelves and crates are outside the original flight corridor.
    for rack, x in enumerate((-4, 5, 12)):
        for side in (-1, 1):
            box(f"Rack_{rack}_Post_{side}", (x + side * 1.5, 5.75, 1.8), (0.09, 0.7, 3.6), steel, True)
        for level, z in enumerate((0.2, 1.7, 3.2)):
            box(f"Rack_{rack}_Shelf_{level}", (x, 5.75, z), (3.1, 0.95, 0.1), yellow, True)
            for j in range(3):
                cx, cz = x - 1 + j, z + 0.47
                box(f"Crate_{rack}_{level}_{j}", (cx, 5.75, cz), (0.85, 0.72, 0.83), wood, True)
                for offset in (-0.32, 0.32):
                    box(f"CrateStrap_{rack}_{level}_{j}_{offset}",
                        (cx + offset, 5.382, cz), (0.045, 0.015, 0.84), steel)

    # Safety paint is flat geometry on top of the existing floor.
    for y in (-2.5, 2.5):
        box(f"Lane_{str(y).replace('.', '_')}", (4, y, 0.008), (18, 0.08, 0.008), yellow)
    for i in range(8):
        box(f"PillarBand_{i}", (2, -0.508, 0.15 + i * 0.14), (1.01, 0.012, 0.14), yellow if i % 2 == 0 else black)
    box("PillarBase", (2, 0, 0.06), (1.15, 1.15, 0.12), black)

    # Ground landing target. The original 3D hover target stays at z=2 m.
    for i in range(64):
        angle = i * 2 * np.pi / 64
        nxt = (i + 1) * 2 * np.pi / 64
        points = [(8 + r * np.cos(a), r * np.sin(a), 0.012)
                  for r, a in ((0.8, angle), (0.8, nxt), (0.72, nxt), (0.72, angle))]
        segment = UsdGeom.Mesh.Define(stage, f"{root}/LandingRing_{i}")
        segment.CreatePointsAttr(points)
        segment.CreateFaceVertexCountsAttr([4])
        segment.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
        segment.CreateSubdivisionSchemeAttr("none")
        UsdShade.MaterialBindingAPI.Apply(segment.GetPrim()).Bind(white)
    for x in (7.75, 8.25):
        box(f"LandingH_{str(x).replace('.', '_')}", (x, 0, 0.013), (0.07, 0.75, 0.01), white)
    box("LandingHMiddle", (8, 0, 0.013), (0.5, 0.07, 0.01), white)

    for i, x in enumerate((-3, 3, 9, 15)):
        box(f"LightHousing_{i}", (x, 1, 6.2), (2.2, 0.4, 0.12), steel)
        box(f"LightDiffuser_{i}", (x, 1, 6.13), (2.05, 0.32, 0.03), luminous)
        light = UsdLux.RectLight.Define(stage, f"{root}/CeilingLight_{i}")
        UsdGeom.Xformable(light).AddTranslateOp().Set(Gf.Vec3d(x, 1, 6.1))
        light.CreateWidthAttr(2.0)
        light.CreateHeightAttr(0.3)
        light.CreateIntensityAttr(1000.0)
        light.CreateColorAttr(Gf.Vec3f(0.9, 0.95, 1.0))
    dome = UsdLux.DomeLight.Define(stage, "/World/ViewerLight")
    dome.CreateIntensityAttr(650.0)
    studio = list(Path("/home/wj/isaacsim/extscache").glob(
        "omni.kit.widget.material_preview-*/data/domeLight/photo_studio_01_4k.hdr"))
    if studio:
        dome.CreateTextureFileAttr(Sdf.AssetPath(str(studio[0])))
        dome.CreateTextureFormatAttr("latlong")
    sun = UsdLux.DistantLight.Define(stage, root + "/Sun")
    sun.CreateIntensityAttr(2300.0)
    sun.CreateAngleAttr(1.2)
    sun.CreateColorAttr(Gf.Vec3f(1.0, 0.91, 0.78))
    UsdGeom.Xformable(sun).AddRotateXYZOp().Set(Gf.Vec3f(-32, -25, -30))
    print("ISAAC6_REALISTIC_SCENE_READY", flush=True)
