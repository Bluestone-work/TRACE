"""用 Blender(bpy) 把 dump_scene.py 导出的 .npz 渲染成 3D 图。

在 trace_render 环境里跑（不是 trace_isaac）:
    conda activate trace_render
    python trace/viz/render3d.py snapshot.npz out.png [--samples 64] [--res 1600 1200]

渲染 Cycles CPU（24 核约数秒）。不依赖 Vulkan/RTX，所以不受本机 RTX 渲染器故障影响。
"""

from __future__ import annotations

import argparse
import sys

import bpy
import numpy as np

# 配色与 2D 俯视图保持一致
C_OCC = (0.12, 0.12, 0.14, 1.0)
C_FRONTIER = (1.00, 0.45, 0.05, 1.0)
C_TASK = (0.84, 0.15, 0.16, 1.0)
C_UAV = (0.12, 0.47, 0.71, 1.0)
C_TRAIL = (0.20, 0.55, 0.85, 1.0)
C_GROUND = (0.88, 0.88, 0.90, 1.0)


def _mat(name, rgba, emission=0.0, alpha=1.0):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    bsdf = m.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = rgba
    bsdf.inputs["Roughness"].default_value = 0.55
    if "Alpha" in bsdf.inputs:
        bsdf.inputs["Alpha"].default_value = alpha
    if emission > 0 and "Emission Strength" in bsdf.inputs:
        bsdf.inputs["Emission Color"].default_value = rgba
        bsdf.inputs["Emission Strength"].default_value = emission
    if alpha < 1.0:
        m.blend_method = 'BLEND'
    return m


def _cubes_mesh(name, centers: np.ndarray, size: float, mat):
    """把一堆体素中心合成单个网格（逐个建 object 会慢几个数量级）。"""
    if len(centers) == 0:
        return None
    h = size * 0.5
    unit = np.array([[-h, -h, -h], [h, -h, -h], [h, h, -h], [-h, h, -h],
                     [-h, -h, h], [h, -h, h], [h, h, h], [-h, h, h]], dtype=np.float32)
    faces_u = [(0, 1, 2, 3), (4, 7, 6, 5), (0, 4, 5, 1),
               (1, 5, 6, 2), (2, 6, 7, 3), (3, 7, 4, 0)]
    n = len(centers)
    verts = (centers[:, None, :] + unit[None, :, :]).reshape(-1, 3)
    offs = (np.arange(n) * 8)[:, None]
    faces = np.concatenate([np.asarray(faces_u, dtype=np.int64)[None] + o for o in offs], axis=0)

    me = bpy.data.meshes.new(name)
    me.from_pydata(verts.tolist(), [], faces.reshape(-1, 4).tolist())
    me.validate()
    ob = bpy.data.objects.new(name, me)
    ob.data.materials.append(mat)
    bpy.context.scene.collection.objects.link(ob)
    return ob


def _tube(name, pts: np.ndarray, radius: float, mat):
    """用折线 + 圆形倒角做轨迹管。"""
    if len(pts) < 2:
        return None
    cu = bpy.data.curves.new(name, type='CURVE')
    cu.dimensions = '3D'
    sp = cu.splines.new('POLY')
    sp.points.add(len(pts) - 1)
    for i, p in enumerate(pts):
        sp.points[i].co = (float(p[0]), float(p[1]), float(p[2]), 1.0)
    cu.bevel_depth = radius
    cu.bevel_resolution = 3
    ob = bpy.data.objects.new(name, cu)
    ob.data.materials.append(mat)
    bpy.context.scene.collection.objects.link(ob)
    return ob


def build_scene(d, show_frontier=True, frontier_max=8000):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    sc = bpy.context.scene
    origin = d["origin"]
    res = float(d["res"])
    dims = d["dims"]
    span = dims.astype(np.float32) * res
    cx, cy = origin[0] + span[0] / 2, origin[1] + span[1] / 2

    m_occ = _mat("occ", C_OCC)
    m_fr = _mat("frontier", C_FRONTIER, emission=0.15, alpha=0.30)
    m_task = _mat("task", C_TASK, emission=1.2)
    m_uav = _mat("uav", C_UAV, emission=0.6)
    m_trail = _mat("trail", C_TRAIL, emission=0.3)
    m_gnd = _mat("ground", C_GROUND)

    # 地面
    bpy.ops.mesh.primitive_plane_add(size=1, location=(cx, cy, origin[2] - 0.02))
    gp = bpy.context.active_object
    gp.scale = (span[0], span[1], 1)
    gp.data.materials.append(m_gnd)

    def centers(idx):
        return origin + (idx.astype(np.float32) + 0.5) * res

    _cubes_mesh("occupied", centers(d["occupied"]), res * 0.95, m_occ)
    if show_frontier and len(d["frontier"]):
        fr = d["frontier"]
        if len(fr) > frontier_max:          # 太密会盖住内部结构
            sel = np.random.default_rng(0).choice(len(fr), frontier_max, replace=False)
            fr = fr[sel]
        _cubes_mesh("frontier", centers(fr), res * 0.7, m_fr)

    # task node：球体，半径随 n_unknown
    tp, tn = d["task_pos"], d["task_nunk"]
    if len(tp):
        scale = tn / max(float(tn.max()), 1e-6)
        for i, (p, s) in enumerate(zip(tp, scale)):
            r = 0.35 + 0.9 * float(s)
            bpy.ops.mesh.primitive_uv_sphere_add(radius=r, location=tuple(p), segments=20, ring_count=12)
            ob = bpy.context.active_object
            ob.name = f"task_{i}"
            ob.data.materials.append(m_task)

    # 轨迹 + 无人机
    for i, tr in enumerate(d["trails"]):
        _tube(f"trail_{i}", tr, res * 0.4, m_trail)
    for i, p in enumerate(d["drone_pos"]):
        bpy.ops.mesh.primitive_cone_add(radius1=0.9, depth=1.8, location=tuple(p),
                                        rotation=(0, 0, 0), vertices=20)
        ob = bpy.context.active_object
        ob.name = f"uav_{i}"
        ob.data.materials.append(m_uav)

    # 光照：太阳 + 环境
    sun = bpy.data.objects.new("sun", bpy.data.lights.new("sun", type='SUN'))
    sc.collection.objects.link(sun)
    sun.location = (cx + span[0], cy - span[1], 40)
    sun.rotation_euler = (0.5, 0.3, 0.8)
    sun.data.energy = 3.0
    sc.world = bpy.data.worlds.new("W")
    sc.world.use_nodes = True
    sc.world.node_tree.nodes["Background"].inputs[0].default_value = (0.9, 0.92, 0.95, 1)
    sc.world.node_tree.nodes["Background"].inputs[1].default_value = 0.6

    # 等轴测视角相机
    cam = bpy.data.objects.new("cam", bpy.data.cameras.new("cam"))
    sc.collection.objects.link(cam)
    diag = float(np.linalg.norm(span[:2]))
    cam.location = (cx - diag * 0.50, cy - diag * 0.50, diag * 0.40)
    cam.rotation_euler = (1.10, 0.0, -0.785)
    cam.data.lens = 42.0
    sc.camera = cam
    return sc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz")
    ap.add_argument("out")
    ap.add_argument("--samples", type=int, default=64)
    ap.add_argument("--res", type=int, nargs=2, default=[1600, 1200])
    ap.add_argument("--no-frontier", action="store_true")
    ap.add_argument("--frontier-max", type=int, default=8000, help="渲染的 frontier 体素上限")
    a = ap.parse_args()

    d = dict(np.load(a.npz))
    print(f"[OK] 载入 {a.npz}: occupied={len(d['occupied'])} frontier={len(d['frontier'])} "
          f"task={len(d['task_pos'])} known_ratio={float(d['known_ratio']):.4f}", flush=True)

    sc = build_scene(d, show_frontier=not a.no_frontier, frontier_max=a.frontier_max)
    sc.render.engine = 'CYCLES'
    sc.cycles.device = 'CPU'
    sc.cycles.samples = a.samples
    sc.cycles.use_denoising = True
    sc.render.resolution_x, sc.render.resolution_y = a.res
    sc.render.film_transparent = False
    sc.render.filepath = a.out
    sc.render.image_settings.file_format = 'PNG'

    import time
    t0 = time.time()
    bpy.ops.render.render(write_still=True)
    print(f"[OK] 渲染完成 {time.time()-t0:.1f}s -> {a.out}", flush=True)
    print("[DONE]", flush=True)


if __name__ == "__main__":
    main()
