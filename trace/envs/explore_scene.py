"""多机 3D 探索场景（静态障碍版）。

环境搭建参考 NavRL (isaac-training/training/scripts/env.py):
  - 静态障碍用 Isaac Lab 的 HfDiscreteObstaclesTerrainCfg 地形生成器
  - 障碍宽度 0.4~1.1 m、高度分档，与 NavRL 的取值一致
  - LiDAR 用 RayCaster + LidarPatternCfg，GPU 光线投射

尚未包含动态障碍。RayCaster 只打静态 mesh，动态障碍要换 MultiMeshRayCaster，
见 docs/ISAAC_3D_plan.md 的坑 6。
"""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import RayCasterCfg, patterns
from isaaclab.terrains import TerrainGeneratorCfg, TerrainImporterCfg
from isaaclab.terrains.height_field import HfDiscreteObstaclesTerrainCfg
from isaaclab.utils import configclass

# 场景平面尺寸（米），与体素地图的 size 保持一致
SCENE_SIZE = (40.0, 40.0)

EXPLORE_TERRAIN_CFG = TerrainGeneratorCfg(
    size=SCENE_SIZE,
    border_width=0.0,
    num_rows=1,
    num_cols=1,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    use_cache=False,
    sub_terrains={
        "obstacles": HfDiscreteObstaclesTerrainCfg(
            size=SCENE_SIZE,
            horizontal_scale=0.1,
            vertical_scale=0.005,
            border_width=0.0,
            num_obstacles=80,
            obstacle_height_mode="choice",
            obstacle_width_range=(0.4, 1.1),      # NavRL 取值
            obstacle_height_range=(1.0, 4.0),
            platform_width=2.0,                   # 中心留出起飞平台
        ),
    },
)


def lidar_cfg(prim_path: str, channels: int = 16, h_res: float = 2.0,
              v_fov: tuple[float, float] = (-30.0, 30.0),
              max_distance: float = 12.0) -> RayCasterCfg:
    """挂在 prim_path 上的 LiDAR。默认 16 线 x 180 方位 = 2880 根光线。"""
    return RayCasterCfg(
        prim_path=prim_path,
        ray_alignment="base",   # 旧的 attach_yaw_only=False 等价写法，已弃用
        pattern_cfg=patterns.LidarPatternCfg(
            channels=channels,
            vertical_fov_range=v_fov,
            horizontal_fov_range=(0.0, 360.0),
            horizontal_res=h_res,
        ),
        max_distance=max_distance,
        mesh_prim_paths=["/World/ground"],
        debug_vis=False,
    )


def drone_cfg(prim_path: str, pos: tuple[float, float, float]) -> RigidObjectCfg:
    """LiDAR 挂载体。先用"关重力的动态刚体 + 每步写位姿"占位，后续换真实四旋翼动力学。

    不要用 kinematic_enabled=True：给运动学刚体写位姿后，RayCaster 读取其
    PhysX 刚体 view 会触发 CUDA illegal memory access（见 docs/ISAAC_3D_plan.md 坑 7）。
    """
    return RigidObjectCfg(
        prim_path=prim_path,
        spawn=sim_utils.CuboidCfg(
            size=(0.3, 0.3, 0.1),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=False,
                disable_gravity=True,
                max_linear_velocity=0.0,
                max_angular_velocity=0.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.4, 0.9)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=pos),
    )


@configclass
class ExploreSceneCfg(InteractiveSceneCfg):
    """多机探索场景，最多 4 机。

    4 个 droneN/lidarN 字段都定义着，实际用几个由 TraceExploreEnvCfg.__post_init__
    把多余的置成 None 决定 —— InteractiveScene 会跳过 None 的 asset
    （interactive_scene.py:732），所以置 None 等于不 spawn。
    初始位置都在半径 2 m 的起飞平台上（platform_width=2.0）。
    """

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=EXPLORE_TERRAIN_CFG,
        collision_group=-1,
    )
    light = AssetBaseCfg(prim_path="/World/light",
                         spawn=sim_utils.DomeLightCfg(intensity=2000.0))

    drone0 = drone_cfg("{ENV_REGEX_NS}/Drone0", (-2.0, 0.0, 1.5))
    drone1 = drone_cfg("{ENV_REGEX_NS}/Drone1", (2.0, 0.0, 1.5))
    drone2 = drone_cfg("{ENV_REGEX_NS}/Drone2", (0.0, -2.0, 1.5))
    drone3 = drone_cfg("{ENV_REGEX_NS}/Drone3", (0.0, 2.0, 1.5))
    lidar0 = lidar_cfg("{ENV_REGEX_NS}/Drone0")
    lidar1 = lidar_cfg("{ENV_REGEX_NS}/Drone1")
    lidar2 = lidar_cfg("{ENV_REGEX_NS}/Drone2")
    lidar3 = lidar_cfg("{ENV_REGEX_NS}/Drone3")


# 4 机的初始位姿，都在半径 2 m 的起飞平台上（platform_width=2.0）
DRONE_INIT_POS = [(-2.0, 0.0, 1.5), (2.0, 0.0, 1.5), (0.0, -2.0, 1.5), (0.0, 2.0, 1.5)]


def set_scene_agents(scene_cfg: ExploreSceneCfg, num_agents: int) -> None:
    """按 num_agents 重建 droneN/lidarN 字段，多余的置 None。

    为什么要"重建"而不只是"置 None"：只置 None 的话，一旦 __post_init__ 被调用
    两次（第二次 num_agents 更大），先前被置 None 的机就再也回不来了，
    而且报错会推迟到 scene["droneN"] 取不到的时候，很难查。这里每次全量重写，
    所以 num_agents 可以在任何时刻改。
    """
    if not 1 <= num_agents <= len(DRONE_INIT_POS):
        raise ValueError(f"num_agents 必须在 1~{len(DRONE_INIT_POS)}，收到 {num_agents}")
    for i, pos in enumerate(DRONE_INIT_POS):
        if i < num_agents:
            setattr(scene_cfg, f"drone{i}", drone_cfg("{ENV_REGEX_NS}/Drone%d" % i, pos))
            setattr(scene_cfg, f"lidar{i}", lidar_cfg("{ENV_REGEX_NS}/Drone%d" % i))
        else:
            setattr(scene_cfg, f"drone{i}", None)
            setattr(scene_cfg, f"lidar{i}", None)


def lidar_ray_dirs_w(lidar) -> "torch.Tensor":
    """LiDAR 各光线的世界系单位方向，形状 (E, R, 3)。

    RayCaster 内部算过这个量（``_ray_directions_w``，ray_caster.py:292），
    但没有在 ``RayCasterData`` 里暴露。这里按 ``ray_alignment`` 的三个分支重算，
    与 ray_caster.py:267-287 一致，避免依赖私有属性。

    用途：``RayCaster`` 对未命中的光线返回 inf，方向信息丢失。把这个方向传给
    ``VoxelOccupancyMap.update(ray_dirs=...)``，打空的光线才能正确把 max_range
    内的自由空间清出来，否则探索率会被低估。
    """
    from isaaclab.utils.math import quat_apply

    align = getattr(lidar.cfg, "ray_alignment", "base")
    if align in ("world", "yaw"):
        # 这两种对齐下方向不随姿态旋转（只有 ray_starts 旋转）
        return lidar.ray_directions
    return quat_apply(lidar.data.quat_w.repeat(1, lidar.num_rays), lidar.ray_directions)
