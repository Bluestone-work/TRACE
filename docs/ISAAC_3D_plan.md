# 3D 路线：Isaac Sim 5.1 + Isaac Lab 2.3.2

选这条路的前提：你要 3D 无人机仿真，且本机无 sudo。
Isaac Sim 可以纯 pip 装（已验证 `pypi.nvidia.com` 提供 `isaacsim 5.1.0.0`，py3.11），
不碰系统权限。

## 版本配对（已核对）

| 组件 | 版本 | 依据 |
|---|---|---|
| Isaac Sim | 5.1.0.0 | pip 源只提供 5.0.0.0 / 5.1.0.0，6.x 不在 pip 上 |
| Isaac Lab | **v2.3.2** | 官方兼容表：`v2.3.X` ↔ Isaac Sim 4.5/5.0/5.1 |
| Python | 3.11 | isaacsim 5.1 的 wheel 是 `cp311` |
| conda env | `trace_isaac` | 与 2D 的 `trace_explorer` 隔离，互不干扰 |

注意：Isaac Lab `main` / `release/3.0.0` 要 Isaac Sim **6.2**，pip 装不到，别跟着官网 main 分支的文档走。

## Isaac Lab 现成能用的（已在 clone 里核对）

这几项决定了"自己写"的工作量比想象的小：

- **`DirectMARLEnv`** — `source/isaaclab/isaaclab/envs/direct_marl_env.py`
  原生多智能体 env 基类，不用自己造多机接口。
- **MAPPO（CTDE）** — skrl 后端已带 MAPPO 配置，
  例子：`direct/shadow_hand_over/agents/skrl_mappo_cfg.yaml`、
  `direct/cart_double_pendulum/agents/skrl_mappo_cfg.yaml`。
  正好是文档 4.1 框架图里的 `MAPPO Actor + Centralized Critic`。
- **四旋翼任务** — `direct/quadcopter/quadcopter_env.py`（+ rsl_rl / skrl / rl_games 三套 PPO 配置）
  可直接当 3D 单机 baseline 跑通，确认物理和训练链路。
- **LiDAR 传感器** — `sensors/ray_caster/patterns/` 提供
  `LidarPatternCfg`（`channels` / `vertical_fov_range` / `horizontal_fov_range` / `horizontal_res`）、
  `BpearlPatternCfg`、`GridPatternCfg`、`PinholeCameraPatternCfg`。
  GPU 光线投射，可按 Mid360 的参数配，直接支撑文档 4.4 的
  `z_i^lidar = {(d_k, m_k)}` 角度分区表示。
  注意这是旋转式扫描模型，不是 Livox 的非重复扫描型式。
- **RL 库** — rsl-rl 3.1.2 / skrl>=1.4.3 / rl-games / sb3，按需选。

## 必须自己写的（这条路的代价）

Isaac Lab 给的是物理 + 传感 + RL 框架，**没有探索任务层**。要补：

1. **3D occupancy map**：从 LiDAR 点云增量建 voxel 地图，维护 free / occupied / **unknown** 三态。
   文档 4.4 明确要区分"没看到"和"看到了但没障碍"，这是 `m_k` 的来源。
2. **Frontier / unknown component 提取**：3D 版连通分量标记，产出文档 3.1.1 的 task node。
   这是 C²-Explorer 的核心，代码没开源，得自己实现（3D CCL + 受限 A*）。
3. **多机地图融合 + 通信模型**：谁在什么时候能收到谁的 task 信息。
4. **Task graph → MARL 接口**：把 task unit 打包成 `DirectMARLEnv` 的 observation。
   这是文档 5.1 自己点明的"真正需要重点开发的部分"。
5. **动态障碍**：Isaac 里加运动物体不难（物理引擎现成），`Risk_m^t` / `TTC` 由此而来。

工作量约等于把 MARVEL 的 env 层重写到 3D。好处是写完就是真 3D + 真物理 + 可 sim-to-real，
且 2×4090 能跑并行环境，训练吞吐远超 ROS1 路线。

## 2D baseline 的作用没废

`upstream/MARVEL/` 那套仍然有用：创新点二（注意力偏置）和创新点四（counterfactual critic）
是纯决策层改动，在 2D 里做 ablation 快得多，结论可直接搬到 3D 的网络里。
3D 侧先打通 occupancy + frontier + task graph 这条链，两边并行推进最省时间。

## 未采用的方案及原因

- **MARSIM**（文档 5.1 点名的仿真器）：真 Livox 点云、多机、动态障碍、GPU 版都有，
  但 **ROS2 分支只移植了单机**——`triple_drone_mid360.launch` 仍是 ROS1 语法
  （`type=` / `$(find ...)`），多机必须 ROS1 Noetic，即需要 Docker 和 sudo。
- **RACER / FALCON**：ROS1 Noetic（FALCON 只有 `ros1-noetic` 分支），同上。
  要做 CVRP 强 baseline 数值对比，这步绕不开，建议后期单独用容器补。
- **Aerial Gym**：GPU 并行 + 光追 LiDAR 很对口，但仍基于已弃用的 Isaac Gym
  （要 NVIDIA 账号手动下载），其 Isaac Lab 支持官方标注 under development。
- **OmniDrones**：多机无人机 RL 平台，但作者已在 README 公开声明停止维护。

---

# 已验证（2026-11-01）

三项全部通过，Isaac 侧 baseline 立住了。自检：`./scripts/isaac_smoke.sh`

## 1. 启动 + 物理 — `isaac_tests/01_launch_physics.py`

```
[OK] AppLauncher 启动成功（未崩溃）
[OK] torch: 2.7.0+cu128 cuda: True ngpu: 2
[OK] SimulationContext 创建完成, device = cuda:0
[OK] 物理步进 200 步完成, 仿真时间 = 1.667 s
```

## 2. RL 训练链路 — skrl PPO

`Isaac-Quadcopter-Direct-v0`，**256 并行环境，240 timesteps，7.16 秒**，无报错。
rsl-rl 3.1.2 / rl_games 1.6.1 / sb3 也都装好了（sb3 要 torch>=2.8，当前 2.7.0，用它得先升）。

## 3. GPU 光线投射 LiDAR — `isaac_tests/02_lidar_raycast.py`

40 线 × 720 方位 = 28800 根光线/机，4 个并行环境：

```
[OK] 点云 shape = (4, 28800, 3)
[OK] env0 有效命中 = 23760 / 28800
[OK] 命中距离 = 5.08 ~ 46.76 m (载具高 5m)
[OK] 张量在 GPU 上: True
```

数字可自校验：载具 5 m 高，最陡光线 −80°，5/sin(80°) = 5.08 m，与最小命中距离一致；
未命中的 5040 根正是 +10° 朝上打不到地面的那些。点云直接在 GPU 上，不过 PCIe。

---

# 必须知道的坑（都踩过了）

## 1. 不要用 `isaacsim.SimulationApp`，用 Isaac Lab 的 `AppLauncher`

这是最关键的一条。直接用 Isaac Sim 自带的 base python 实验配置会**启动即崩**：

```
[5.884s] app ready
rtx.scenedb.plugin  →  UsdContext::createViewportImpl - engine type: 'rtx'
Crash detected in pid ... thread ...
```

原因是它在 headless 下仍加载数百个扩展并创建 RTX viewport。
Isaac Lab 的 `apps/isaaclab.python.headless.kit` 只启用 **19 个**扩展、不加载 viewport 窗口扩展，
换过去之后零崩溃。

排查时错判过两次，记录下来免得重走：**不是驱动版本**（官方 Linux 要 595.58.03，本机 595.84，
同分支更新版）；**不是多卡/IOMMU**（单卡 + `multi_gpu=False` 同样崩在相同位置）。
日志里那些 `IOMMU is enabled`、`PCIe link width current (2)`、`P2P 带宽 3.2 GB/s` 都是噪音警告，
**不需要改 GRUB、不需要关 IOMMU、不需要重启**。

## 2. `simulation_app.close()` 会无限挂起

物理步进全部跑完后卡死，实测超 120 秒不返回。脚本结尾直接 `os._exit(0)`，别调 `close()`。
批量跑实验必须套 `timeout`，否则堆僵尸进程占显存。
（Isaac Lab 自带的 `train.py` 不受影响，能自己干净退出。）

## 3. `isaaclab.sh --install` 会静默装不上核心包

退出码是 0，但 `isaaclab` 核心包根本没装上。根因在日志里：

```
flatdict 4.0.1 → ModuleNotFoundError: No module named 'pkg_resources'
ERROR: Failed to build 'flatdict' when getting requirements to build wheel
```

**setuptools ≥81 移除了 `pkg_resources`**，而 flatdict 的 setup.py 还在用。修法：

```bash
pip install "setuptools<81"
pip install "flatdict==4.0.1" --no-build-isolation
pip install -e source/isaaclab          # 手动补装核心包
```

装完务必 `pip list | grep -E '^isaaclab '` 确认，别只看退出码。

## 4. packaging 版本会被顶高，要钉回去

`isaacsim-core` 锁 `packaging==23.0`，`isaaclab-rl` 要 `<24`，但 isaaclab 的依赖链会把它拉到 26.3。

```bash
pip install "packaging==23.0" "torchaudio==2.7.0" --extra-index-url https://download.pytorch.org/whl/cu128
```

装完 `wheel` 和 `pytest` 会抱怨要 `packaging>=24` —— 只影响构建和测试，运行不受影响，忽略。
`torchaudio` 是 `isaacsim-core` 的硬依赖，但 `isaaclab.sh` 不会装，要自己补。

## 5. EULA

首次启动要接受 NVIDIA Omniverse 许可协议，非交互运行会直接 EOF 失败。
已设成 env 永久变量：`conda env config vars set OMNI_KIT_ACCEPT_EULA=YES -n trace_isaac`

## 6. `RayCasterCfg` 的两个限制

- 场景字段要包在 `AssetBaseCfg` 里（`ground = AssetBaseCfg(prim_path=..., spawn=GroundPlaneCfg())`），
  直接把 `GroundPlaneCfg()` 当字段用会失败。
- `RayCaster` 只打**静态** mesh，`mesh_prim_paths` 指静态几何。
  要扫动态障碍（TRACE 的 `Risk_m^t` / `TTC` 需要）得用 `MultiMeshRayCaster`，
  参考 `source/isaaclab/test/sensors/test_multi_mesh_ray_caster.py`。

## 7. 运动学刚体 + RayCaster = CUDA illegal memory access

给 `kinematic_enabled=True` 的刚体写位姿后，RayCaster 下一次 `update()` 必崩：

```
[omni.physx.tensors.plugin] CUDA error: an illegal memory access was encountered:
    GpuRigidBodyView.cpp: 512
Traceback ... sensors/sensor_base.py line 186, in update
RuntimeError: CUDA error: an illegal memory access was encountered
```

原因：RayCaster 会沿 prim 往上找最近的带 `UsdPhysics.RigidBodyAPI` 的祖先，
用 PhysX 刚体 view 读位姿（`ray_caster.py:366-384`）。运动学刚体在 GPU 管线里
不走直接位姿写入这条路。

正确做法——**非运动学刚体 + 关重力 + 每步清零速度**：

```python
rigid_props=sim_utils.RigidBodyPropertiesCfg(
    kinematic_enabled=False, disable_gravity=True,
    max_linear_velocity=0.0, max_angular_velocity=0.0),
mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
...
d.write_root_pose_to_sim(root[:, :7].contiguous())
d.write_root_velocity_to_sim(torch.zeros_like(root[:, 7:13]))
```

排查提示：报错行号指向 `write_root_pose_to_sim` 的下一句 `scene.update()`，
容易误判成位姿写入本身的问题。把"纯步进"和"写位姿"分开跑就能定位。

## 8. 单 sub-terrain 下多 env 会重叠

`TerrainGeneratorCfg(num_rows=1, num_cols=1)` 只生成一块地形，
`terrain_type="generator"` 时 `scene.env_origins` 由 sub-terrain 原点决定、
**不看 `env_spacing`**。num_envs>1 时所有 env 原点都是 (0,0,0)，会叠在一起。
要并行多 env 得把 `num_rows`/`num_cols` 开到够用。

---

# 占据栅格（已实现并验证）

`trace/mapping/occupancy_map.py` — 按 num_envs 批量化的三态 log-odds 体素地图，
算法和参数沿用 NavRL 的 `ros2/map_manager`，重写为 torch GPU 张量实现。

## 三态语义（文档 4.4 要求区分"没看到"和"看到了但没障碍"）

| 状态 | 判据 | log-odds 值 |
|---|---|---|
| unknown | `occ < p_min_log` | 初始 −2.002 |
| free | `p_min_log ≤ occ < p_occ_log` | −1.992 ~ 1.386 |
| occupied | `occ ≥ p_occ_log` | ≥ 1.386 |

参数取 NavRL 的 `cfg/map_param.yaml`：`p_hit 0.70 / p_miss 0.35 / p_min 0.12 /
p_max 0.97 / p_occ 0.80`，对应 log-odds `+0.847 / −0.619 / −1.992 / +3.476 / +1.386`。

更新为 hit 优先：同帧同体素既被穿过又被命中时取 `p_hit_log`。
unknown 体素仅被穿过时直接置 `p_min_log`（free 先验）。

**一个有意的滞后**：真实障碍体素从 unknown 起要约 4 次命中才越过 `p_occ_log`
（(1.386+1.992)/0.847 ≈ 3.99），这是 NavRL 的噪声过滤设计。
但 unknown → known 在第一次观测即翻转，所以探索覆盖率指标不滞后。

## 实现要点

- 地图张量 `(num_envs, nx, ny, nz)` float32，多机共享同一张队伍地图，
  `update()` 时把各机光线拼在 agent 维即可。
- 自由空间用沿光线等间隔采样（步长 = 分辨率）后 `scatter_add_` 累加，
  不做精确 DDA。按 `ray_chunk` 分块限制峰值显存，`ray_stride` 可下采样光线。
- `RayCaster` 对未命中光线返回 `inf`，丢失方向信息。`update()` 支持可选
  `ray_dirs` 参数：给了它，打空的光线会沿该方向把 `max_range` 内清成 free；
  不给则这些方向保持 unknown（偏保守，会低估探索率）。

## 踩过的 bug：表面前的 unknown 壳

早期版本自由空间采样条件写成 `t < dist - res/2`（想排除命中体素），
结果每个表面前留下一层**永不被观测**的 unknown 体素。
实测：光线 (0,0,1)→(3,0,1)，自由采样只到 x=2.625（体素 30），
命中体素是 32，体素 31 从未被观测。这会产生假 frontier 并低估覆盖率。

修法是去掉余量（`t < dist`），重叠靠 hit 优先规则解决。

## 验证结果

**单元测试** `isaac_tests/00_occupancy_unit.py`（纯合成数据，可解析验证）：

```
1 次更新后 free=13  (= 0~3m / 0.25m = 12 穿过 + 1 命中，无空洞)
命中体素 log-odds: -1.155 -> -0.308 -> 0.539 -> 1.387
第 4 次命中后转为 occupied，occupied 体素数 = 1
```

**集成测试** `isaac_tests/03_occupancy_map.py`（2 机 + 80 个静态障碍 + 真实 LiDAR）：

```
体素地图: 160x160x24 = 614400 体素/env  (40x40x6 m, res 0.25)
  step | known_ratio |  free   | occupied | unknown
     0 |   0.0142    |    8755 |        0 |   605645
    80 |   0.0665    |   40150 |      707 |   573543
   160 |   0.1127    |   68447 |      791 |   545162
   239 |   0.1504    |   91391 |     1007 |   522002
```

known_ratio 单调增长，occupied 数量在 ~550→1007 间有小幅波动是正常的——
掠射角下的体素会被反复判为穿过而掉回 free。

## 接上 ray_dirs 的效果与代价

`RayCaster` 对未命中光线返回 `inf`。把光线方向传给 `update(ray_dirs=...)` 后：

| | known_ratio | occupied | free |
|---|---|---|---|
| 不传 ray_dirs | 0.1504 | 1007 | 91391 |
| **传 ray_dirs** | **0.8730** | 525 | 535848 |

覆盖率从 15% 升到 87%——之前低估得很厉害，因为 LiDAR 垂直视场 ±30°、量程 12 m，
大量光线打空返回 inf，这些方向的自由空间原来完全没被清出来。

**代价：occupied 从 1007 掉到 525。** 这不是 bug，是 log-odds 的固有权衡。
打空的光线会沿途清空自由空间，而等间隔采样在掠射角下会"切到"障碍体素的边缘，
给它累加 miss。`p_hit(+0.847) > |p_miss|(0.619)` 的设计本意就是让占据体素
需要比 miss 更频繁的 hit 才能维持，所以薄障碍会被侵蚀。

调节手段（按需要选）：
- 提高 `p_hit` 或降低 `p_miss`（最直接）
- 把自由空间采样步长从 `res` 减到 `res/2`（减少"切角"，但显存和耗时翻倍）
- 对命中光线把清空距离再缩短半个体素（会退回"表面前 unknown 壳"问题，不推荐）

当前对探索覆盖率指标没影响（unknown→known 的翻转不受此影响），
但如果后面要用这张图做碰撞检查或 CBF 安全层，必须先把这个调好。

---

# frontier 提取 / task node（已实现并验证）

`trace/mapping/frontier.py` —— 对应文档 3.1.1：把空间上不连通的未知区域
拆成独立 task unit。

定义：

    frontier 体素 := free 且 6(或26)-邻域中存在 unknown
    task unit     := frontier 体素的连通分量

`extract_task_nodes()` 输出按体素数降序、填充到 `max_nodes` 的张量字典：
`pos`（质心）/ `n_voxels` / `n_unknown`（IG_m 代理量）/ `bbox_min` / `bbox_max` /
`valid` / `num_nodes`，可直接接 `DirectMARLEnv` 的 observation。

## 重要：CCL 默认用 scipy，不要用 GPU 迭代法

GPU 迭代标签传播（每轮取邻域最大标签）看起来很美——免 CPU 同步、全程张量化——
但在真实地图上 **256 轮经常不收敛**，而未收敛会把一个连通分量拆成多个标签，
造成过分割。同一场景同一时刻的对比：

| step | GPU 迭代法 簇数 | scipy 精确 簇数 |
|---|---|---|
| 0 | 179 | 182 |
| 80 | 64 (未收敛) | 36 |
| 160 | 138 (未收敛) | 47 |
| 240 | **272** (未收敛) | **63** |
| 320 | 141 (未收敛) | 59 |
| 399 | 39 | 46 |

task unit 数量是创新点一的核心输入，过分割不可接受。所以 `FrontierCfg.method`
默认 `"cpu"`（`scipy.ndimage.label`，精确）。GPU 路径保留为 `method="gpu"`，
只在确认收敛（返回的第二个值为 True）时可用。

想加速也不要给 max_pool 加 dilation —— 带 dilation 的最大值池化会让标签
跨越背景空隙传播，把本不连通的分量错误合并。

## 性能

160×160×24 = 614400 体素，scipy 精确 CCL 实测 **126~175 ms**。
文档 4.2 设定高层决策 1~2 Hz，这个开销完全够用；但不要放进每个物理步。

## 验证结果

**单元测试** `isaac_tests/04_frontier_unit.py`（合成地图，全部手算核对）：
两块被 unknown 隔开的自由空间 → 恰好 2 个分量，
质心 (0.75, 2.0, 0.25) / (3.25, 2.0, 0.25)，各 24 体素，
`min_cluster_voxels=30` 时两簇都被正确丢弃。cpu 与 gpu 两条路径结果一致。

**集成测试** `isaac_tests/05_frontier_integration.py`（80 障碍 + 2 机真实 LiDAR）：

```
  step | known  | frontier | 簇数 | 最大簇 | 精确 | CCL耗时
     0 | 0.1134 |    54733 |  182 |  47891 | True |  128.1 ms
    80 | 0.4325 |    49581 |   36 |  45159 | True |  174.9 ms
   240 | 0.7018 |    54739 |   63 |  16026 | True |  170.7 ms
   399 | 0.9142 |    29155 |   46 |   2229 | True |  130.5 ms
```

最大簇从 47891 体素降到 2229，即那层包裹未知区域的大 frontier 壳随探索推进被打碎。
最终前 4 簇体素数几乎相同（2229/2219/2218/2206），质心在
(±15.1, ±15.1, 1.5) 四个角——两机沿半径约 13 m 的圆飞行，四角探不到，
对称性和位置都符合预期，是个有力的正确性旁证。

---

# MARL 环境接口（已实现并验证）

`trace/envs/trace_explore_env.py` —— `DirectMARLEnv` 子类，把
occupancy map → frontier → task graph 这条链接到 MARL 接口上。
这正是文档 5.1 点明的"真正需要重点开发的部分：Task Graph → MARL Interface"。

## 两时间尺度（文档 4.2）

| 层 | 频率 | 实现 |
|---|---|---|
| 物理 | 60 Hz | `sim.dt = 1/60` |
| 低层控制 | 15 Hz | `decimation = 4`，动作为速度指令 |
| 建图 | 7.5 Hz | `map_update_every = 2` |
| 高层任务图 | 1 Hz | `task_graph_every = 15` |

高层频率必须压到 1 Hz 左右，因为精确 CCL 单次 130~175 ms，放进每个控制步会直接拖死吞吐。

## 观测布局（每机 163 维）

```
[0:3]     ego 位置（相对地图中心，按地图尺寸归一化）
[3:6]     ego 速度（按 max_speed 归一化）
[6:42]    36 方位 LiDAR 最小距离环（文档 4.4 的角度分区思想）
[42:154]  task graph: 16 个 task node x 7 维
          [Δpos(3), 距离(1), n_voxels(1), n_unknown(1), valid(1)]
[154:160] 队友相对位置/速度
[160:163] g_topo：当前选中 task 的相对方向（文档 4.3 的 long-horizon guidance）
```

特权状态 109 维（`_get_states()`，CTDE 中心化 critic 输入）：
全部 agent 位置+速度(12) + known_ratio(1) + task graph 摘要(96)。
**创新点四 Counterfactual Marginal Utility 就落在这里**：
`u_i = V(s) − V(M_i(s))`，把 agent i 的状态连同其独有地图证据一起屏蔽后重算 critic。

## 奖励（文档 4.5）

`r = w_I·r_info − w_R·r_redundancy − w_C·r_collision − w_T·r_time`

`r_info` 用团队新增已知体素占比。`r_utility`（创新点四）尚未实现，
当前用"队友过近则惩罚"的代理项占位 —— 对应 MARVEL 里手工版的
`calculate_overlap_reward`，将来由中心化 critic 的差值替换。

## 高层任务选择是占位的

`_select_tasks()` 现在是贪心启发式：效用 = `n_unknown / (距离+1)` 取 argmax。
**这正是创新点三 Foresighted Task-Value Head 要替换的对象**，
留在这里刚好作为对照 baseline。换成学习式只需把这里改成 `argmax_m Q_{i,m}^exp`。

## 验证结果

**环境端到端** `isaac_tests/06_marl_env.py`（2 env × 2 机，随机动作）：

```
[OK] 控制频率 = 15.0 Hz, 建图 7.5 Hz, 任务图 1.00 Hz
[OK] 观测维度 = (2, 163)   特权状态维度 = (2, 109)
  step | known  | task节点 | 平均奖励 | 观测有限
     0 | 0.1926 |      16 |   -0.135 | True
    20 | 0.3051 |      16 |   -0.152 | True
    39 | 0.3242 |      16 |   +0.043 | True
[OK] 最终 known_ratio = [0.3242, 0.3545]
```

两个 env 的 known_ratio 不同，确认环境相互独立（地形网格按 num_envs 自动铺开）。

**MAPPO 训练链路** `scripts/train_trace_marl.py`：

```
[INFO] 任务=Trace-Explore-MARL-v0 num_envs=4 timesteps=80
[skrl:INFO] Environment wrapper: Isaac Lab (multi-agent)
100%|██████████| 80/80
[DONE] 训练流程结束
```

skrl 正确识别为 multi-agent 并走 MAPPO（CTDE）路径，80 timesteps 无报错，
吞吐 13~43 it/s。整条链路打通：
建图 → frontier → task graph → 观测 → MAPPO(中心化 critic 吃特权状态) → 动作 → 位姿 → 新观测。

## 当前版本的已知短板（训练出结果前必须处理）

1. **奖励尺度没调**。随机策略下平均奖励在 −0.15 ~ +0.04，`r_info` 太小，
   被 redundancy 和 time 惩罚盖住。真正训练前要先把 `w_info` 调大或重新归一化。
2. **task node 被截断**。`max_task_nodes=16`，但实测簇数可达 46，
   观测里只放了最大的 16 个。要么调大，要么按文档 3.1.1 先做任务聚合。
3. **动力学是占位的**。速度指令直接积分位姿，不是真实四旋翼。
   换真实动力学后 `r_collision` 和安全层才有意义。
4. **CCL 开销**。1 Hz × 130~175 ms，4 env 时是实打实的吞吐税。
   要扩大并行规模得考虑增量式 frontier 更新（只重算地图变化的局部）。

## 9. 本机 RTX 渲染器不可用 —— 三条原生可视化路径全崩

实测结论：

| 方式 | 命令 | 结果 |
|---|---|---|
| GUI 窗口 | 去掉 `--headless` | **段错误**，`rtx.scenedb.plugin` |
| 离屏录像 | `--enable_cameras`（`--video` 会自动开） | **段错误**，同一栈 |
| WebRTC 串流 | `--livestream 2` | **段错误**，同一栈 |
| 不渲染 headless | `--headless` | 正常 |

三者崩在完全相同的位置，和坑 1 是同一个根因：这台机器的 RTX 渲染管线坏了，
只有"不开渲染的 headless 物理 + warp 光线投射"这条路可用。
注意 warp 的 LiDAR 光线投射**不经过 RTX**，所以传感器完全正常。

因此 TRACE 的可视化走 matplotlib 直接画数据，见下节。

---

# 可视化（已实现并验证）

`trace/viz/map_viz.py` + `isaac_tests/07_visualize.py` —— 把体素地图投影成俯视图，
叠加 frontier、task node、无人机位置与轨迹，输出 PNG 序列和 GIF。

```bash
cd upstream/IsaacLab
python -u ../../isaac_tests/07_visualize.py --headless --steps 400 --every 25
# 产物: runs/viz/frame_*.png 和 runs/viz/explore.gif
```

配色：unknown 灰 / free 白 / occupied 黑 / frontier 橙 / task node 红圈
（圈大小 ∝ n_unknown）/ 无人机蓝三角 + 轨迹。

俯视投影默认跳过 z<0.75 m：地面层整片 occupied，全高度投影会把图糊成黑的。

## 可视化立刻抓出的两个真 bug

画出来才发现的，光看数字发现不了 —— 这是做可视化最大的收益。

### bug 1：用质心当 task 位置，对环形簇是错的

图上看到标号 0 的"最大 task node"被画在无人机起点旁边的**已探明自由空间**里。
原因：外圈那条 frontier 环是**一个**连通分量，它的质心落在环的几何中心，
而那里早就探明了。贪心策略会把无人机派去一个它去过的点。

修法：`extract_task_nodes()` 的 `pos` 改为**代表点** —— 簇内离质心最近的
frontier 体素，保证落在 frontier 上；质心另存为 `centroid` 仅作统计。

验证（单层地图造半径 7 m 的 frontier 环）：

```
质心   centroid=[0.0, 0.0, 0.25]      在 frontier 上 = False   <- 证实 bug
代表点 pos     =[-5.75, -2.75, 0.25]  在 frontier 上 = True    <- 修复
```

### bug 2：6-连通会把数字化曲面切碎

同一个半径 7 m 的 frontier 环：

```
cluster=6  : frontier=76  分量数=32    <- 曲面被切成 32 段
cluster=26 : frontier=108 分量数=1     <- 正确
```

数字化圆弧的相邻体素是斜对角关系，6-连通不认。真实地图的 frontier 壳是曲面，
所以之前报的簇数里绝大部分是假分割。

修法：`FrontierCfg` 把原来混在一起的 `connectivity` 拆成两个语义：

- `adjacency = 6` —— 判定 frontier（free 体素的**面相邻**邻域里有 unknown），标准定义
- `cluster = 26` —— frontier 体素之间聚类，**必须 26**

对真实数据的影响（同一场景同一时刻）：

| step | 修复前 (6-连通) | 修复后 (26-连通) |
|---|---|---|
| 0 | 64 | **2** |
| 50 | 45 | **4** |
| 399 | 39 | ~10 |

**所以之前文档里"46 个 task unit"那个数是错的**，真实的拓扑分量数是个位数到十几个。
这反过来说明 `max_task_nodes=16` 其实够用，不需要调大。

## 可视化还确认了一件事

障碍物在图上只剩零星小黑点，几乎没有遮挡阴影 —— 直观证实前面提的
occupied 侵蚀问题比数字看起来更严重。80 个 0.4~1.1 m 宽的柱子在 0.25 m
分辨率下每个本该有几十个 occupied 体素。用这张图做碰撞检查或 CBF 之前必须先修。

## 一个建议

`FrontierCfg.z_range` 目前没用上，frontier 在全高度范围内提取，
所以会出现落在无人机飞行高度带之外的 task node（图上表现为红圈下面没有橙色）。
建议在环境里把它设成无人机可达高度带（如 (0.8, 4.0)），保证 task node 可达。

---

# RTX 渲染器故障的完整定位（结论：环境性，放弃修）

## 结论先行

Isaac Sim 的 RTX 渲染管线在这台机器上**起不来，且与版本无关**。
原生可视化（GUI / `--enable_cameras` / `--livestream`）全部不可用。
已改用 Blender(bpy) 离屏渲染出 3D 图，见下节。

**不要再尝试**：换 Isaac Sim 版本、关 IOMMU、换驱动。理由见下面的证据。

## 关键证据一：崩溃是主机侧段错误，GPU 是健康的

内核日志（`journalctl -k` / `/var/log/kern.log`，在 `adm` 组里无需 sudo）
全历史只有 **2 条 Xid**：

```
09:17:32  NVRM: Xid (PCI:0000:02:00): 31, name=python,
          MMU Fault: ENGINE GRAPHICS GPC2 ... faulted @ 0x0_00000000
09:18:31  同上
```

而 RTX 崩溃发生在 09:39~09:54 —— **崩溃时刻一条 Xid 都没有**。

那 2 条 Xid 已定位到我自己的 bug：`occ_map_test.log`（唯一含
"illegal memory access" 的日志）修改时间 09:17:33，正是坑 7 那个
"运动学刚体 + RayCaster" 问题，已修复。

所以：**GPU 和驱动是健康的**，计算路径（训练、warp 光线投射、PhysX）零错误。
RTX 崩溃是 `librtx.scenedb.plugin.so` 在 CPU 侧的段错误，GPU 根本没参与。
这直接否掉了"关 IOMMU""换驱动"这类针对 GPU 访问的修法。

## 关键证据二：Isaac Sim 5.0 崩在完全相同的位置

装了独立环境跑 Isaac Sim 5.0.0.0：

```
librtx.scenedb.plugin.so!std::vector<std::tuple<char const*, float, ...>>::_M_realloc_insert
librtx.scenedb.plugin.so!carbOnPluginStartup+0x3b2fe
```

与 5.1 **同一个函数、同一个符号**。两个版本都报同样的显存：

```
nvidia-smi 报  49140 MiB (48 GB)
Vulkan 层报    81898 MB
公版 4090 是   24 GB
```

不是版本回归，是环境性的。

## 关键证据三：换非 RTX 渲染后端就不崩

```bash
--kit_args="--/renderer/enabled=pxr --/renderer/active=pxr"
```

用 pxr（Storm/OpenGL）后端 + `--enable_cameras`，**exit=0，零崩溃**。
把范围锁死在 RTX 管线，不是 Vulkan 整体坏、不是显存不足。

但 pxr 不能出图：`capture_viewport_to_file` 拿得到 viewport 对象，
120 帧 `app.update()` 后仍不落盘（Kit 截图走 RTX 的 AOV 通道）；
GUI 模式用 pxr 照样崩。所以 pxr 只有诊断价值。

## 最强假设（无法进一步验证）

显卡是**改装卡**：device ID `0x2684`（原生 AD102）但 sub-device `0x16F3`、
VBIOS `95.02.3C.00.02`、48 GB 显存（公版 24 GB）。
崩溃符号是在插件启动期构建一张"字符串+数值"的表 —— 枚举 GPU 能力或显存堆的
典型形态，而这两张卡的显存上报本身就不自洽（81898 vs 49140）。
符号化是就近匹配，不能当铁证，但所有证据都指向这里，且不是我们能改的。

## 试过但无效的开关（全部仍崩）

`CUDA_VISIBLE_DEVICES=0` 锁单卡 / `/renderer/multiGpu/enabled=false` /
`/rtx/sceneDb/maxInstances=65536` /
`/rtx-transient/resourcemanager/enableTextureStreaming=false` /
清 shader+DerivedData+texture 缓存 / `/app/useFabricSceneDelegate=false` /
`--rendering_mode performance`

顺带两个小坑：`--enable_cameras` 比纯 headless 崩得更早（连 `app ready` 都到不了）；
`--kit_args` 的值若是单个以 `--` 开头的 token 会被 argparse 当成选项，要写
`--kit_args="--/a=b"` 这种 `=` 形式。

---

# 3D 出图：Blender(bpy) 离屏渲染

两段式管线（bpy 与 isaacsim 的依赖锁冲突，必须分环境，用 `.npz` 做中间格式）：

```bash
# 1) Isaac 侧：跑仿真并导出快照
conda activate trace_isaac
cd upstream/IsaacLab
python -u ../../isaac_tests/08_dump_for_render.py --headless --steps 300
# -> runs/viz/snapshot.npz

# 2) Blender 侧：渲染
conda activate trace_render
python trace/viz/render3d.py runs/viz/snapshot.npz runs/viz/render3d.png \
    --samples 64 --res 1600 1200 [--frontier-max 8000] [--no-frontier]
```

环境 `trace_render` = python 3.11 + `bpy 5.0.1` + numpy + imageio，纯 pip 无需 sudo。
Cycles CPU 渲染（24 核），1600×1200 / 64 samples 约 **13 秒**。
完全不碰 Vulkan/RTX，所以不受本机 RTX 故障影响。

渲染内容：占据体素（深灰）、frontier（半透明橙，默认下采样到 8000 个避免糊住内部）、
task node（粉球，半径 ∝ n_unknown）、无人机（蓝锥）、轨迹（蓝管）。

实测一帧：`occupied=849 frontier=33894 task=26 known_ratio=0.8954`。

## 顺带纠正前面一个判断

前面根据 2D 俯视图说"障碍物只剩零星小黑点，侵蚀严重"——**这个说法是错的**。
3D 渲染里障碍物是成形的体素柱，每个十几到几十个体素。
之前那是 160×160 投影的尺度假象（10 个体素投成几个像素）。
侵蚀现象本身存在（接 ray_dirs 后 occupied 1007→525 是真的），但严重程度被我夸大了。
要判断是否真的侵蚀过头，得和地形真值比对，目前还没做。
