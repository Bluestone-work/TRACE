# Baseline 基座结构

## 1. 研究层次

```text
TRACE 自研任务层
  ├─ trace/mapping/occupancy_map.py   三态 log-odds 体素地图
  ├─ trace/mapping/frontier.py        frontier 连通分量与 task node
  ├─ trace/envs/trace_explore_env.py DirectMARLEnv / MAPPO-CTDE 接口
  └─ trace/viz/                       2D 调试与 Blender/Isaac 3D 出图

3D 运行层（Isaac Lab）
  ├─ upstream/IsaacLab/               源码/配置快照
  └─ tests/isaac_tests/               启动、LiDAR、地图、frontier、MARL、消融测试

飞行动力学与部署层（Isaac Sim 6.0.1）
  └─ upstream/NavRL/isaac6-training/
       NavRL PPO → 三维速度命令 → Lee/电机模型 → Hummingbird USD → PhysX
       A: NavRL 随机目标导航
       B: RGB 标记检测 → alpha-beta → SEARCH/TRACK/COAST → LiDAR 安全层

对比与通信参考
  ├─ upstream/MARVEL/                2D 多机探索基线
  └─ upstream/IR2/                  通信受限多机器人参考实现
```

## 2. 当前基线的观测边界

A 使用官方 NavRL 状态、36×4 LiDAR、动态障碍输入和 PPO 均值动作。B 只使用自己的 RGB、自己的位置/时间以及自己的 LiDAR；B 的 RGB 检测使用 A 的洋红色视觉标记和已知球半径估计距离。A 的真实状态只在 runner 的评估部分计算间距、误差和安全指标，不进入 B 的控制输入。

两架无人机共用一个 240 Hz PhysX 时钟，策略/动作频率为 30 Hz，相机与视觉跟踪更新为 10 Hz。两机使用同一 Hummingbird USD、质量、惯量、推力上限、电机响应和速度指令上限。

## 3. 归档和实验产物的边界

`TRACE_baseline_foundation` 保存可复用的代码、依赖快照、模型和小型验证报告。PNG/GIF 帧序列、JSONL 轨迹和训练日志仍留在原始 `TRACE_baseline`，这样基座目录不会随着每次实验不断膨胀。新增实验建议把输出写入基座外的独立 run 目录，并在 `results/` 中登记最终 JSON 指标。
