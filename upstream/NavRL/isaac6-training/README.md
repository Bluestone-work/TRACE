# NavRL / Isaac Sim 6.0.1

本目录使用本机 `/home/wj/isaacsim`（Python 3.12），保留 NavRL 的三维速度动作和
8 维状态、36×4 LiDAR、动态障碍输入形状。原 `isaac-training` 与 `quick-demos`
源文件未修改。默认已切换为旋翼动力学；早期的直接位姿适配仅保留为对照模式。

双无人机第一版（A：NavRL 随机目标导航；B：RGB 搜索跟踪；同速同动力学）：

```bash
./run_tracking_demo.sh --gui --loop --realtime
```

参数、视觉数据边界、评估和录制见 [TRACKING_DEMO.md](TRACKING_DEMO.md)。

## 启动可视化

```bash
cd /home/wj/桌面/无人机/TRACE_baseline_foundation/upstream/NavRL/isaac6-training
./run_realistic.sh
```

这是原 PyTorch 预训练策略驱动的 Isaac Sim RTX 实际渲染。默认工业场景、跟随镜头，
无人机会倾斜加速和反向倾斜制动，旋翼转动。窗口显示实际速度、倾角与总推力，
`Show LiDAR / trajectory` 可打开传感器/轨迹显示。地面 H 标记上方 2 m 是目标。

- `./run_viewer.sh`：调试场景，默认显示 LiDAR 和轨迹。
- `./run_realistic.sh --camera overview`：固定全景。
- `./run_realistic.sh --dynamics kinematic`：对照用的直接位姿适配。
- `--checkpoint artifacts/navrl6_rotor_finetune.pt`：加载指定权重。

保留调用终端的 DISPLAY。若没有设置，则优先使用已有 VNC 桌面 `:3`（本机端口
5903），否则使用 `:0`。物理桌面非活动时可使用 `DISPLAY=:3`。首次 RTX 着色器
编译可能需要约 1–2 分钟。

## 推理、评估与录制

```bash
./run_torch_policy.sh --dynamics rotor --steps 360 \
  --report artifacts/rotor_policy_evaluation.json
DISPLAY=:3 ./run_torch_policy.sh --gui --style realistic --dynamics rotor \
  --steps 180 --record-dir artifacts/rotor_frames --record-every 6
/home/wj/miniconda3/envs/navrl/bin/python make_replay.py \
  artifacts/rotor_frames artifacts/navrl6_rotor_replay.gif
```

`--screenshot artifacts/preview.png` 保存最后一帧，`--renderer PathTracing` 可选择
离线路径追踪（默认实时路径追踪）。`--loop --realtime` 持续运行到窗口关闭。
模型默认使用官方发布的 `quick-demos/ckpts/navrl_checkpoint.pt`，评估时取分布均值。
启动日志会明确打印 checkpoint 路径和动力学模式。

## 训练

```bash
./run_train_navrl6.sh --dynamics rotor --iterations 10 --steps 32 \
  --output artifacts/navrl6_rotor_checkpoint.pt
```

当前训练入口从官方预训练网络进行微调。rollout 来自实际旋翼动力学和测量速度，
每批通过 GAE、PPO clipping、actor/critic 反向传播更新参数。`--checkpoint ...`
可从指定网络权重开始；当前保存的是网络与 ValueNorm 状态，不包含 Adam 状态，
所以这属于权重恢复，不是训练进程的逐步精确恢复。两次短更新仅验证训练链路，
不代表策略已经重新收敛。

目前是单环境、静态任务的最小训练器。随机化多环境、原始大规模动态障碍任务、
完整训练统计与论文指标尚未全部迁移。静态奖励使用原导航任务的速度、对数净空、
平滑性和高度项；目标到达、接近障碍、飞行边界和倾角异常触发终止。

## 动力学验证

```bash
PYTHONPATH=/home/wj/isaacsim/torch_site \
  /home/wj/isaacsim/python.sh check_controller_parity.py
/home/wj/isaacsim/python.sh verify_dynamics.py
```

已验证控制器与电机模型各 100 组输入的原实现数值一致性，以及自由落体、6 秒悬停、
X/Y 速度跟踪、制动、爬升、受扰姿态恢复。还完成了 360 步预训练策略评估和
2×32 步动力学 rollout 的 PPO 更新、权重变化与有限性检查。

结果：`artifacts/controller_parity.json`、`artifacts/dynamics_checks.json`、
`artifacts/rotor_policy_evaluation.json`、`artifacts/rotor_training_verification.json`。

映射公式、参数、频率与训练方案解释见 [FLIGHT_DYNAMICS.md](FLIGHT_DYNAMICS.md)。

## 运行环境

Torch 包位于 `/home/wj/isaacsim/torch_site`，包含 Torch 2.7.1、TensorDict 0.7.1、
TorchRL 0.7.1。加载学习模型期间暂时屏蔽 Isaac FastFinder，防止它将 Torch 子模块
指向废弃 `omni.isaac.ml_archive`；完成后恢复导入器。TorchRL C++ 扩展警告不影响
已验证的这条 Python PPO 路径。
