# PSTO P0 验收报告

本报告记录 TRACE PSTO 迁移在本地 Isaac Sim 6.0.1 的验收状态。报告只列出已经实际执行的命令和产物；没有运行的长实验保持 `BLOCKED`，不会用 smoke 结果代替收敛证据。

## 当前结论

P0 当前为 **BLOCKED**。时间尺度、种子接口、配置追溯、GAE/timeout 语义、三机 PhysX、两套短训练、LiDAR 性能测量和十个真实 Episode 已通过本地验证。长时间 MAPPO 收敛仍未进行，且十个随机 Episode 都因碰撞提前结束，因此暂不建议启动正式长训练。

## P0-1 时间尺度

`configs/psto/official_reproduction.yaml` 来自上游 `sim.dt=0.01`、`frame_skip=10`，因此 `policy_dt=0.1`、10 Hz、280 步为 28 s。`configs/psto/trace_native.yaml` 保留 Isaac Sim 6 适配器的 240 Hz PhysX 与 30 Hz 策略，8 个物理步对应一个策略步，280 步为 9.333333 s。环境将真实观测时间间隔传给 LSTM，并在配置快照中记录 frame skip 和 episode duration。

状态：**PASS（配置和启动链路）**。两套配置均已在真实 Isaac Sim 6.0.1 中启动并完成短更新。

## P0-2 随机种子和场景

训练入口同时初始化 Python、NumPy、Torch CPU 和 CUDA（可用时），并将 seed 传入 `PSTOPursuitEnv`/`PSTOPursuitScene`。Scene 使用独立 NumPy RNG；`reset(seed=...)` 可以重建 episode 初始状态，`randomize_resets` 控制无人机位置扰动，`randomize_obstacles` 控制障碍物布局扰动。默认配置关闭随机化以保持官方初始排列，评估脚本显式打开并使用 `seed + episode`。

状态：**PASS（接口和十个不同 seed 的真实场景已复核）**。十个 Episode 使用 seed 4348–4357，平均 64.3 步；捕获 0、逃逸 0、碰撞 10、timeout 0。这个结果证明 reset/标记链路工作，但也暴露出当前固定动作验收策略的碰撞率问题，不能当作策略性能。

## P0-3 配置和实验记录

唯一入口为 `scripts/train_psto.py --config ...`。YAML 中的学习率、gamma、GAE、PPO clipping、熵系数、ValueNorm、LSTM 监督频率、时间尺度和输出目录会传入构造器。每次运行在 checkpoint 目录保存 `config_snapshot.yaml` 和 JSON 元数据，其中包含 Git SHA、Isaac Sim build、Python、Torch、CUDA、seed、时间尺度、帧数、更新次数、loss 和 gradient norm。默认 checkpoint 目录已有文件时会创建 `_runN`，避免覆盖旧实验。

状态：**PASS**。

## P0-4 LiDAR

原始语义保持 `[24, 480]` proximity，`proximity = range - hit_distance`，无命中为零；网络下采样仍为 `[6, 120]`。可见性判断使用最大距离、射线方向对齐和遮挡距离，并拒绝超过 10 m 传感器量程的目标。三架无人机各自过滤自身刚体。Scene 对同一个 PhysX 时刻缓存扫描，动作推进后失效缓存，避免 evader hit-point 和 observe 的重复扫描；缓存不会改变数值语义。

实际命令：

```bash
PYTHONPATH=/home/wj/isaacsim/extsDeprecated/omni.isaac.ml_archive/pip_prebundle:$PWD \
PYTHONEXE=/home/wj/isaacsim/kit/python/bin/python3 \
/home/wj/isaacsim/python.sh scripts/profile_psto_lidar.py --steps 1 --headless \
  --out results/psto_p0/lidar_profile
```

实际结果：`status=pass`，raw/downsampled 形状为 `24x480`/`6x120`。强制新扫描的平均三机扫描 0.1194 s，observe 0.1177 s，env step 0.1815 s（5.51 step/s）。结果见 `results/psto_p0/lidar_profile_fresh/`。同一 PhysX 状态的缓存命中不再重复射线；profile 脚本会显式失效缓存后测量原始开销。

## P0-5 GAE、timeout 和 reset

GAE 使用两个 mask：`terminated` 只控制 TD bootstrap；`episode_end` 控制优势递推。捕获、逃逸、碰撞等真正终止不 bootstrap；timeout 保留最终状态 Critic value，但不把优势传播到下一 episode。训练 rollout 保存每个 transition 的 `next_value`，环境返回 `terminated`、`truncated`、`episode_end` 三个独立字段。

`PYTHONPATH=$PWD /home/wj/miniconda3/envs/trace_isaac/bin/pytest -q tests/psto` 的实际结果为 **16 passed, 1 warning**。其中包含连续状态、terminal bootstrap、timeout bootstrap/递推屏蔽、84-D state、遮挡历史和奖励逐项测试。

## 真实 PhysX 和训练证据

三机集成：

```bash
.../python.sh scripts/test_psto_integration.py --steps 2 --headless \
  --out results/psto_p0/integration_smoke
```

结果：`pass`，20/20 连续策略步，84-D centralized state、2-D Beta action、无 NaN、actor 未接收未来标签。CSV 见 `results/psto_p0/integration_20/integration.csv`。

10 s 悬停和 30 s 独立控制：

```bash
.../python.sh scripts/test_psto_physics.py --headless \
  --out results/psto_p0/physics_long
```

结果：`pass`，真实 240 Hz PhysX、30 Hz 策略和三架独立 RotorFlight，无非有限状态；轨迹采样写入 `physics.csv`。

两套配置短训练：

```bash
.../python.sh scripts/train_psto.py --config configs/psto/trace_native.yaml \
  --headless --total_frames 4 --rollout_steps 4 --training_epoch_num 1 \
  --num_minibatches 1 --checkpoint results/psto_p0/smoke/native/checkpoint.pt
.../python.sh scripts/train_psto.py --config configs/psto/official_reproduction.yaml \
  --headless --total_frames 2 --rollout_steps 2 --training_epoch_num 1 \
  --num_minibatches 1 --checkpoint results/psto_p0/smoke/official/checkpoint.pt
```

两次均为真实 Isaac Sim 6.0.1 `status=pass`，Actor/Critic gradient 和 loss 有限，checkpoint 已保存。native 的 `value_loss=0.6146`、official 的 `value_loss=0.6239` 仅是链路 smoke 数值，不代表收敛。LSTM 在少于 20 帧时没有有效监督样本，因此这两个极短 smoke 的 `sl_loss=0` 是预期行为；需要更长短训才能验收 LSTM 更新和 resume。

随后使用 native checkpoint 执行了 `--resume --total_frames 1`，实际恢复 optimizer/model state 并完成一次更新，结果 `status=pass`（`value_loss=0.2795`，gradient finite）。

为跨过 10 帧历史和 10 帧延迟标签窗口，又执行了 native `total_frames=20` smoke；结果 `status=pass`，`sl_loss=0.01941`、`lstm_grad_norm=0.05867`，说明真实环境中的 LSTM 监督更新确实发生。

使用 `--max_episode_steps 1 --total_frames 2` 的真实 PhysX timeout smoke 记录了 `terminated_steps=0`、`truncated_steps=2`，并完成了有限的 PPO 更新；结果在 `results/psto_p0/smoke/timeout/`。

代码随后提交为本地 commit `90abbb6`（未 Push），并在该 commit 上再次完成了 1 帧 native PhysX PPO smoke；结果见 `results/psto_p0/post_commit_smoke/`，JSON 中记录了该 SHA 和 Isaac Sim 6.0.1 build。

评估入口也已改为读取同一 YAML 时间尺度、按 episode 使用独立 seed，并在输出旁保存配置快照。当前 post-commit checkpoint 的 1 episode/2 step 评估实际返回 `status=pass`、visibility ratio `0.5`、无终止事件；由于只有 2 步，ADE/FDE 尚无有效延迟窗口，这只是入口 smoke。

## 十 Episode 实际结果

```bash
.../python.sh scripts/test_psto_episodes.py --episodes 10 --steps 280 --headless \
  --out results/psto_p0/episodes_10
```

结果：`status=pass`，10/10 个 episode、280 步上限、无 NaN/Inf，轨迹写入 `trajectories.csv`，逐 episode 指标写入 `episodes.csv`。所有 episode 在 55–76 步间触发真实碰撞终止；没有把提前碰撞重新包装成 timeout。

## 尚未完成

- 没有进行长时间 MAPPO 收敛、多 seed 论文指标或正式模型性能声明。
- 固定 cuboid 障碍场是物理和 LiDAR 验收环境，尚未声称与上游 TerrainImporter 的字节级地形生成一致。
- 当前物理适配器仍是单 World、`num_envs=1`。

因此当前不能把 P0 标记为 COMPLETE，也不能把 smoke 训练当成正式复现实验。
