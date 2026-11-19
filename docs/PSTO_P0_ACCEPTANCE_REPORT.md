# PSTO P0 验收报告

本报告记录 TRACE PSTO 迁移在本地 Isaac Sim 6.0.1 的验收状态。报告只列出已经实际执行的命令和产物；没有运行的长实验保持 `BLOCKED`，不会用 smoke 结果代替收敛证据。

## 当前结论

P0 当前为 **BLOCKED**。时间尺度、种子接口、配置追溯、GAE/timeout 语义、三机 PhysX、两套短训练和 LiDAR 性能测量已通过本地验证。固定动作环境审计的执行状态为 `pass`，但它不评价策略；更新后的十个随机 Episode 中 8 次为 pursuer-obstacle collision、1 次 capture、1 次 timeout。MAPPO 只完成了短 checkpoint smoke，尚未进行长时间收敛，因此暂不建议启动正式长训练。

## P0-1 时间尺度

`configs/psto/official_reproduction.yaml` 来自上游 `sim.dt=0.01`、`frame_skip=10`，因此 `policy_dt=0.1`、10 Hz、280 步为 28 s。`configs/psto/trace_native.yaml` 保留 Isaac Sim 6 适配器的 240 Hz PhysX 与 30 Hz 策略，8 个物理步对应一个策略步，280 步为 9.333333 s。环境将真实观测时间间隔传给 LSTM，并在配置快照中记录 frame skip 和 episode duration。

状态：**PASS（配置和启动链路）**。两套配置均已在真实 Isaac Sim 6.0.1 中启动并完成短更新。

## P0-2 随机种子和场景

训练入口同时初始化 Python、NumPy、Torch CPU 和 CUDA（可用时），并将 seed 传入 `PSTOPursuitEnv`/`PSTOPursuitScene`。Scene 使用独立 NumPy RNG；`reset(seed=...)` 可以重建 episode 初始状态，`randomize_resets` 控制无人机位置扰动，`randomize_obstacles` 控制障碍物布局扰动。默认配置关闭随机化以保持官方初始排列，评估脚本显式打开并使用 `seed + episode`。

状态：**PASS（接口和真实场景已复核）**。`test_psto_episodes.py` 现在同时比较三机位置和 `scene.obstacle_positions`；实际 seed-contract smoke 为 `status=pass`。障碍物中心高度统一为 `scale_z / 2`，底部贴地，不再把高度误用为中心 Z。

## P0-3 配置和实验记录

唯一入口为 `scripts/train_psto.py --config ...`。YAML 中的学习率、gamma、GAE、PPO clipping、熵系数、ValueNorm、LSTM 监督频率、时间尺度和输出目录会传入构造器。每次运行在 checkpoint 目录保存 `config_snapshot.yaml` 和 JSON 元数据，其中包含 Git SHA、Isaac Sim build、Python、Torch、CUDA、seed、时间尺度、帧数、更新次数、loss 和 gradient norm。默认 checkpoint 目录已有文件时会创建 `_runN`，避免覆盖旧实验。

状态：**PASS**。checkpoint 现在额外保存 Python/NumPy/Torch/CUDA RNG、MAPPO update counter、LSTM minibatch counter、累计 frames/updates 和可恢复的环境状态；NumPy RNG 已转换为纯列表格式，避免跨 NumPy 版本依赖私有模块路径。

LSTM 监督更新已改为 minibatch 级计数：当 `lstm_train_counter % lstm_update_frequency == 0` 时执行一次，并且计数器跨 PPO epoch 和 checkpoint 保存。4 epoch、2 minibatch、frequency=3 的单元测试实际得到 8 个 minibatch、3 次 LSTM update。

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

`PYTHONPATH=$PWD /home/wj/miniconda3/envs/trace_isaac/bin/pytest -q tests/psto` 的实际结果为 **19 passed, 1 warning**。其中包含连续状态、terminal bootstrap、timeout bootstrap/递推屏蔽、84-D state、遮挡历史、奖励逐项测试和 minibatch 级 LSTM 频率测试。

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

在修复后的代码上重新执行 native 20-frame smoke，得到 `sl_loss=0.01941`、`lstm_updates=1`、`minibatches=1`；随后使用同一 checkpoint 将累计 frames 从 20 恢复到 21，`--resume` 实际完成第二次更新并恢复了 RNG 与环境状态。

在 commit `4a1fda9` 上再次执行了相同的真实 Isaac Sim 6.0.1 验证：`results/psto_p0/final_validation/post_commit20/checkpoint.json` 记录 20 帧、1 次 PPO/LSTM 更新（`sl_loss=0.0194136`），随后 `--resume --total_frames 21` 记录累计 frames `20 -> 21`、updates `1 -> 2`。`.pt` 权重留在本地，仓库只提交 JSON 和配置快照。

使用 `--max_episode_steps 1 --total_frames 2` 的真实 PhysX timeout smoke 记录了 `terminated_steps=0`、`truncated_steps=2`，并完成了有限的 PPO 更新；结果在 `results/psto_p0/smoke/timeout/`。

此前的基准 commit `90abbb6` 已由后续验证 commit 推送到远程；其 1 帧 native PhysX PPO smoke 结果见 `results/psto_p0/post_commit_smoke/`，JSON 中记录了 Isaac Sim 6.0.1 build。

评估入口也已改为读取同一 YAML 时间尺度、按 episode 使用独立 seed，并在输出旁保存配置快照。当前 post-commit checkpoint 的 1 episode/2 step 评估实际返回 `status=pass`、visibility ratio `0.5`、无终止事件；由于只有 2 步，ADE/FDE 尚无有效延迟窗口，这只是入口 smoke。

## 十 Episode 实际结果

```bash
.../python.sh scripts/test_psto_episodes.py --episodes 10 --steps 280 --headless \
  --out results/psto_p0/episodes_10
```

结果：脚本输出 `status=pass`，但该字段只表示执行和数据记录完成；固定动作模式标为 `task_performance_status=not_evaluated`。更新后的 10/10 episode 无 NaN/Inf，平均 147.5 步：8 次 `pursuer_obstacle_collision`，1 次 capture，1 次 timeout；没有 pursuer-pursuer 或 evader-obstacle collision。逐类字段已写入 `episodes.csv`，障碍物布局写入每个 episode 的 `obstacle_layout`。

MAPPO 控制器也已实际接入脚本：

```bash
.../python.sh scripts/test_psto_episodes.py --episodes 1 --steps 20 \
  --controller mappo --checkpoint <checkpoint.pt> --headless
```

该 smoke 输出 `test_scope=MAPPO policy rollout`、`task_performance_status=evaluated`；它仍然不是收敛性能声明。

轨迹预览 GIF 已保存为 `results/psto_p0/episodes_10_collision_audit/psto_episode0.gif` 和 `results/psto_p0/mappo_episode_smoke/psto_mappo_smoke.gif`。这是由真实 PhysX 状态 CSV 生成的俯视/高度轨迹动画，不代表 Isaac Sim 相机渲染质量。

## 尚未完成

- 没有进行长时间 MAPPO 收敛、多 seed 论文指标或正式模型性能声明。
- 固定 cuboid 障碍场是物理和 LiDAR 验收环境，尚未声称与上游 TerrainImporter 的字节级地形生成一致。
- 当前物理适配器仍是单 World、`num_envs=1`。
- 官方 `9.8e9` frames/`1024` envs 不能直接用于当前单环境适配器；实测环境约 5.5 step/s，正式训练必须使用独立的 TRACE-native frames 预算，不能宣称与官方吞吐等价。

因此当前不能把 P0 标记为 COMPLETE，也不能把 smoke 训练当成正式复现实验。
