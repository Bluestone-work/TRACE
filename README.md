# TRACE Baseline Foundation

这是从 `/home/wj/桌面/无人机/TRACE_baseline` 抽出的实验基座归档。它把当前已经搭好的代码按“自研算法、Isaac Lab 3D、NavRL 6.0.1 动力学、多机视觉 Demo、参考基线、验证结果”重新组织，原目录保持不变。

详细模块关系见 [`docs/BASELINE_ARCHITECTURE.md`](docs/BASELINE_ARCHITECTURE.md)。

## 内容

```text
TRACE_baseline_foundation/
├── trace/                         TRACE 三态占据图、frontier、DirectMARLEnv
├── tests/isaac_tests/             00–15 全部单元/集成/消融测试
├── scripts/                       训练、评测、自检和 NavRL 便捷入口
├── upstream/
│   ├── NavRL/quick-demos/         官方 NavRL quick-demos 依赖
│   ├── NavRL/isaac6-training/     Isaac Sim 6.0.1 旋翼动力学与双机森林跟踪
│   ├── MARVEL/                    2D 多机探索基线
│   ├── IR2/                       通信受限多机器人参考实现
│   └── IsaacLab/                  Isaac Lab 源码/配置快照
├── docs/                          迁移、动力学、跟踪和研究综述
├── results/                       紧凑 JSON 验证结果
├── requirements_locked.txt
└── ARCHIVE_MANIFEST.json
```

## 直接运行当前双机森林基线

需要本机已安装 `/home/wj/isaacsim` 6.0.1，并使用现有 VNC display `:3`：

```bash
cd /home/wj/桌面/无人机/TRACE_baseline_foundation
DISPLAY=:3 ./scripts/run_navrl6_tracking.sh \
  --gui --scene forest --seconds 20 --seed 7 --speed-limit 2 \
  --record-dir artifacts/foundation_forest_frames \
  --record-every 3 \
  --report artifacts/foundation_forest_report.json
```

生成 GIF：

```bash
./scripts/make_navrl6_gif.sh \
  artifacts/foundation_forest_frames \
  artifacts/foundation_forest.gif --interval-ms 100
```

## 运行 NavRL 旋翼策略

```bash
cd /home/wj/桌面/无人机/TRACE_baseline_foundation
./scripts/run_navrl6_policy.sh --dynamics rotor --steps 360 \
  --report artifacts/foundation_navrl6_policy.json
```

## 检查归档完整性

```bash
./scripts/check_archive.sh
```

这项检查只验证归档文件存在、Python 可编译、shell 语法正确，不会自动启动 Isaac Sim。完整动力学和森林视觉验收报告见 `results/navrl6/`。

## 运行边界

Isaac Sim、CUDA、Torch 和 Isaac Lab 运行时仍使用机器上的安装；归档里的 `upstream/IsaacLab` 是源码与配置快照，不复制 Isaac Sim 安装本体。生成的 PNG/GIF 帧序列和训练日志没有复制到这里，防止代码基座被实验产物淹没；它们的原始位置见 `results/README.md`。

## 版本来源

创建日期：2026-11-18。原始工程：`/home/wj/桌面/无人机/TRACE_baseline`。具体文件清单和源仓库 commit 见 `ARCHIVE_MANIFEST.json`。
