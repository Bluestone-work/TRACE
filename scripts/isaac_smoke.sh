#!/usr/bin/env bash
# Isaac 侧环境自检：占据栅格单测 → 启动/物理 → RL 训练 → LiDAR → 占据栅格集成
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LAB="$ROOT/upstream/IsaacLab"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate trace_isaac
cd "$LAB"
PAT='^\[OK\]|^\[DONE\]'

echo "=========== [0/7] 占据栅格数学单元测试（不启动 Isaac）==========="
timeout 180 python -u "$ROOT/isaac_tests/00_occupancy_unit.py" 2>&1 | grep -E "$PAT"

echo "=========== [1/7] 启动 + 物理步进 ==========="
timeout 300 python -u "$ROOT/isaac_tests/01_launch_physics.py" --headless 2>&1 | grep -E "$PAT"

echo "=========== [2/7] RL 训练链路（skrl PPO）==========="
timeout 600 python -u scripts/reinforcement_learning/skrl/train.py \
  --task Isaac-Quadcopter-Direct-v0 --headless --num_envs 256 --max_iterations 10 2>&1 \
  | grep -E 'Number of environments|Training time'

echo "=========== [3/7] LiDAR 光线投射 ==========="
timeout 300 python -u "$ROOT/isaac_tests/02_lidar_raycast.py" --headless 2>&1 | grep -E "$PAT"

echo "=========== [4/7] 占据栅格集成（真实 LiDAR 驱动）==========="
timeout 900 python -u "$ROOT/isaac_tests/03_occupancy_map.py" --headless --steps 160 2>&1 \
  | grep -E "$PAT|^ +[0-9]+ \|"

echo "=========== [5/7] frontier 单元测试（不启动 Isaac）==========="
timeout 180 python -u "$ROOT/isaac_tests/04_frontier_unit.py" 2>&1 | grep -E "$PAT"

echo "=========== [6/7] frontier 集成（真实 LiDAR + 精确 CCL）==========="
timeout 900 python -u "$ROOT/isaac_tests/05_frontier_integration.py" --headless --steps 240 2>&1 \
  | grep -E "$PAT|^ +[0-9]+ \|"

echo "=========== [7/7] MARL 环境端到端 ==========="
timeout 600 python -u "$ROOT/isaac_tests/06_marl_env.py" --headless --num_envs 2 --steps 40 2>&1 \
  | grep -E "$PAT|^ +[0-9]+ \|"

echo "=========== 自检结束 ==========="
