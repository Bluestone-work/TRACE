#!/usr/bin/env bash
# 评测 MARVEL 预训练策略（baseline 指标）
# 测试配置在 upstream/MARVEL/test_parameter.py 里改（上游设计如此）
set -euo pipefail
REPO="$(cd "$(dirname "$0")/../upstream/MARVEL" && pwd)"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate trace_explorer
# 必须限制线程：ray worker 各自起满线程会把 24 核跑崩，实测吞吐差一个数量级
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
cd "$REPO"
exec python -u test_driver.py "$@"
