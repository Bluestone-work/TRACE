#!/usr/bin/env bash
# 环境自检：跑 2 个评测 episode，结束后把 test_parameter.py 还原，仓库保持干净。
set -euo pipefail
REPO="$(cd "$(dirname "$0")/../upstream/MARVEL" && pwd)"
cd "$REPO"
BAK="$(mktemp)"; cp test_parameter.py "$BAK"
restore() { cp "$BAK" test_parameter.py; rm -f "$BAK"; rm -rf utils/__pycache__/*cpython-311* 2>/dev/null || true; }
trap restore EXIT

sed -i 's/^NUM_TEST = .*/NUM_TEST = 2/; s/^NUM_META_AGENT = .*/NUM_META_AGENT = 2/; s/^SAVE_GIFS = .*/SAVE_GIFS = False/' test_parameter.py
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate trace_explorer
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
python -u test_driver.py 2>&1 | grep -E '^\|#|Total time'
echo "自检通过：策略加载、图构建、注意力推理、多机仿真、指标统计全链路正常"
