#!/usr/bin/env bash
# 训练 MARVEL。超参在 upstream/MARVEL/parameter.py 里改。
#   --from-pretrained  用 load_model/MARVEL/checkpoint.pth 续训
#                      （parameter.py 的 LOAD_MODEL=True 读的是 model/<FOLDER_NAME>/checkpoint.pth）
# 注意 MINIMUM_BUFFER_SIZE 必须 >= BATCH_SIZE，否则 random.sample 直接报错。
set -euo pipefail
REPO="$(cd "$(dirname "$0")/../upstream/MARVEL" && pwd)"
if [[ "${1:-}" == "--from-pretrained" ]]; then
  mkdir -p "$REPO/model/marvel"
  cp -n "$REPO/load_model/MARVEL/checkpoint.pth" "$REPO/model/marvel/checkpoint.pth"
  echo "已把预训练权重放到 model/marvel/（LOAD_MODEL=True 时从此续训）"
  shift
fi
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate trace_explorer
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
cd "$REPO"
echo "tensorboard --logdir $REPO/train"
exec python -u driver.py "$@"
