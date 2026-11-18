#!/usr/bin/env bash
# 重建 trace_explorer conda 环境（已验证可用的版本组合）
set -euo pipefail
ENV_NAME="${1:-trace_explorer}"
MIRROR="https://pypi.tuna.tsinghua.edu.cn/simple"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda create -y -n "$ENV_NAME" python=3.11
conda activate "$ENV_NAME"

# torch 2.5.1 的 PyPI 默认构建即 cu124，匹配上游 marvel.yml
pip install -i "$MIRROR" torch==2.5.1 torchvision==0.20.1
pip install -i "$MIRROR" \
  "numpy==1.26.4" "scikit-image==0.25.0" "scikit-learn==1.6.1" "scipy==1.15.1" \
  "shapely==2.0.4" "matplotlib==3.10.0" "imageio==2.37.0" "ray==2.44.0" \
  "pandas==2.2.3" tensorboard tensorboardX wandb ffmpeg-python python-dotenv

python - <<'PY'
import torch, numpy
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), "ngpu", torch.cuda.device_count())
print("numpy", numpy.__version__)
PY
echo "环境 $ENV_NAME 就绪"
