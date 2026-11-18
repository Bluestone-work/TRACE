#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/wj/miniconda3/envs/navrl/bin/python}"
exec "$PYTHON_BIN" "$ROOT/upstream/NavRL/isaac6-training/make_replay.py" "$@"
