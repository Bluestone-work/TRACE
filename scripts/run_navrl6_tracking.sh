#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec "$ROOT/upstream/NavRL/isaac6-training/run_tracking_demo.sh" "$@"
