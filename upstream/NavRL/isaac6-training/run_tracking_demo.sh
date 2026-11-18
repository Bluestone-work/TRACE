#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ISAACSIM_ROOT="${ISAACSIM_ROOT:-/home/wj/isaacsim}"
export PYTHONPATH="${ISAACSIM_ROOT}/torch_site:${SCRIPT_DIR}/../quick-demos:${SCRIPT_DIR}:${PYTHONPATH:-}"
if [[ -z "${DISPLAY:-}" && -S /tmp/.X11-unix/X3 ]]; then
    export DISPLAY=:3
fi
export XAUTHORITY="${XAUTHORITY:-/home/wj/.Xauthority}"
exec "${ISAACSIM_ROOT}/python.sh" "${SCRIPT_DIR}/run_tracking_demo.py" "$@"
