#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -z "${DISPLAY:-}" ]]; then
    # This workstation also has wj's persistent VNC desktop. Unlike an
    # inactive physical seat, it continues to render after a user switch.
    if [[ -S /tmp/.X11-unix/X3 ]]; then
        export DISPLAY=:3
    else
        export DISPLAY=:0
    fi
fi
export XAUTHORITY="${XAUTHORITY:-/home/wj/.Xauthority}"
echo "Opening NavRL on DISPLAY=${DISPLAY}"
exec "${SCRIPT_DIR}/run_torch_policy.sh" --gui --loop --realtime --steps 120 "$@"
