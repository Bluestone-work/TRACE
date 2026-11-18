#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python3 -m compileall -q "$ROOT/trace" "$ROOT/tests" "$ROOT/scripts" "$ROOT/upstream/NavRL/isaac6-training"
for f in "$ROOT"/scripts/*.sh "$ROOT"/upstream/NavRL/isaac6-training/*.sh; do bash -n "$f"; done
python3 - <<PY2
from pathlib import Path
root = Path(r"$ROOT")
required = [
    root / "trace/envs/trace_explore_env.py",
    root / "scripts/train_trace_marl.py",
    root / "upstream/NavRL/isaac6-training/run_tracking_demo.py",
    root / "upstream/NavRL/isaac6-training/forest_scene.py",
    root / "upstream/MARVEL/driver.py",
    root / "upstream/IR2/driver.py",
]
missing = [str(p) for p in required if not p.is_file()]
if missing: raise SystemExit("missing: " + ", ".join(missing))
print("[OK] archive structure and Python/shell syntax")
PY2
