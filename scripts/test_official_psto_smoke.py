#!/usr/bin/env python3
"""Probe the independent HITSZ-MAS PSTO runtime without pretending it is TRACE.

The official release depends on OmniDrones and Isaac Sim 4.1.0.  This script
records an explicit BLOCKED result when those dependencies are absent; it does
not silently fall back to the Isaac Sim 6 adapter.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import sys
from pathlib import Path


OFFICIAL_COMMIT = "764e2a156f4f74fc3d24a25dfaa6871c57ec7944"


def probe_module(name: str) -> str | None:
    try:
        spec = importlib.util.find_spec(name)
    except Exception as exc:
        return f"probe error: {type(exc).__name__}: {exc}"
    return None if spec is not None else "not installed"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--official-root", default="upstream/psto-aav-pursuit-official")
    parser.add_argument("--out", default="results/psto_official_smoke/official_smoke.json")
    args = parser.parse_args()
    root = Path(args.official_root).resolve()
    required_files = ["env.py", "mappo.py", "lstm.py", "nav_utils.py", "models/model.py"]
    missing_files = [name for name in required_files if not (root / name).is_file()]
    missing_modules = {
        name: probe_module(name)
        for name in ("omni_drones", "torchrl", "tensordict", "omni.isaac.lab")
    }
    missing_modules = {name: reason for name, reason in missing_modules.items() if reason}
    result = {
        "status": "pass" if not missing_files and not missing_modules else "blocked",
        "official_commit": OFFICIAL_COMMIT,
        "official_root": str(root),
        "required_files_missing": missing_files,
        "required_modules_missing": missing_modules,
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "requested_isaac_sim": "4.1.0",
            "requested_omnidrones": "external editable install",
            "requested_torchrl": "0.3.1",
        },
        "next_steps": [
            "Create a Python 3.10 environment with Isaac Sim 4.1.0.",
            "Install OmniDrones from its documented repository in editable mode.",
            "Install torchrl==0.3.1 and tensordict compatible with that OmniDrones release.",
            "Run this probe again, then launch the official train.py with a 10-frame smoke override.",
        ],
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
