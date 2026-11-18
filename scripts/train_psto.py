#!/usr/bin/env python3
"""PSTO training entry point.

Use `--smoke` for a dependency-only forward check. Physical training requires launching
this file with the TRACE Isaac environment and is intentionally rejected until a real
rollout storage/checkpoint is supplied, preventing accidental claims of training.
"""
import argparse
from scripts.smoke_psto import main as smoke
p=argparse.ArgumentParser(); p.add_argument('--smoke',action='store_true'); args=p.parse_args()
if args.smoke: smoke()
else: raise SystemExit('PSTO physical MAPPO training entry is staged: run --smoke now; rollout/checkpoint integration is the next phase.')
