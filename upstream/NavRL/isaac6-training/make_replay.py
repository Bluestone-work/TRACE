"""Convert recorded Isaac Sim viewport frames to a portable GIF (Pillow)."""

import argparse
from pathlib import Path

from PIL import Image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("frames", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--interval-ms", type=int, default=200)
    args = parser.parse_args()
    paths = sorted(args.frames.glob("frame_*.png"))
    if not paths:
        parser.error(f"no recorded PNG frames found in {args.frames}")
    frames = []
    for path in paths:
        with Image.open(path) as frame:
            frame = frame.convert("RGB")
            frame.thumbnail((960, 540))
            frames.append(frame)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(args.output, save_all=True, append_images=frames[1:],
                   duration=args.interval_ms, loop=0)
    print(f"Saved {len(frames)} rendered frames: {args.output}")


if __name__ == "__main__":
    main()
