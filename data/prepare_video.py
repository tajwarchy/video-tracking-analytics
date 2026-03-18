"""
data/prepare_video.py
─────────────────────
Utilities to download, validate, and preprocess sample videos
for the Video Tracking Analytics pipeline.

Usage:
    # Download built-in samples
    python data/prepare_video.py --download

    # Validate + resize an existing video
    python data/prepare_video.py --input path/to/video.mp4 --width 1280 --height 720

    # Inspect video metadata
    python data/prepare_video.py --info path/to/video.mp4
"""

import argparse
import os
import sys
import urllib.request
from pathlib import Path

import cv2
import numpy as np

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
SAMPLE_DIR = ROOT / "data" / "sample_videos"

# ── Freely-licensed sample video URLs ─────────────────────────────────────────
# These are Pexels / Pixabay MP4 direct links (no login required)
SAMPLE_VIDEOS = {
    "pedestrians.mp4": (
        "https://www.pexels.com/download/video/854671/",
        "Street pedestrians – free Pexels video",
    ),
    "traffic.mp4": (
        "https://www.pexels.com/download/video/2103099/",
        "Road traffic – free Pexels video",
    ),
}

# ── Fallback: generate a synthetic test video ──────────────────────────────────
def generate_synthetic_video(
    out_path: Path,
    width: int = 1280,
    height: int = 720,
    fps: int = 30,
    duration_s: int = 10,
    n_objects: int = 5,
) -> None:
    """
    Creates a simple synthetic video with moving coloured rectangles.
    Useful as a guaranteed-available smoke-test input when no real video
    is present.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))

    total_frames = fps * duration_s
    rng = np.random.default_rng(42)

    # Initialise object states: [cx, cy, vx, vy, w, h, color]
    objects = []
    for _ in range(n_objects):
        cx = int(rng.uniform(100, width - 100))
        cy = int(rng.uniform(100, height - 100))
        vx = float(rng.uniform(2, 6)) * rng.choice([-1, 1])
        vy = float(rng.uniform(1, 4)) * rng.choice([-1, 1])
        w  = int(rng.uniform(40, 80))
        h  = int(rng.uniform(80, 140))
        color = tuple(int(c) for c in rng.integers(80, 255, 3))
        objects.append([cx, cy, vx, vy, w, h, color])

    for _ in range(total_frames):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:] = (30, 30, 30)  # dark grey background

        for obj in objects:
            cx, cy, vx, vy, w, h, color = obj
            cx += vx; cy += vy

            # Bounce off walls
            if cx - w // 2 < 0 or cx + w // 2 > width:
                vx = -vx
                cx += vx * 2
            if cy - h // 2 < 0 or cy + h // 2 > height:
                vy = -vy
                cy += vy * 2

            obj[0], obj[1], obj[2], obj[3] = cx, cy, vx, vy

            x1 = max(0, int(cx - w // 2))
            y1 = max(0, int(cy - h // 2))
            x2 = min(width,  int(cx + w // 2))
            y2 = min(height, int(cy + h // 2))
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, -1)

        writer.write(frame)

    writer.release()
    print(f"[prepare_video] Synthetic video saved → {out_path}")


# ── Download helpers ───────────────────────────────────────────────────────────
def _progress_hook(block_num: int, block_size: int, total_size: int) -> None:
    downloaded = block_num * block_size
    if total_size > 0:
        pct = min(downloaded / total_size * 100, 100)
        bar = int(pct // 2)
        print(f"\r  [{'█' * bar}{'░' * (50 - bar)}] {pct:5.1f}%", end="", flush=True)


def download_samples(out_dir: Path = SAMPLE_DIR) -> None:
    """
    Attempt to download free sample videos. Falls back to the synthetic
    video generator if any download fails (Pexels may require a browser
    session for some links).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    for filename, (url, description) in SAMPLE_VIDEOS.items():
        dest = out_dir / filename
        if dest.exists():
            print(f"[prepare_video] Already exists, skipping: {dest.name}")
            continue
        print(f"[prepare_video] Downloading {description} …")
        try:
            urllib.request.urlretrieve(url, dest, reporthook=_progress_hook)
            print()  # newline after progress bar
            validate_video(dest)
        except Exception as e:
            print(f"\n[prepare_video] Download failed ({e}). Generating synthetic fallback …")
            dest.unlink(missing_ok=True)
            generate_synthetic_video(out_dir / "synthetic_test.mp4")
            break

    # Always ensure at least one synthetic video exists for quick testing
    synth = out_dir / "synthetic_test.mp4"
    if not synth.exists():
        generate_synthetic_video(synth)


# ── Validation ────────────────────────────────────────────────────────────────
def validate_video(path: Path) -> dict:
    """
    Open a video file and return its metadata. Raises RuntimeError if
    the file cannot be read.
    """
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")

    info = {
        "path":        str(path),
        "width":       int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height":      int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps":         cap.get(cv2.CAP_PROP_FPS),
        "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "duration_s":  int(cap.get(cv2.CAP_PROP_FRAME_COUNT) / max(cap.get(cv2.CAP_PROP_FPS), 1)),
        "codec":       int(cap.get(cv2.CAP_PROP_FOURCC)),
    }
    cap.release()

    print(
        f"[prepare_video] ✓ {path.name}  "
        f"{info['width']}×{info['height']}  "
        f"{info['fps']:.1f}fps  "
        f"{info['frame_count']} frames  "
        f"({info['duration_s']}s)"
    )
    return info


# ── Resize / preprocess ───────────────────────────────────────────────────────
def resize_video(
    src: Path,
    dst: Path,
    width: int,
    height: int,
    fps: float | None = None,
) -> None:
    """
    Resize a video to (width × height). Optionally re-encode at a new FPS.
    Output is always MP4 (mp4v codec).
    """
    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open: {src}")

    out_fps = fps or cap.get(cv2.CAP_PROP_FPS)
    dst.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(dst), fourcc, out_fps, (width, height))

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[prepare_video] Resizing {src.name} → {width}×{height} …")

    for i in range(total):
        ok, frame = cap.read()
        if not ok:
            break
        resized = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
        writer.write(resized)
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{total} frames …", end="\r")

    cap.release()
    writer.release()
    print(f"\n[prepare_video] Saved → {dst}")


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare sample videos for the tracking pipeline."
    )
    parser.add_argument("--download", action="store_true",
                        help="Download (or generate) sample videos into data/sample_videos/")
    parser.add_argument("--synthetic", action="store_true",
                        help="Generate a synthetic test video only")
    parser.add_argument("--info",   type=str, metavar="VIDEO",
                        help="Print metadata for a video file")
    parser.add_argument("--input",  type=str, metavar="VIDEO",
                        help="Source video for resize/validate")
    parser.add_argument("--output", type=str, metavar="OUTPUT",
                        help="Output path for resized video")
    parser.add_argument("--width",  type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps",    type=float, default=None,
                        help="Re-encode at this FPS (default: keep original)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.info:
        validate_video(Path(args.info))

    elif args.download:
        download_samples()

    elif args.synthetic:
        out = SAMPLE_DIR / "synthetic_test.mp4"
        generate_synthetic_video(out)

    elif args.input:
        src = Path(args.input)
        dst = Path(args.output) if args.output else src.with_stem(src.stem + "_resized")
        validate_video(src)
        resize_video(src, dst, args.width, args.height, args.fps)
        validate_video(dst)

    else:
        print("[prepare_video] No action specified. Use --help for options.")
        print("[prepare_video] Tip: run with --download to get started quickly.")
        sys.exit(0)


if __name__ == "__main__":
    main()