"""
benchmark.py
─────────────
Measures pipeline performance on M1 MPS vs CPU and computes
MOTA / MOTP approximations on the synthetic test video.

Usage
─────
    python benchmark.py
    python benchmark.py --frames 150 --runs 3
    python benchmark.py --device cpu        # force CPU only
    python benchmark.py --skip-tracking     # detector speed only
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml

# ── Helpers ───────────────────────────────────────────────────────────────────

def _mean(vals: list[float]) -> float:
    return sum(vals) / len(vals) if vals else 0.0


def _table(rows: list[tuple], headers: list[str]) -> str:
    col_w = [max(len(h), max((len(str(r[i])) for r in rows), default=0))
             for i, h in enumerate(headers)]
    sep  = "─" * (sum(col_w) + 3 * len(headers) + 1)
    fmt  = "│ " + " │ ".join(f"{{:<{w}}}" for w in col_w) + " │"
    lines = [sep, fmt.format(*headers), sep]
    for row in rows:
        lines.append(fmt.format(*[str(v) for v in row]))
    lines.append(sep)
    return "\n".join(lines)


# ── Per-device benchmark ──────────────────────────────────────────────────────

def bench_device(
    cfg       : dict,
    device    : str,
    video_path: str,
    n_frames  : int,
    run_tracking: bool,
) -> dict:
    """Run the pipeline on `device` and return timing stats."""
    from tracking.detector         import build_detector
    from tracking.tracker          import build_tracker
    from tracking.motion_estimator import build_motion_estimator
    from tracking.trajectory       import build_trajectory_store

    cfg = dict(cfg)                          # shallow copy
    cfg["model"] = dict(cfg["model"])
    cfg["model"]["device"] = device

    detector = build_detector(cfg)
    tracker  = build_tracker(cfg)  if run_tracking else None
    motion   = build_motion_estimator(cfg) if run_tracking else None
    traj     = build_trajectory_store(cfg) if run_tracking else None

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open: {video_path}")

    out_w = cfg["video"].get("resize_width",  1280)
    out_h = cfg["video"].get("resize_height",  720)

    detect_times  : list[float] = []
    track_times   : list[float] = []
    total_dets    : list[int]   = []
    total_tracks  : list[int]   = []

    for _ in range(n_frames):
        ok, frame = cap.read()
        if not ok:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = cap.read()
            if not ok:
                break

        frame = cv2.resize(frame, (out_w, out_h))

        # ── Detection ─────────────────────────────────────────────────────────
        t0 = time.perf_counter()
        dets = detector.detect(frame)
        detect_times.append(time.perf_counter() - t0)
        total_dets.append(len(dets))

        # ── Tracking ──────────────────────────────────────────────────────────
        if run_tracking and tracker:
            t1 = time.perf_counter()
            objects = tracker.update(dets, frame)
            objects = motion.update(objects)
            traj.update(objects, frame_idx=len(detect_times))
            track_times.append(time.perf_counter() - t1)
            total_tracks.append(len(objects))

    cap.release()

    avg_det   = _mean(detect_times)
    avg_track = _mean(track_times) if track_times else 0.0
    avg_total = avg_det + avg_track
    fps       = 1.0 / avg_total if avg_total > 0 else 0.0

    return {
        "device"      : device.upper(),
        "n_frames"    : len(detect_times),
        "avg_det_ms"  : round(avg_det   * 1000, 1),
        "avg_track_ms": round(avg_track * 1000, 1),
        "avg_total_ms": round(avg_total * 1000, 1),
        "fps"         : round(fps, 1),
        "avg_dets"    : round(_mean(total_dets), 1),
        "avg_tracks"  : round(_mean(total_tracks), 1) if total_tracks else "—",
    }


# ── MOTA / MOTP approximation ─────────────────────────────────────────────────

def compute_mot_metrics(
    cfg       : dict,
    video_path: str,
    n_frames  : int,
    iou_thresh: float = 0.5,
) -> dict:
    """
    Approximate MOTA and MOTP on the test video.

    Since we have no ground-truth annotations, we use a self-consistency
    approach: detections from the first pass act as pseudo-GT, and we
    measure how consistently ByteTrack maintains IDs across frames.

    Real MOTA/MOTP against MOT17 ground truth would require the official
    MOT evaluation toolkit — this gives a useful relative benchmark.
    """
    from tracking.detector         import build_detector
    from tracking.tracker          import build_tracker
    from tracking.trajectory       import build_trajectory_store

    detector = build_detector(cfg)
    tracker  = build_tracker(cfg)
    traj     = build_trajectory_store(cfg)

    cap = cv2.VideoCapture(video_path)
    out_w = cfg["video"].get("resize_width",  1280)
    out_h = cfg["video"].get("resize_height",  720)

    id_switches   = 0
    total_matches = 0
    total_iou     = 0.0
    prev_ids: set[int] = set()

    all_track_ids: dict[int, list] = {}   # track_id → list of frame indices

    for fi in range(n_frames):
        ok, frame = cap.read()
        if not ok:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = cap.read()
            if not ok:
                break

        frame   = cv2.resize(frame, (out_w, out_h))
        dets    = detector.detect(frame)
        objects = tracker.update(dets, frame)
        traj.update(objects, fi)

        curr_ids = {o.track_id for o in objects}

        # ID switches: tracks that appeared, disappeared, then reappeared
        # approximated as IDs in curr but not in prev that were seen before
        seen_before = set(all_track_ids.keys())
        new_ids     = curr_ids - prev_ids
        switches    = new_ids & seen_before
        id_switches += len(switches)

        for obj in objects:
            all_track_ids.setdefault(obj.track_id, []).append(fi)
            total_matches += 1
            # Self-IOU: box area / max-possible area (proxy for localisation)
            x1, y1, x2, y2 = obj.bbox
            box_area  = max(0, x2 - x1) * max(0, y2 - y1)
            frame_area = out_w * out_h
            total_iou += min(box_area / frame_area * 20, 1.0)   # scaled proxy

        prev_ids = curr_ids

    cap.release()

    unique_tracks = len(all_track_ids)
    # Track durations — long tracks = stable tracking
    durations = [len(v) for v in all_track_ids.values()]
    avg_dur   = _mean(durations) if durations else 0.0

    # MOTA proxy: penalise ID switches relative to total matches
    mota = max(0.0, 1.0 - (id_switches / max(total_matches, 1)))
    motp = total_iou / max(total_matches, 1)

    return {
        "unique_tracks" : unique_tracks,
        "id_switches"   : id_switches,
        "total_matches" : total_matches,
        "avg_track_dur" : round(avg_dur, 1),
        "mota_proxy"    : round(mota * 100, 2),   # as %
        "motp_proxy"    : round(motp * 100, 2),   # as %
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Pipeline benchmark")
    parser.add_argument("--config",         default="configs/tracking_config.yaml")
    parser.add_argument("--source",         default="data/sample_videos/street_video.mp4")
    parser.add_argument("--frames",         type=int,  default=100)
    parser.add_argument("--runs",           type=int,  default=1)
    parser.add_argument("--device",         default=None,
                        help="Force a single device (mps|cpu). Default: benchmark both.")
    parser.add_argument("--skip-tracking",  action="store_true")
    parser.add_argument("--skip-metrics",   action="store_true")
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config))
    src = args.source

    if not Path(src).exists():
        print(f"[benchmark] Video not found: {src}")
        print("[benchmark] Run: python -m data.prepare_video -street_video")
        return

    print(f"\n{'═'*58}")
    print(f"  Video Tracking Analytics — Benchmark")
    print(f"{'═'*58}")
    print(f"  Source  : {src}")
    print(f"  Frames  : {args.frames}  |  Runs: {args.runs}")
    print(f"  PyTorch : {torch.__version__}")
    print(f"  MPS     : {torch.backends.mps.is_available()}")
    print(f"{'═'*58}\n")

    # ── Devices to test ───────────────────────────────────────────────────────
    if args.device:
        devices = [args.device]
    else:
        devices = ["mps"] if torch.backends.mps.is_available() else []
        devices.append("cpu")

    # ── Speed benchmark ───────────────────────────────────────────────────────
    print("── Speed Benchmark ──────────────────────────────────\n")
    speed_rows = []
    for device in devices:
        print(f"  Testing {device.upper()} × {args.runs} run(s) × {args.frames} frames …")
        run_results = []
        for _ in range(args.runs):
            r = bench_device(cfg, device, src, args.frames,
                             run_tracking=not args.skip_tracking)
            run_results.append(r)

        # Average across runs
        avg = {
            "device"      : device.upper(),
            "avg_det_ms"  : round(_mean([r["avg_det_ms"]   for r in run_results]), 1),
            "avg_track_ms": round(_mean([r["avg_track_ms"] for r in run_results]), 1),
            "avg_total_ms": round(_mean([r["avg_total_ms"] for r in run_results]), 1),
            "fps"         : round(_mean([r["fps"]          for r in run_results]), 1),
        }
        speed_rows.append((
            avg["device"],
            f"{avg['avg_det_ms']} ms",
            f"{avg['avg_track_ms']} ms",
            f"{avg['avg_total_ms']} ms",
            f"{avg['fps']} fps",
        ))

    print()
    print(_table(
        speed_rows,
        ["Device", "Detect", "Track", "Total", "FPS"],
    ))

    # ── Speedup ───────────────────────────────────────────────────────────────
    if len(speed_rows) == 2:
        fps_vals = [float(r[4].replace(" fps", "")) for r in speed_rows]
        if fps_vals[1] > 0:
            speedup = fps_vals[0] / fps_vals[1]
            print(f"\n  MPS speedup over CPU: {speedup:.2f}×\n")

    # ── MOT metrics ───────────────────────────────────────────────────────────
    if not args.skip_metrics:
        print("\n── Tracking Metrics (proxy, no GT annotations) ──────\n")
        m = compute_mot_metrics(cfg, src, args.frames)
        metric_rows = [
            ("Unique tracks",        m["unique_tracks"]),
            ("ID switches",          m["id_switches"]),
            ("Total matched frames", m["total_matches"]),
            ("Avg track duration",   f"{m['avg_track_dur']} frames"),
            ("MOTA (proxy)",         f"{m['mota_proxy']} %"),
            ("MOTP (proxy)",         f"{m['motp_proxy']} %"),
        ]
        print(_table(metric_rows, ["Metric", "Value"]))
        print()
        print("  Note: MOTA/MOTP here are self-consistency proxies.")
        print("  For official scores, evaluate against MOT17 ground truth")
        print("  using the py-motmetrics library.\n")

    print("── Done ─────────────────────────────────────────────\n")


if __name__ == "__main__":
    main()