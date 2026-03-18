"""
inference/live_stream.py
─────────────────────────
Real-time webcam (or RTSP/file) tracking pipeline.

Mirrors process_video.py but optimised for low latency:
  - No VideoWriter by default (optional with --save)
  - Keyboard hotkeys for toggling overlays at runtime
  - Stats snapshot saved on demand

Keyboard controls
─────────────────
  q  — quit
  h  — toggle heatmap overlay
  t  — toggle trajectory trails
  b  — toggle bounding boxes
  l  — toggle counting lines
  u  — toggle HUD
  s  — save current stats + heatmap snapshot to results/

Usage
─────
    # Webcam
    python inference/live_stream.py --config configs/tracking_config.yaml

    # Video file in live mode (loops)
    python inference/live_stream.py --config configs/tracking_config.yaml \
        --source data/sample_videos/synthetic_test.mp4

    # With output recording
    python inference/live_stream.py --config configs/tracking_config.yaml --save
"""

from __future__ import annotations

import argparse
import time
from datetime import datetime
from pathlib import Path

import cv2
import yaml

from tracking.detector         import build_detector
from tracking.tracker          import build_tracker
from tracking.motion_estimator import build_motion_estimator
from tracking.trajectory       import build_trajectory_store
from analytics.counter         import build_counter
from analytics.statistics      import build_stats_tracker
from analytics.heatmap_generator import build_heatmap_generator
from analytics.report_generator  import build_report_generator
from inference.visualization   import build_visualizer


# ── Hotkey map ────────────────────────────────────────────────────────────────
_HOTKEYS = {
    ord("q"): "quit",
    ord("h"): "toggle_heatmap",
    ord("t"): "toggle_trails",
    ord("b"): "toggle_boxes",
    ord("l"): "toggle_lines",
    ord("u"): "toggle_hud",
    ord("s"): "save_snapshot",
}


# ── Live pipeline ─────────────────────────────────────────────────────────────

def run_live(cfg: dict, source=None, save: bool = False) -> None:
    """
    Run the real-time tracking loop.

    Parameters
    ----------
    cfg    : loaded tracking_config.yaml dict
    source : camera index (int) or file/RTSP path (str).
             Defaults to config video.source, falls back to webcam 0.
    save   : if True, write annotated frames to results/tracked_videos/
    """
    vid_cfg  = cfg["video"]
    out_cfg  = cfg["output"]

    # ── Resolve source ────────────────────────────────────────────────────────
    if source is None:
        src_raw = vid_cfg.get("source", 0)
        # "0" as string → int (webcam index)
        try:
            src = int(src_raw)
        except (ValueError, TypeError):
            src = src_raw
    else:
        try:
            src = int(source)
        except (ValueError, TypeError):
            src = source

    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source: {src}")

    # Try to set capture resolution
    out_w = vid_cfg.get("resize_width")  or 1280
    out_h = vid_cfg.get("resize_height") or 720
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  out_w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, out_h)

    # Read back actual dimensions (camera may not honour the request)
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"[live_stream] Source : {src}")
    print(f"[live_stream] Size   : {actual_w}×{actual_h}")
    print(f"[live_stream] Hotkeys: q=quit  h=heatmap  t=trails  "
          f"b=boxes  l=lines  u=hud  s=snapshot")

    # ── Build pipeline ────────────────────────────────────────────────────────
    detector = build_detector(cfg)
    tracker  = build_tracker(cfg)
    motion   = build_motion_estimator(cfg)
    traj     = build_trajectory_store(cfg)
    counter  = build_counter(cfg, actual_w, actual_h)
    stats    = build_stats_tracker(cfg)
    heatmap  = build_heatmap_generator(cfg, actual_w, actual_h)
    report   = build_report_generator(cfg)
    viz      = build_visualizer(cfg, class_names=detector.class_names)

    # ── Optional writer ───────────────────────────────────────────────────────
    writer = None
    if save:
        out_dir = Path(out_cfg.get("results_dir", "results")) / "tracked_videos"
        out_dir.mkdir(parents=True, exist_ok=True)
        ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = out_dir / f"live_{ts}.mp4"
        fourcc  = cv2.VideoWriter_fourcc(*out_cfg.get("video_codec", "mp4v"))
        out_fps = out_cfg.get("video_fps", 30.0)
        writer  = cv2.VideoWriter(str(out_path), fourcc, out_fps, (actual_w, actual_h))
        print(f"[live_stream] Recording → {out_path}")

    # ── State ─────────────────────────────────────────────────────────────────
    frame_idx = 0
    running   = True

    # ── Main loop ─────────────────────────────────────────────────────────────
    while running:
        ok, frame = cap.read()
        if not ok:
            # For file sources, attempt to loop
            if isinstance(src, str):
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, frame = cap.read()
                if not ok:
                    break
            else:
                break

        frame_idx += 1

        # Resize if capture dimensions differ from target
        if (actual_w, actual_h) != (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                                     int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))):
            frame = cv2.resize(frame, (actual_w, actual_h))

        # ── Core pipeline ─────────────────────────────────────────────────────
        detections = detector.detect(frame)
        objects    = tracker.update(detections, frame)
        objects    = motion.update(objects)
        traj.update(objects, frame_idx)

        if counter:
            counter.update(objects, frame_idx)

        if heatmap:
            heatmap.update([obj.centroid for obj in objects], frame)

        stats.tick(
            objects,
            total_unique   = traj.total_unique_tracks(),
            counter_totals = counter.get_counts() if counter else None,
        )

        # ── Draw ──────────────────────────────────────────────────────────────
        annotated = viz.draw(
            frame.copy(), objects, traj, counter,
            stats.snapshot(), heatmap,
        )

        # ── Write ─────────────────────────────────────────────────────────────
        if writer:
            writer.write(annotated)

        # ── Display ───────────────────────────────────────────────────────────
        cv2.imshow("Video Tracking Analytics — Live", annotated)
        key = cv2.waitKey(1) & 0xFF
        action = _HOTKEYS.get(key)

        if action == "quit":
            running = False

        elif action == "toggle_heatmap":
            viz.heatmap_enabled = not viz.heatmap_enabled
            print(f"[live] Heatmap: {'ON' if viz.heatmap_enabled else 'OFF'}")

        elif action == "toggle_trails":
            viz.trails_enabled = not viz.trails_enabled
            print(f"[live] Trails: {'ON' if viz.trails_enabled else 'OFF'}")

        elif action == "toggle_boxes":
            viz.show_boxes = not viz.show_boxes
            print(f"[live] Boxes: {'ON' if viz.show_boxes else 'OFF'}")

        elif action == "toggle_lines":
            viz.show_lines = not viz.show_lines
            print(f"[live] Counting lines: {'ON' if viz.show_lines else 'OFF'}")

        elif action == "toggle_hud":
            viz.show_hud = not viz.show_hud
            print(f"[live] HUD: {'ON' if viz.show_hud else 'OFF'}")

        elif action == "save_snapshot":
            _save_snapshot(cfg, traj, stats, counter, heatmap, annotated)

    # ── Cleanup ───────────────────────────────────────────────────────────────
    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()

    # Auto-generate final report
    print("\n[live_stream] Generating final report …")
    report.generate(
        video_stem      = f"live_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        trajectory_list = traj.to_export_list(),
        stats_summary   = stats.summary(),
        counter_totals  = counter.get_counts() if counter else {},
        save_csv        = out_cfg.get("save_csv",  True),
        save_json       = out_cfg.get("save_json", True),
    )
    print("[live_stream] Done.")


# ── Snapshot helper ───────────────────────────────────────────────────────────

def _save_snapshot(cfg, traj, stats, counter, heatmap, frame) -> None:
    """Save a mid-session snapshot: heatmap PNG + stats JSON."""
    from analytics.report_generator import build_report_generator
    import json

    out_dir = Path(cfg.get("output", {}).get("results_dir", "results"))
    ts      = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save annotated frame
    snap_path = out_dir / "statistics" / f"snapshot_{ts}.jpg"
    snap_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(snap_path), frame)

    # Save current stats
    stats_path = out_dir / "statistics" / f"snapshot_{ts}_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats.summary(), f, indent=2, default=str)

    # Save heatmap
    if heatmap:
        hmap_path = out_dir / "heatmaps" / f"snapshot_{ts}_heatmap.png"
        heatmap.save(hmap_path)

    print(f"[live] Snapshot saved → {snap_path.parent}/snapshot_{ts}*")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Live webcam tracking pipeline.")
    p.add_argument("--config", default="configs/tracking_config.yaml")
    p.add_argument("--source", default=None,
                   help="Camera index (0,1,…) or video file / RTSP URL")
    p.add_argument("--save",   action="store_true",
                   help="Record the annotated stream to results/tracked_videos/")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg  = yaml.safe_load(open(args.config))
    run_live(cfg, source=args.source, save=args.save)


if __name__ == "__main__":
    main()