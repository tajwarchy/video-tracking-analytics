"""
inference/process_video.py
───────────────────────────
Offline video processing pipeline.

Reads a video file frame-by-frame, runs detect → track → analyse →
draw, writes an annotated output video, and exports all reports.

Usage
─────
    python inference/process_video.py --config configs/tracking_config.yaml
    python inference/process_video.py --config configs/tracking_config.yaml \
        --source data/sample_videos/synthetic_test.mp4 --show
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import yaml
from tqdm import tqdm

# ── Pipeline modules ──────────────────────────────────────────────────────────
from tracking.detector        import build_detector
from tracking.tracker         import build_tracker
from tracking.motion_estimator import build_motion_estimator
from tracking.trajectory      import build_trajectory_store
from analytics.counter        import build_counter
from analytics.statistics     import build_stats_tracker
from analytics.heatmap_generator import build_heatmap_generator
from analytics.report_generator  import build_report_generator
from inference.visualization  import build_visualizer


# ── Pipeline ──────────────────────────────────────────────────────────────────

def process_video(cfg: dict, source: str | None = None, show: bool = False) -> dict:
    """
    Run the full tracking pipeline on a single video file.

    Parameters
    ----------
    cfg    : loaded tracking_config.yaml dict
    source : override config video.source (useful for batch processing)
    show   : open an OpenCV preview window while processing

    Returns
    -------
    dict  — paths of generated output files
    """
    # ── Config shortcuts ──────────────────────────────────────────────────────
    vid_cfg  = cfg["video"]
    out_cfg  = cfg["output"]
    perf_cfg = cfg.get("performance", {})

    src_path = Path(source or vid_cfg["source"])
    if not src_path.exists():
        raise FileNotFoundError(f"Video not found: {src_path}")

    stem = src_path.stem

    # ── Open video ────────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(str(src_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {src_path}")

    orig_w  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Optional resize
    out_w = vid_cfg.get("resize_width")  or orig_w
    out_h = vid_cfg.get("resize_height") or orig_h

    max_frames  = perf_cfg.get("max_frames")  or total
    skip_frames = perf_cfg.get("skip_frames") or 0

    print(f"\n[process_video] Source : {src_path}")
    print(f"[process_video] Size   : {orig_w}×{orig_h} → {out_w}×{out_h}")
    print(f"[process_video] FPS    : {src_fps:.1f}   Frames: {total}")

    # ── Build pipeline components ─────────────────────────────────────────────
    detector  = build_detector(cfg)
    tracker   = build_tracker(cfg)
    motion    = build_motion_estimator(cfg)
    traj      = build_trajectory_store(cfg)
    counter   = build_counter(cfg, out_w, out_h)
    stats     = build_stats_tracker(cfg)
    heatmap   = build_heatmap_generator(cfg, out_w, out_h)
    report    = build_report_generator(cfg)
    viz       = build_visualizer(cfg, class_names=detector.class_names)

    # ── Video writer ──────────────────────────────────────────────────────────
    writer = None
    if out_cfg.get("save_video", True):
        out_dir  = Path(out_cfg.get("results_dir", "results")) / "tracked_videos"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{stem}_tracked.mp4"
        fourcc   = cv2.VideoWriter_fourcc(*out_cfg.get("video_codec", "mp4v"))
        out_fps  = out_cfg.get("video_fps", src_fps)
        writer   = cv2.VideoWriter(str(out_path), fourcc, out_fps, (out_w, out_h))
        print(f"[process_video] Output : {out_path}")

    # ── Main loop ─────────────────────────────────────────────────────────────
    frame_idx = 0
    t_start   = time.perf_counter()

    with tqdm(total=min(total, max_frames), desc="Processing", unit="fr") as pbar:
        while frame_idx < max_frames:
            ok, frame = cap.read()
            if not ok:
                break

            frame_idx += 1

            # Resize if needed
            if (out_w, out_h) != (orig_w, orig_h):
                frame = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA)

            # Skip frames (process every Nth)
            if skip_frames > 0 and (frame_idx % (skip_frames + 1)) != 0:
                pbar.update(1)
                continue

            # ── Detect ────────────────────────────────────────────────────────
            detections = detector.detect(frame)

            # ── Track ─────────────────────────────────────────────────────────
            objects = tracker.update(detections, frame)

            # ── Motion estimation ─────────────────────────────────────────────
            objects = motion.update(objects)

            # ── Trajectory ────────────────────────────────────────────────────
            traj.update(objects, frame_idx)

            # ── Analytics ─────────────────────────────────────────────────────
            if counter:
                counter.update(objects, frame_idx)

            if heatmap:
                centroids = [obj.centroid for obj in objects]
                heatmap.update(centroids, frame)

            stats.tick(
                objects,
                total_unique   = traj.total_unique_tracks(),
                counter_totals = counter.get_counts() if counter else None,
            )

            # ── Draw ──────────────────────────────────────────────────────────
            annotated = viz.draw(
                frame.copy(), objects, traj, counter,
                stats.snapshot(), heatmap,
            )

            # ── Write / display ───────────────────────────────────────────────
            if writer:
                writer.write(annotated)

            if show:
                cv2.imshow("Video Tracking Analytics", annotated)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    print("\n[process_video] Interrupted by user.")
                    break
                if key == ord("h") and heatmap:
                    viz.heatmap_enabled = not viz.heatmap_enabled
                if key == ord("t"):
                    viz.trails_enabled = not viz.trails_enabled

            pbar.update(1)

    # ── Cleanup ───────────────────────────────────────────────────────────────
    cap.release()
    if writer:
        writer.release()
    if show:
        cv2.destroyAllWindows()

    elapsed = time.perf_counter() - t_start
    print(f"\n[process_video] Done in {elapsed:.1f}s  "
          f"({frame_idx / elapsed:.1f} fps avg)")

    # ── Save heatmap ──────────────────────────────────────────────────────────
    generated: dict[str, str] = {}
    if heatmap:
        hmap_path = (
            Path(out_cfg.get("results_dir", "results")) / "heatmaps" / f"{stem}_heatmap.png"
        )
        # Use all centroids for the final end-of-video heatmap
        heatmap.reset()
        heatmap.add_centroids(traj.get_all_centroids())
        heatmap.save(hmap_path)
        generated["heatmap"] = str(hmap_path)

    # ── Generate report ───────────────────────────────────────────────────────
    files = report.generate(
        video_stem      = stem,
        trajectory_list = traj.to_export_list(),
        stats_summary   = stats.summary(),
        counter_totals  = counter.get_counts() if counter else {},
        save_csv        = out_cfg.get("save_csv",  True),
        save_json       = out_cfg.get("save_json", True),
    )
    generated.update(files)
    if writer:
        generated["video"] = str(out_path)

    print("\n[process_video] Output files:")
    for k, v in generated.items():
        print(f"  {k:<16} {v}")

    return generated


# ── Batch processing ──────────────────────────────────────────────────────────

def batch_process(cfg: dict, video_dir: str, show: bool = False) -> None:
    """Process every video file found in video_dir."""
    video_dir = Path(video_dir)
    videos = sorted(
        list(video_dir.glob("*.mp4")) +
        list(video_dir.glob("*.avi")) +
        list(video_dir.glob("*.mov"))
    )
    if not videos:
        print(f"[batch] No videos found in {video_dir}")
        return

    print(f"[batch] Found {len(videos)} video(s).")
    for i, vid in enumerate(videos, 1):
        print(f"\n[batch] ── {i}/{len(videos)}: {vid.name} ──")
        try:
            process_video(cfg, source=str(vid), show=show)
        except Exception as e:
            print(f"[batch] ERROR on {vid.name}: {e}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Offline video tracking pipeline.")
    p.add_argument("--config", default="configs/tracking_config.yaml")
    p.add_argument("--source", default=None,
                   help="Override config video source path")
    p.add_argument("--batch",  default=None,
                   help="Process all videos in this directory")
    p.add_argument("--show",   action="store_true",
                   help="Display annotated frames in a preview window")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg  = yaml.safe_load(open(args.config))

    if args.batch:
        batch_process(cfg, args.batch, show=args.show)
    else:
        process_video(cfg, source=args.source, show=args.show)


if __name__ == "__main__":
    main()