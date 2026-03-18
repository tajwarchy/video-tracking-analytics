"""
analytics/report_generator.py
───────────────────────────────
Exports tracking results to CSV and JSON, and writes a human-readable
summary report.

Output files (all under results/)
──────────────────────────────────
results/statistics/<video_stem>_summary.json   — high-level stats
results/statistics/<video_stem>_summary.txt    — human-readable summary
results/reports/<video_stem>_tracks.csv        — per-track flat table
results/reports/<video_stem>_tracks.json       — full trajectory data

Consumed by:
  - inference/process_video.py  (called at end of each video)
"""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path

import numpy as np


# ── Report Generator ─────────────────────────────────────────────────────────

class ReportGenerator:
    """
    Parameters
    ----------
    results_dir : str | Path
        Root results directory (matches config output.results_dir).
    class_names : dict[int, str]
        COCO id → human label, e.g. {0: "person", 2: "car"}.
    video_fps : float
        Used to convert frame indices to timestamps.
    """

    def __init__(
        self,
        results_dir : str | Path = "results",
        class_names : dict[int, str] | None = None,
        video_fps   : float = 30.0,
    ) -> None:
        self.results_dir = Path(results_dir)
        self.class_names = class_names or {}
        self.video_fps   = video_fps

        # Sub-directories
        self._stats_dir   = self.results_dir / "statistics"
        self._reports_dir = self.results_dir / "reports"
        self._stats_dir.mkdir(parents=True, exist_ok=True)
        self._reports_dir.mkdir(parents=True, exist_ok=True)

    # ── Public API ────────────────────────────────────────────────────────────

    def generate(
        self,
        video_stem      : str,
        trajectory_list : list[dict],   # from TrajectoryStore.to_export_list()
        stats_summary   : dict,         # from StatsTracker.summary()
        counter_totals  : dict,         # from LineCounter.get_counts()
        save_csv        : bool = True,
        save_json       : bool = True,
    ) -> dict:
        """
        Build and save all report files for one processed video.

        Returns
        -------
        dict  — paths of all generated files.
        """
        generated: dict[str, str] = {}

        flat_rows = self._flatten_tracks(trajectory_list)

        if save_csv:
            p = self._write_csv(video_stem, flat_rows)
            generated["csv"] = str(p)

        if save_json:
            p = self._write_json(video_stem, trajectory_list)
            generated["json"] = str(p)

        p = self._write_summary_json(video_stem, stats_summary, counter_totals)
        generated["summary_json"] = str(p)

        p = self._write_summary_txt(video_stem, stats_summary, counter_totals, flat_rows)
        generated["summary_txt"] = str(p)

        return generated

    # ── Writers ───────────────────────────────────────────────────────────────

    def _write_csv(self, stem: str, rows: list[dict]) -> Path:
        path = self._reports_dir / f"{stem}_tracks.csv"
        if not rows:
            path.write_text("")
            return path

        fieldnames = list(rows[0].keys())
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        print(f"[ReportGenerator] CSV  → {path}  ({len(rows)} rows)")
        return path

    def _write_json(self, stem: str, trajectory_list: list[dict]) -> Path:
        path = self._reports_dir / f"{stem}_tracks.json"
        with open(path, "w") as f:
            json.dump(trajectory_list, f, indent=2)
        print(f"[ReportGenerator] JSON → {path}  ({len(trajectory_list)} tracks)")
        return path

    def _write_summary_json(
        self,
        stem           : str,
        stats_summary  : dict,
        counter_totals : dict,
    ) -> Path:
        path = self._stats_dir / f"{stem}_summary.json"
        payload = {
            "generated_at"  : datetime.now().isoformat(),
            "video"         : stem,
            "stats"         : stats_summary,
            "line_counts"   : counter_totals,
        }
        # numpy types aren't JSON serialisable — convert them
        with open(path, "w") as f:
            json.dump(payload, f, indent=2, default=_json_default)
        return path

    def _write_summary_txt(
        self,
        stem           : str,
        stats_summary  : dict,
        counter_totals : dict,
        flat_rows      : list[dict],
    ) -> Path:
        path = self._stats_dir / f"{stem}_summary.txt"
        lines: list[str] = []
        _h = lambda s: lines.append(f"\n{'─'*50}\n{s}\n{'─'*50}")

        lines.append(f"Video Tracking Analytics — Report")
        lines.append(f"Video   : {stem}")
        lines.append(f"Created : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        _h("PERFORMANCE")
        lines.append(f"  Frames processed : {stats_summary.get('frame_count', 0)}")
        lines.append(f"  Avg FPS          : {stats_summary.get('fps', 0)}")
        lines.append(f"  Peak active      : {stats_summary.get('peak_active', 0)} objects")

        _h("OBJECT COUNTS")
        lines.append(f"  Total unique tracks : {stats_summary.get('total_unique', 0)}")
        by_class = stats_summary.get("active_by_class", {})
        for cid, cnt in by_class.items():
            name = self.class_names.get(int(cid), f"class_{cid}")
            lines.append(f"  {name:<15} : {cnt} active in last frame")

        _h("LINE CROSSING COUNTS")
        if counter_totals:
            for line_name, counts in counter_totals.items():
                lines.append(
                    f"  {line_name}: IN={counts['count_in']}  "
                    f"OUT={counts['count_out']}  TOTAL={counts['total']}"
                )
        else:
            lines.append("  (no counting lines configured)")

        _h("SPEED (px/s)")
        for cid, spd in stats_summary.get("avg_speed_by_class", {}).items():
            name = self.class_names.get(int(cid), f"class_{cid}")
            lines.append(f"  {name:<15} : {spd} px/s avg")

        _h("TRACKS SUMMARY")
        if flat_rows:
            durations = [r["duration_frames"] for r in flat_rows]
            lines.append(f"  Total tracks    : {len(flat_rows)}")
            lines.append(f"  Avg duration    : {np.mean(durations):.1f} frames "
                         f"({np.mean(durations)/self.video_fps:.2f}s)")
            lines.append(f"  Max duration    : {max(durations)} frames")

        path.write_text("\n".join(lines))
        print(f"[ReportGenerator] TXT  → {path}")
        return path

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _flatten_tracks(self, trajectory_list: list[dict]) -> list[dict]:
        """
        Convert full trajectory records to one flat CSV row per track
        (first/last position, duration, class name).
        """
        rows = []
        for rec in trajectory_list:
            path = rec.get("path", [])
            first = path[0]  if path else {}
            last  = path[-1] if path else {}
            cid   = rec.get("class_id", -1)
            rows.append({
                "track_id"       : rec["track_id"],
                "class_id"       : cid,
                "class_name"     : self.class_names.get(cid, f"class_{cid}"),
                "frame_first"    : rec.get("frame_first", 0),
                "frame_last"     : rec.get("frame_last",  0),
                "duration_frames": rec.get("duration_frames", 0),
                "duration_s"     : round(rec.get("duration_frames", 0) / self.video_fps, 3),
                "start_cx"       : round(first.get("cx", 0), 1),
                "start_cy"       : round(first.get("cy", 0), 1),
                "end_cx"         : round(last.get("cx", 0), 1),
                "end_cy"         : round(last.get("cy", 0), 1),
            })
        return rows


# ── JSON serialisation helper ─────────────────────────────────────────────────

def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Not serialisable: {type(obj)}")


# ── Convenience factory ───────────────────────────────────────────────────────

def build_report_generator(cfg: dict) -> ReportGenerator:
    """Build from tracking_config.yaml dict."""
    out      = cfg.get("output", {})
    m_cfg    = cfg.get("model", {})
    classes  = cfg.get("classes", {})

    # Build class_names from config
    raw = classes.get("names", {})
    class_names = {int(k): v for k, v in raw.items()}

    return ReportGenerator(
        results_dir = out.get("results_dir", "results"),
        class_names = class_names,
        video_fps   = out.get("video_fps", 30.0),
    )