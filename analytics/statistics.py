"""
analytics/statistics.py
────────────────────────
Real-time statistics tracker.

Maintains rolling-window FPS, active track counts, class
distribution, and per-class average speed.

Consumed by:
  - inference/visualization.py   (HUD overlay)
  - analytics/report_generator.py (summary section)
"""

from __future__ import annotations

import time
from collections import defaultdict, deque

import numpy as np

from tracking.tracker import TrackedObject


class StatsTracker:
    """
    Collects per-frame metrics and exposes a snapshot dict for the HUD.

    Parameters
    ----------
    fps_window : int
        Number of recent frame timestamps used for the rolling FPS
        calculation.
    speed_window : int
        Number of recent speed samples per class used for smoothing.
    """

    def __init__(
        self,
        fps_window   : int = 30,
        speed_window : int = 60,
    ) -> None:
        self.fps_window   = fps_window
        self.speed_window = speed_window

        # Timing
        self._frame_times  : deque[float] = deque(maxlen=fps_window)
        self._last_tick    : float | None  = None

        # Counters
        self.frame_count          : int = 0
        self.total_unique_tracks  : int = 0

        # Per-frame live counts
        self._active_by_class : dict[int, int] = defaultdict(int)

        # Speed smoothing {class_id: deque of px/s values}
        self._speed_buf : dict[int, deque[float]] = defaultdict(
            lambda: deque(maxlen=speed_window)
        )

        # Peak tracking
        self._peak_active : int = 0

        # Per-class cumulative object-seconds (for density stats)
        self._class_frame_counts : dict[int, int] = defaultdict(int)

    # ── Main update ───────────────────────────────────────────────────────────

    def tick(
        self,
        objects          : list[TrackedObject],
        total_unique     : int,
        counter_totals   : dict | None = None,   # from LineCounter.get_counts()
    ) -> None:
        """
        Call once per processed frame, after tracking + motion estimation.

        Parameters
        ----------
        objects        : active tracked objects this frame
        total_unique   : total unique track IDs seen (from TrajectoryStore)
        counter_totals : optional crossing counts from LineCounter
        """
        now = time.perf_counter()
        if self._last_tick is not None:
            self._frame_times.append(now - self._last_tick)
        self._last_tick = now

        self.frame_count         = self.frame_count + 1
        self.total_unique_tracks = total_unique

        # Reset per-frame class counts
        self._active_by_class.clear()

        for obj in objects:
            cid = obj.class_id
            self._active_by_class[cid] += 1
            self._class_frame_counts[cid] += 1

            speed = getattr(obj, "speed_ps", 0.0)   # px/second
            if speed is not None:
                self._speed_buf[cid].append(speed)

        active_total = len(objects)
        if active_total > self._peak_active:
            self._peak_active = active_total

        self._counter_totals = counter_totals or {}

    # ── Snapshot ──────────────────────────────────────────────────────────────

    def snapshot(self) -> dict:
        """
        Return a flat dict of current stats — safe to call every frame.

        Keys
        ────
        fps               : float   rolling FPS
        frame_count       : int
        active_total      : int     objects in current frame
        active_by_class   : dict    {class_id: count}
        total_unique      : int     all-time unique track IDs
        peak_active       : int
        avg_speed_by_class: dict    {class_id: px/s float}
        counter_totals    : dict    from LineCounter (may be empty)
        """
        return {
            "fps"               : self._rolling_fps(),
            "frame_count"       : self.frame_count,
            "active_total"      : sum(self._active_by_class.values()),
            "active_by_class"   : dict(self._active_by_class),
            "total_unique"      : self.total_unique_tracks,
            "peak_active"       : self._peak_active,
            "avg_speed_by_class": self._avg_speeds(),
            "counter_totals"    : self._counter_totals,
        }

    def summary(self) -> dict:
        """
        End-of-video summary dict for the report generator.
        Includes per-class object-second fractions.
        """
        snap = self.snapshot()
        total_frames = max(self.frame_count, 1)
        class_density = {
            cid: cnt / total_frames
            for cid, cnt in self._class_frame_counts.items()
        }
        return {**snap, "class_density": class_density}

    # ── Internals ─────────────────────────────────────────────────────────────

    def _rolling_fps(self) -> float:
        if len(self._frame_times) < 2:
            return 0.0
        avg_dt = float(np.mean(list(self._frame_times)))
        return round(1.0 / avg_dt, 1) if avg_dt > 0 else 0.0

    def _avg_speeds(self) -> dict[int, float]:
        return {
            cid: round(float(np.mean(list(buf))), 1)
            for cid, buf in self._speed_buf.items()
            if buf
        }

    def reset(self) -> None:
        """Reset all stats (call between videos)."""
        self._frame_times.clear()
        self._last_tick = None
        self.frame_count = 0
        self.total_unique_tracks = 0
        self._active_by_class.clear()
        self._speed_buf.clear()
        self._peak_active = 0
        self._class_frame_counts.clear()
        self._counter_totals = {}


# ── Convenience factory ───────────────────────────────────────────────────────

def build_stats_tracker(cfg: dict) -> StatsTracker:
    """Build from tracking_config.yaml dict."""
    return StatsTracker(fps_window=30, speed_window=60)