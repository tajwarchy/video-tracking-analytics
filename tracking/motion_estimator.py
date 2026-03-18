"""
tracking/motion_estimator.py
─────────────────────────────
Per-object velocity, speed, and direction estimation.

Uses frame-to-frame centroid displacement.  Optionally converts
pixel speed to real-world units if a pixel-per-metre calibration
factor is provided.

Consumed by:
  - trajectory.py  (attaches velocity to TrackedObject)
  - analytics/statistics.py  (average speed per class)
  - inference/visualization.py  (direction arrow overlay)
"""

from __future__ import annotations

import math
from collections import defaultdict, deque

import numpy as np

from tracking.tracker import TrackedObject

# Cardinal + intercardinal direction labels
_DIRECTIONS = [
    "E", "NE", "N", "NW", "W", "SW", "S", "SE"
]


class MotionEstimator:
    """
    Maintains a short history of centroid positions per track_id and
    computes smoothed velocity vectors.

    Parameters
    ----------
    history_len : int
        Number of past positions used for smoothing (EMA window).
    px_per_metre : float | None
        If provided, speed is also expressed in m/s.
    fps : float
        Frame rate — used when converting pixel/frame → pixel/second.
    stationary_thresh_px : float
        Objects moving less than this many pixels/frame are flagged
        as stationary.
    """

    def __init__(
        self,
        history_len       : int   = 5,
        px_per_metre      : float | None = None,
        fps               : float = 30.0,
        stationary_thresh : float = 2.0,
    ) -> None:
        self.history_len       = history_len
        self.px_per_metre      = px_per_metre
        self.fps               = fps
        self.stationary_thresh = stationary_thresh

        # track_id → deque of (cx, cy)
        self._history: dict[int, deque[tuple[float, float]]] = defaultdict(
            lambda: deque(maxlen=history_len + 1)
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def update(self, objects: list[TrackedObject]) -> list[TrackedObject]:
        """
        Update centroid history and annotate each TrackedObject in-place
        with `.velocity`, `.speed_px`, `.speed_ms`, `.direction`,
        and `.is_stationary`.

        Returns the same list (mutated) for convenient chaining.
        """
        active_ids = set()

        for obj in objects:
            tid = obj.track_id
            active_ids.add(tid)
            self._history[tid].append(obj.centroid)

            vx, vy, speed = self._compute_velocity(tid)
            obj.velocity    = (vx, vy)
            obj.speed_px    = speed                              # px/frame
            obj.speed_ps    = speed * self.fps                   # px/second
            obj.speed_ms    = (                                  # m/s  (optional)
                speed * self.fps / self.px_per_metre
                if self.px_per_metre else None
            )
            obj.direction        = _angle_to_direction(math.atan2(-vy, vx))
            obj.is_stationary    = speed < self.stationary_thresh

        # Prune history for tracks that disappeared this frame
        stale = set(self._history.keys()) - active_ids
        for tid in stale:
            del self._history[tid]

        return objects

    def reset(self) -> None:
        """Clear all stored histories (call between videos)."""
        self._history.clear()

    # ── Internals ─────────────────────────────────────────────────────────────

    def _compute_velocity(self, tid: int) -> tuple[float, float, float]:
        """
        Compute smoothed (vx, vy, speed) for a track using the mean
        displacement over its stored history.

        Returns (0, 0, 0) when fewer than 2 positions are available.
        """
        hist = self._history[tid]
        if len(hist) < 2:
            return 0.0, 0.0, 0.0

        positions = list(hist)
        # Compute frame-to-frame displacements
        dxs = [positions[i+1][0] - positions[i][0] for i in range(len(positions)-1)]
        dys = [positions[i+1][1] - positions[i][1] for i in range(len(positions)-1)]

        vx = float(np.mean(dxs))
        vy = float(np.mean(dys))
        speed = math.hypot(vx, vy)
        return vx, vy, speed


# ── Helpers ───────────────────────────────────────────────────────────────────

def _angle_to_direction(angle_rad: float) -> str:
    """
    Convert an angle in radians (standard math convention, x-right, y-up)
    to a compass direction label.

    Image coordinates have y increasing downward, so the caller should
    pass atan2(-vy, vx) to get an intuitive compass direction.
    """
    idx = round(math.degrees(angle_rad) / 45) % 8
    return _DIRECTIONS[idx]


def compute_iou(box_a: tuple, box_b: tuple) -> float:
    """
    Axis-aligned bounding-box IOU.
    Boxes are (x1, y1, x2, y2).
    Exposed here as a utility for downstream modules.
    """
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2); iy2 = min(ay2, by2)

    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter == 0.0:
        return 0.0

    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter / (area_a + area_b - inter)


# ── Convenience factory ───────────────────────────────────────────────────────

def build_motion_estimator(cfg: dict) -> MotionEstimator:
    """Build from config dict (tracking_config.yaml)."""
    perf = cfg.get("performance", {})
    vid  = cfg.get("video", {})
    fps  = vid.get("fps", 30.0)
    return MotionEstimator(
        history_len       = 5,
        px_per_metre      = None,   # set manually if camera calibration known
        fps               = fps,
        stationary_thresh = 2.0,
    )