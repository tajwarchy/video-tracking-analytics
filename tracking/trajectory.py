"""
tracking/trajectory.py
───────────────────────
Stores and manages per-track centroid histories.

Consumed by:
  - analytics/heatmap_generator.py  (all centroid positions ever seen)
  - inference/visualization.py      (trail polylines)
  - analytics/report_generator.py   (full trajectory export)
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field

import numpy as np

from tracking.tracker import TrackedObject


@dataclass
class TrajectoryRecord:
    """
    Complete lifetime record for a single track.

    Attributes
    ----------
    track_id    : persistent ByteTrack ID
    class_id    : COCO class
    trail       : recent centroid positions (capped at max_trail_length)
    full_path   : every centroid ever recorded (unbounded — for export)
    frame_first : frame index when the track was first seen
    frame_last  : frame index of the most recent update
    is_active   : True while the track appears in the current frame
    """
    track_id   : int
    class_id   : int
    trail      : deque = field(default_factory=deque)   # (cx, cy) pairs
    full_path  : list  = field(default_factory=list)    # (frame, cx, cy)
    frame_first: int   = 0
    frame_last : int   = 0
    is_active  : bool  = True


class TrajectoryStore:
    """
    Maintains trajectory records for every track seen so far.

    Parameters
    ----------
    max_trail_length : int
        Maximum number of recent positions kept for drawing the visible
        trail.  Older positions are dropped (FIFO).
    min_track_length : int
        Minimum number of frames a track must have been seen before its
        trail is surfaced for drawing.
    """

    def __init__(
        self,
        max_trail_length: int = 60,
        min_track_length: int = 5,
    ) -> None:
        self.max_trail_length = max_trail_length
        self.min_track_length = min_track_length

        # track_id → TrajectoryRecord
        self._records: dict[int, TrajectoryRecord] = {}

    # ── Public API ────────────────────────────────────────────────────────────

    def update(
        self,
        objects   : list[TrackedObject],
        frame_idx : int,
    ) -> None:
        """
        Ingest a frame's active tracked objects.

        - Creates a new TrajectoryRecord for unseen track IDs.
        - Appends the current centroid to trail and full_path.
        - Marks previously active tracks as inactive if they disappear.
        """
        active_ids = set()

        for obj in objects:
            tid = obj.track_id
            active_ids.add(tid)

            if tid not in self._records:
                rec = TrajectoryRecord(
                    track_id    = tid,
                    class_id    = obj.class_id,
                    trail       = deque(maxlen=self.max_trail_length),
                    frame_first = frame_idx,
                )
                self._records[tid] = rec
            else:
                rec = self._records[tid]

            rec.trail.append(obj.centroid)
            rec.full_path.append((frame_idx, *obj.centroid))
            rec.frame_last = frame_idx
            rec.is_active  = True

        # Mark disappeared tracks as inactive (keep record for export)
        for tid, rec in self._records.items():
            if tid not in active_ids:
                rec.is_active = False

    def get_trail(self, track_id: int) -> list[tuple[float, float]] | None:
        """
        Return the recent centroid trail for a track, or None if the
        track is unknown or hasn't met the minimum length threshold yet.
        """
        rec = self._records.get(track_id)
        if rec is None or len(rec.trail) < self.min_track_length:
            return None
        return list(rec.trail)

    def get_all_centroids(self) -> np.ndarray:
        """
        Return every centroid position ever recorded across all tracks,
        as a float32 array of shape (M, 2).  Used by the heatmap generator.
        """
        pts = []
        for rec in self._records.values():
            for _, cx, cy in rec.full_path:
                pts.append((cx, cy))
        if not pts:
            return np.zeros((0, 2), dtype=np.float32)
        return np.array(pts, dtype=np.float32)

    def get_active_records(self) -> list[TrajectoryRecord]:
        """Return records for tracks active in the most recent frame."""
        return [r for r in self._records.values() if r.is_active]

    def get_all_records(self) -> list[TrajectoryRecord]:
        """Return all records (active + historical)."""
        return list(self._records.values())

    def total_unique_tracks(self) -> int:
        """Total number of distinct track IDs seen since last reset."""
        return len(self._records)

    def reset(self) -> None:
        """Clear all records (call between independent videos)."""
        self._records.clear()

    # ── Serialisation helpers ─────────────────────────────────────────────────

    def to_export_list(self) -> list[dict]:
        """
        Serialise all trajectory records to a list of dicts suitable
        for JSON / CSV export (consumed by report_generator.py).
        """
        out = []
        for rec in self._records.values():
            out.append({
                "track_id"   : rec.track_id,
                "class_id"   : rec.class_id,
                "frame_first": rec.frame_first,
                "frame_last" : rec.frame_last,
                "duration_frames": rec.frame_last - rec.frame_first + 1,
                "path"       : [
                    {"frame": f, "cx": cx, "cy": cy}
                    for f, cx, cy in rec.full_path
                ],
            })
        return out


# ── Convenience factory ───────────────────────────────────────────────────────

def build_trajectory_store(cfg: dict) -> TrajectoryStore:
    """Build from tracking_config.yaml dict."""
    t = cfg.get("trajectory", {})
    return TrajectoryStore(
        max_trail_length = t.get("max_trail_length", 60),
        min_track_length = t.get("min_track_length", 5),
    )