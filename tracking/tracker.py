"""
tracking/tracker.py
───────────────────
ByteTrack wrapper.

Consumes the (N, 6) detection array from detector.py and returns
a list of TrackedObject dataclasses — one per active track.

Each TrackedObject carries:
    track_id  : int          persistent ID across frames
    bbox      : (x1,y1,x2,y2) current bounding box
    class_id  : int
    confidence: float
    centroid  : (cx, cy)     centre of the bounding box
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from boxmot import ByteTrack


@dataclass
class TrackedObject:
    track_id  : int
    bbox      : tuple[float, float, float, float]   # x1, y1, x2, y2
    class_id  : int
    confidence: float
    centroid  : tuple[float, float]                 # cx, cy

    # Optional fields populated by downstream modules
    velocity  : tuple[float, float] = field(default=(0.0, 0.0))
    speed_px  : float               = field(default=0.0)


class Tracker:
    """
    Wraps boxmot's ByteTrack with a clean, pipeline-friendly API.

    Parameters
    ----------
    track_thresh : float
        Minimum detection score to enter the high-confidence pool.
    match_thresh : float
        IOU threshold for track–detection association.
    track_buffer : int
        Number of frames a lost track is kept alive before deletion.
    frame_rate   : int
        Expected input FPS — used by ByteTrack internally.
    min_box_area : float
        Detections whose box area (px²) is below this are dropped
        before tracking.
    """

    def __init__(
        self,
        track_thresh : float = 0.4,
        match_thresh : float = 0.8,
        track_buffer : int   = 30,
        frame_rate   : int   = 30,
        min_box_area : float = 100.0,
    ) -> None:
        self.min_box_area = min_box_area

        self._byte = ByteTrack(
            track_thresh = track_thresh,
            match_thresh = match_thresh,
            track_buffer = track_buffer,
            frame_rate   = frame_rate,
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def update(
        self,
        detections: np.ndarray,   # (N, 6)  [x1,y1,x2,y2,conf,cls]
        frame: np.ndarray,        # BGR frame (required by boxmot API)
    ) -> list[TrackedObject]:
        """
        Feed one frame's detections into ByteTrack.

        Returns
        -------
        list[TrackedObject]
            Active tracked objects for this frame.
            Empty list when no tracks are active.
        """
        dets = self._filter(detections)

        # boxmot expects float32 (N,6) — same layout as our detector output
        if dets.shape[0] == 0:
            dets = np.empty((0, 6), dtype=np.float32)

        # ByteTrack.update() returns (M, 7): [x1,y1,x2,y2,track_id,conf,cls]
        tracks = self._byte.update(dets, frame)

        return self._to_objects(tracks)

    def reset(self) -> None:
        """Reset tracker state (call between independent videos)."""
        self._byte.reset()

    # ── Internals ─────────────────────────────────────────────────────────────

    def _filter(self, dets: np.ndarray) -> np.ndarray:
        """Drop detections with bounding-box area below min_box_area."""
        if dets.shape[0] == 0:
            return dets
        w = dets[:, 2] - dets[:, 0]
        h = dets[:, 3] - dets[:, 1]
        mask = (w * h) >= self.min_box_area
        return dets[mask]

    @staticmethod
    def _to_objects(tracks: np.ndarray) -> list[TrackedObject]:
        """
        Convert ByteTrack output array → list of TrackedObject.

        ByteTrack returns rows of [x1, y1, x2, y2, track_id, conf, cls].
        Some boxmot versions may return (x1,y1,x2,y2,id,conf,cls,det_ind);
        we handle both by indexing positionally.
        """
        objects = []
        if tracks is None or len(tracks) == 0:
            return objects

        for row in tracks:
            x1, y1, x2, y2 = float(row[0]), float(row[1]), float(row[2]), float(row[3])
            track_id  = int(row[4])
            conf      = float(row[5])
            class_id  = int(row[6])
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            objects.append(
                TrackedObject(
                    track_id  = track_id,
                    bbox      = (x1, y1, x2, y2),
                    class_id  = class_id,
                    confidence= conf,
                    centroid  = (cx, cy),
                )
            )
        return objects


# ── Convenience factory ───────────────────────────────────────────────────────

def build_tracker(cfg: dict) -> Tracker:
    """
    Instantiate a Tracker from the 'tracker' section of
    tracking_config.yaml (already loaded as a dict).
    """
    t = cfg.get("tracker", {})
    return Tracker(
        track_thresh = t.get("track_thresh", 0.4),
        match_thresh = t.get("match_thresh", 0.8),
        track_buffer = t.get("track_buffer", 30),
        frame_rate   = t.get("frame_rate",   30),
        min_box_area = t.get("min_box_area", 100.0),
    )