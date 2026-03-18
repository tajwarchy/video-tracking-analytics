"""
analytics/counter.py
─────────────────────
Virtual line crossing counter.

Each counting line is defined by two points in relative coordinates
(0.0–1.0 of frame width/height).  The counter detects when a tracked
object's centroid crosses the line and increments IN or OUT depending
on which side it moved from.

Crossing direction logic
────────────────────────
We record the signed side of the line (using the 2-D cross product)
for the previous centroid position.  When the sign flips the object
has crossed.  The sign of the cross product at the *new* position
determines IN (+) vs OUT (−) — configurable per line.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field

from tracking.tracker import TrackedObject


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class CountingLine:
    name      : str
    # Absolute pixel coordinates (set when frame size is known)
    x1        : float = 0.0
    y1        : float = 0.0
    x2        : float = 0.0
    y2        : float = 0.0
    # Relative coordinates from config (0.0–1.0)
    rel_p1    : tuple[float, float] = (0.5, 0.0)
    rel_p2    : tuple[float, float] = (0.5, 1.0)
    # Counts
    count_in  : int = 0
    count_out : int = 0

    @property
    def total(self) -> int:
        return self.count_in + self.count_out

    def to_dict(self) -> dict:
        return {
            "name"     : self.name,
            "count_in" : self.count_in,
            "count_out": self.count_out,
            "total"    : self.total,
        }


@dataclass
class CrossingEvent:
    """Fired each time an object crosses a line."""
    frame_idx : int
    track_id  : int
    class_id  : int
    line_name : str
    direction : str   # "in" | "out"


# ── Counter ───────────────────────────────────────────────────────────────────

class LineCounter:
    """
    Manages one or more virtual counting lines.

    Parameters
    ----------
    line_configs : list[dict]
        Each dict from the 'counting.lines' section of
        tracking_config.yaml, e.g.::

            {"name": "Line A",
             "points": [[0.5, 0.0], [0.5, 1.0]],
             "direction": "horizontal"}

    frame_width, frame_height : int
        Pixel dimensions of the video frame.  Must be set before the
        first call to update().  Use configure_frame() if the
        dimensions aren't known at construction time.
    """

    def __init__(
        self,
        line_configs  : list[dict],
        frame_width   : int = 1280,
        frame_height  : int = 720,
    ) -> None:
        self.frame_width  = frame_width
        self.frame_height = frame_height

        self.lines  : list[CountingLine] = []
        self.events : list[CrossingEvent] = []

        # track_id → {line_name: signed_side}
        self._prev_side: dict[int, dict[str, float]] = defaultdict(dict)

        # Per-class counts  {line_name: {class_id: {"in": n, "out": n}}}
        self._class_counts: dict[str, dict[int, dict[str, int]]] = defaultdict(
            lambda: defaultdict(lambda: {"in": 0, "out": 0})
        )

        self._build_lines(line_configs)

    # ── Configuration ─────────────────────────────────────────────────────────

    def configure_frame(self, width: int, height: int) -> None:
        """Update absolute line coordinates when frame size changes."""
        self.frame_width  = width
        self.frame_height = height
        for line in self.lines:
            self._resolve_coords(line)

    # ── Main update ───────────────────────────────────────────────────────────

    def update(
        self,
        objects   : list[TrackedObject],
        frame_idx : int = 0,
    ) -> list[CrossingEvent]:
        """
        Check every active object against every counting line.

        Returns
        -------
        list[CrossingEvent]
            New crossing events detected this frame (may be empty).
        """
        new_events: list[CrossingEvent] = []

        for obj in objects:
            tid = obj.track_id
            cx, cy = obj.centroid

            for line in self.lines:
                side = _signed_side(cx, cy, line.x1, line.y1, line.x2, line.y2)
                prev = self._prev_side[tid].get(line.name)

                if prev is not None and prev != 0.0 and side != 0.0:
                    if (prev > 0) != (side > 0):   # sign flip → crossing
                        direction = "in" if side > 0 else "out"
                        if direction == "in":
                            line.count_in  += 1
                        else:
                            line.count_out += 1
                        self._class_counts[line.name][obj.class_id][direction] += 1

                        ev = CrossingEvent(
                            frame_idx = frame_idx,
                            track_id  = tid,
                            class_id  = obj.class_id,
                            line_name = line.name,
                            direction = direction,
                        )
                        new_events.append(ev)
                        self.events.append(ev)

                self._prev_side[tid][line.name] = side

        return new_events

    # ── Accessors ─────────────────────────────────────────────────────────────

    def get_counts(self) -> dict[str, dict]:
        """Return a summary dict keyed by line name."""
        return {line.name: line.to_dict() for line in self.lines}

    def get_class_counts(self) -> dict[str, dict[int, dict[str, int]]]:
        """Per-class breakdown: {line_name: {class_id: {in:n, out:n}}}."""
        return dict(self._class_counts)

    def get_total_in(self) -> int:
        return sum(l.count_in for l in self.lines)

    def get_total_out(self) -> int:
        return sum(l.count_out for l in self.lines)

    def reset(self) -> None:
        """Reset all counts and state (call between videos)."""
        for line in self.lines:
            line.count_in  = 0
            line.count_out = 0
        self._prev_side.clear()
        self._class_counts.clear()
        self.events.clear()

    # ── Internals ─────────────────────────────────────────────────────────────

    def _build_lines(self, configs: list[dict]) -> None:
        for cfg in configs:
            pts = cfg.get("points", [[0.5, 0.0], [0.5, 1.0]])
            line = CountingLine(
                name   = cfg.get("name", "Line"),
                rel_p1 = tuple(pts[0]),
                rel_p2 = tuple(pts[1]),
            )
            self._resolve_coords(line)
            self.lines.append(line)

    def _resolve_coords(self, line: CountingLine) -> None:
        line.x1 = line.rel_p1[0] * self.frame_width
        line.y1 = line.rel_p1[1] * self.frame_height
        line.x2 = line.rel_p2[0] * self.frame_width
        line.y2 = line.rel_p2[1] * self.frame_height


# ── Geometry helper ───────────────────────────────────────────────────────────

def _signed_side(
    px: float, py: float,
    ax: float, ay: float,
    bx: float, by: float,
) -> float:
    """
    Returns the signed area of the triangle (A, B, P).
    Positive = P is to the left of AB; negative = right.
    Zero = P is on the line.
    """
    return (bx - ax) * (py - ay) - (by - ay) * (px - ax)


# ── Convenience factory ───────────────────────────────────────────────────────

def build_counter(
    cfg          : dict,
    frame_width  : int = 1280,
    frame_height : int = 720,
) -> LineCounter | None:
    """
    Build a LineCounter from tracking_config.yaml dict.
    Returns None if counting is disabled in config.
    """
    counting = cfg.get("counting", {})
    if not counting.get("enabled", True):
        return None
    lines = counting.get("lines", [])
    return LineCounter(lines, frame_width, frame_height)