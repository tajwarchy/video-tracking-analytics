"""
inference/visualization.py
───────────────────────────
All OpenCV drawing operations in one place.

Draws onto a BGR frame (in-place) and returns it.

Responsibilities
────────────────
- Bounding boxes + track ID labels (colour-coded per class)
- Trajectory trail polylines with optional fade
- Counting lines + IN/OUT labels
- HUD overlay (FPS, active count, unique total, line counts)
- Heatmap blend toggle
"""

from __future__ import annotations

import cv2
import numpy as np

from tracking.tracker import TrackedObject
from tracking.trajectory import TrajectoryStore
from analytics.counter import LineCounter
from analytics.heatmap_generator import HeatmapGenerator


# ── Default colour palette (BGR) — used when config doesn't specify ──────────
_DEFAULT_COLORS: dict[int, tuple[int, int, int]] = {
    0:  (0,   255, 127),   # person      → spring green
    2:  (0,   165, 255),   # car         → orange
    3:  (0,   215, 255),   # motorcycle  → gold
    5:  (255, 0,   0  ),   # bus         → blue
    7:  (0,   0,   255),   # truck       → red
}
_FALLBACK_COLOR = (200, 200, 200)   # grey for unknown classes


def _get_color(class_id: int, color_cfg: dict | None) -> tuple[int, int, int]:
    if color_cfg:
        raw = color_cfg.get(class_id) or color_cfg.get(str(class_id))
        if raw:
            return tuple(raw)
    return _DEFAULT_COLORS.get(class_id, _FALLBACK_COLOR)


# ── Main Visualizer class ─────────────────────────────────────────────────────

class Visualizer:
    """
    Stateless drawing helper — all state lives in the pipeline modules.

    Parameters
    ----------
    cfg : dict
        The full tracking_config.yaml dict.
    class_names : dict[int, str]
        COCO id → human label.
    """

    def __init__(
        self,
        cfg         : dict,
        class_names : dict[int, str] | None = None,
    ) -> None:
        self.class_names = class_names or {}
        v = cfg.get("visualization", {})

        self.show_boxes    = v.get("show_boxes",    True)
        self.show_ids      = v.get("show_ids",      True)
        self.show_labels   = v.get("show_labels",   True)
        self.show_conf     = v.get("show_confidence", False)
        self.show_trails   = v.get("show_trails",   True)
        self.show_hud      = v.get("show_hud",      True)
        self.show_lines    = v.get("show_counting_lines", True)
        self.trail_fade    = v.get("trail_fade",    True)
        self.box_thickness = v.get("box_thickness", 2)
        self.font_scale    = v.get("font_scale",    0.6)
        self._color_cfg    = v.get("colors", {})

        # Toggle flags (runtime, not config)
        self.heatmap_enabled = True
        self.trails_enabled  = True

    # ── Public draw methods ───────────────────────────────────────────────────

    def draw(
        self,
        frame       : np.ndarray,
        objects     : list[TrackedObject],
        traj_store  : TrajectoryStore,
        counter     : LineCounter | None,
        stats_snap  : dict,
        heatmap_gen : HeatmapGenerator | None = None,
    ) -> np.ndarray:
        """
        Master draw call — applies every layer in the correct order.

        Returns the annotated frame (same array, modified in-place).
        """
        # 1. Heatmap blend (bottom layer)
        if heatmap_gen and self.heatmap_enabled:
            frame = heatmap_gen.render(frame)

        # 2. Counting lines
        if counter and self.show_lines:
            self._draw_counting_lines(frame, counter)

        # 3. Trajectory trails (below boxes)
        if self.show_trails and self.trails_enabled:
            self._draw_trails(frame, objects, traj_store)

        # 4. Bounding boxes + labels
        if self.show_boxes:
            self._draw_boxes(frame, objects)

        # 5. HUD (top layer)
        if self.show_hud:
            self._draw_hud(frame, stats_snap, counter)

        return frame

    # ── Individual layers ─────────────────────────────────────────────────────

    def _draw_boxes(
        self,
        frame   : np.ndarray,
        objects : list[TrackedObject],
    ) -> None:
        for obj in objects:
            x1, y1, x2, y2 = (int(v) for v in obj.bbox)
            color = _get_color(obj.class_id, self._color_cfg)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, self.box_thickness)

            # Build label string
            parts = []
            if self.show_ids:
                parts.append(f"#{obj.track_id}")
            if self.show_labels:
                parts.append(self.class_names.get(obj.class_id, f"cls{obj.class_id}"))
            if self.show_conf:
                parts.append(f"{obj.confidence:.2f}")

            if parts:
                label = " ".join(parts)
                self._put_label(frame, label, x1, y1, color)

    def _draw_trails(
        self,
        frame      : np.ndarray,
        objects    : list[TrackedObject],
        traj_store : TrajectoryStore,
    ) -> None:
        for obj in objects:
            trail = traj_store.get_trail(obj.track_id)
            if not trail or len(trail) < 2:
                continue

            color = _get_color(obj.class_id, self._color_cfg)
            n = len(trail)

            for i in range(1, n):
                if self.trail_fade:
                    # Fade: older segments are darker
                    alpha = i / n
                    seg_color = tuple(int(c * alpha) for c in color)
                else:
                    seg_color = color

                pt1 = (int(trail[i-1][0]), int(trail[i-1][1]))
                pt2 = (int(trail[i][0]),   int(trail[i][1]))
                cv2.line(frame, pt1, pt2, seg_color, 2, cv2.LINE_AA)

            # Dot at current position
            cx, cy = int(trail[-1][0]), int(trail[-1][1])
            cv2.circle(frame, (cx, cy), 4, color, -1, cv2.LINE_AA)

    def _draw_counting_lines(
        self,
        frame   : np.ndarray,
        counter : LineCounter,
    ) -> None:
        for line in counter.lines:
            p1 = (int(line.x1), int(line.y1))
            p2 = (int(line.x2), int(line.y2))

            # Line
            cv2.line(frame, p1, p2, (0, 255, 255), 2, cv2.LINE_AA)

            # Label at midpoint
            mx = int((line.x1 + line.x2) / 2)
            my = int((line.y1 + line.y2) / 2)
            label = f"{line.name}  IN:{line.count_in}  OUT:{line.count_out}"
            cv2.putText(
                frame, label, (mx - 60, my - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2, cv2.LINE_AA,
            )

    def _draw_hud(
        self,
        frame      : np.ndarray,
        stats_snap : dict,
        counter    : LineCounter | None,
    ) -> None:
        h, w = frame.shape[:2]

        fps       = stats_snap.get("fps",          0)
        active    = stats_snap.get("active_total", 0)
        unique    = stats_snap.get("total_unique", 0)
        frame_n   = stats_snap.get("frame_count",  0)
        by_class  = stats_snap.get("active_by_class", {})

        # Semi-transparent background bar
        bar_h = 36 + max(len(by_class), 1) * 22 + (
            len(counter.lines) * 22 if counter else 0
        )
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (320, bar_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

        lines = [
            f"FPS: {fps:.1f}   Frame: {frame_n}",
            f"Active: {active}   Unique: {unique}",
        ]
        for cid, cnt in by_class.items():
            name = self.class_names.get(int(cid), f"cls{cid}")
            lines.append(f"  {name}: {cnt}")

        if counter:
            for cl in counter.lines:
                lines.append(f"  {cl.name}: IN {cl.count_in} / OUT {cl.count_out}")

        for i, text in enumerate(lines):
            cv2.putText(
                frame, text, (8, 22 + i * 22),
                cv2.FONT_HERSHEY_SIMPLEX, self.font_scale,
                (255, 255, 255), 1, cv2.LINE_AA,
            )

    # ── Label helper ──────────────────────────────────────────────────────────

    def _put_label(
        self,
        frame : np.ndarray,
        text  : str,
        x     : int,
        y     : int,
        color : tuple[int, int, int],
    ) -> None:
        fs   = self.font_scale
        font = cv2.FONT_HERSHEY_SIMPLEX
        th   = 1

        (tw, tgh), baseline = cv2.getTextSize(text, font, fs, th)
        y0 = max(y - 6, tgh + baseline)

        # Background rectangle
        cv2.rectangle(
            frame,
            (x, y0 - tgh - baseline),
            (x + tw + 4, y0 + baseline),
            color, -1,
        )
        # Text (dark for readability)
        cv2.putText(frame, text, (x + 2, y0), font, fs, (20, 20, 20), th, cv2.LINE_AA)


# ── Convenience factory ───────────────────────────────────────────────────────

def build_visualizer(cfg: dict, class_names: dict[int, str] | None = None) -> Visualizer:
    return Visualizer(cfg, class_names)