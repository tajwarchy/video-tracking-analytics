"""
analytics/heatmap_generator.py
───────────────────────────────
Accumulates object centroid positions into a 2-D density grid and
renders a Gaussian-blurred, colour-mapped heatmap overlay.

Two modes
─────────
1. Incremental  — call update() every frame; render() overlays the
                  current density on a reference frame at any point.
2. Batch        — call add_centroids(array) with all positions at once,
                  then render().

Consumed by:
  - inference/visualization.py  (live overlay toggle)
  - inference/process_video.py  (end-of-video heatmap export)
  - results/heatmaps/           (saved PNG)
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

# OpenCV colourmap names → int constants
_COLORMAPS: dict[str, int] = {
    "HOT"     : cv2.COLORMAP_HOT,
    "JET"     : cv2.COLORMAP_JET,
    "INFERNO" : cv2.COLORMAP_INFERNO,
    "PLASMA"  : cv2.COLORMAP_PLASMA,
    "VIRIDIS" : cv2.COLORMAP_VIRIDIS,
    "TURBO"   : cv2.COLORMAP_TURBO,
}


class HeatmapGenerator:
    """
    Parameters
    ----------
    frame_width, frame_height : int
        Pixel dimensions of the video frame.
    blur_kernel : int
        Gaussian blur kernel size (must be odd; automatically rounded up).
    colormap : str
        One of HOT | JET | INFERNO | PLASMA | VIRIDIS | TURBO.
    alpha : float
        Blend weight of the heatmap over the reference frame (0–1).
    accumulate_every_n : int
        Accumulate centroids every N frames (1 = every frame).
    """

    def __init__(
        self,
        frame_width        : int   = 1280,
        frame_height       : int   = 720,
        blur_kernel        : int   = 25,
        colormap           : str   = "HOT",
        alpha              : float = 0.5,
        accumulate_every_n : int   = 1,
    ) -> None:
        self.frame_width  = frame_width
        self.frame_height = frame_height
        self.alpha        = alpha
        self.every_n      = accumulate_every_n
        self.colormap_id  = _COLORMAPS.get(colormap.upper(), cv2.COLORMAP_HOT)

        # Ensure blur kernel is positive and odd
        k = max(3, blur_kernel)
        self.blur_kernel = k if k % 2 == 1 else k + 1

        # Float32 accumulation grid
        self._grid = np.zeros((frame_height, frame_width), dtype=np.float32)
        self._frame_counter = 0

        # Keep a reference frame (last frame seen) for the overlay
        self._reference: np.ndarray | None = None

    # ── Incremental API ───────────────────────────────────────────────────────

    def update(
        self,
        centroids : list[tuple[float, float]],
        frame     : np.ndarray | None = None,
    ) -> None:
        """
        Accumulate centroids from the current frame.

        Parameters
        ----------
        centroids : list of (cx, cy) floats
        frame     : current BGR frame (stored as reference for overlay)
        """
        self._frame_counter += 1
        if frame is not None:
            self._reference = frame

        if self._frame_counter % self.every_n != 0:
            return

        for cx, cy in centroids:
            x = int(np.clip(cx, 0, self.frame_width  - 1))
            y = int(np.clip(cy, 0, self.frame_height - 1))
            self._grid[y, x] += 1.0

    # ── Batch API ─────────────────────────────────────────────────────────────

    def add_centroids(self, points: np.ndarray) -> None:
        """
        Bulk-add a (M, 2) float array of (cx, cy) positions.
        Used by process_video.py after reading full trajectory data.
        """
        if points.shape[0] == 0:
            return
        xs = np.clip(points[:, 0].astype(int), 0, self.frame_width  - 1)
        ys = np.clip(points[:, 1].astype(int), 0, self.frame_height - 1)
        np.add.at(self._grid, (ys, xs), 1.0)

    # ── Rendering ─────────────────────────────────────────────────────────────

    def render(
        self,
        reference_frame : np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Produce a BGR heatmap overlay.

        Parameters
        ----------
        reference_frame : optional BGR frame to blend onto.
            Falls back to the last stored reference, then to black.

        Returns
        -------
        np.ndarray  BGR image (same size as the frame).
        """
        base = (
            reference_frame
            if reference_frame is not None
            else (self._reference if self._reference is not None
                  else np.zeros((self.frame_height, self.frame_width, 3), dtype=np.uint8))
        )

        coloured = self._build_coloured_map()
        return cv2.addWeighted(base, 1.0 - self.alpha, coloured, self.alpha, 0)

    def render_standalone(self) -> np.ndarray:
        """Return the heatmap without blending (black background)."""
        return self._build_coloured_map()

    def save(self, path: str | Path, reference_frame: np.ndarray | None = None) -> None:
        """Save the blended heatmap to disk as PNG."""
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        img = self.render(reference_frame)
        cv2.imwrite(str(out_path), img)
        print(f"[HeatmapGenerator] Saved → {out_path}")

    def get_grid(self) -> np.ndarray:
        """Return the raw (unnormalised) float32 accumulation grid."""
        return self._grid.copy()

    def reset(self) -> None:
        """Clear the accumulation grid (call between videos)."""
        self._grid[:] = 0.0
        self._frame_counter = 0
        self._reference = None

    # ── Internals ─────────────────────────────────────────────────────────────

    def _build_coloured_map(self) -> np.ndarray:
        """Normalise → blur → colourise the accumulation grid."""
        grid = self._grid.copy()

        if grid.max() == 0:
            return np.zeros((self.frame_height, self.frame_width, 3), dtype=np.uint8)

        # Normalise to 0–255
        norm = cv2.normalize(grid, None, 0, 255, cv2.NORM_MINMAX)
        norm = norm.astype(np.uint8)

        # Gaussian blur to smooth hotspots
        blurred = cv2.GaussianBlur(norm, (self.blur_kernel, self.blur_kernel), 0)

        # Apply colour map
        coloured = cv2.applyColorMap(blurred, self.colormap_id)
        return coloured


# ── Convenience factory ───────────────────────────────────────────────────────

def build_heatmap_generator(
    cfg          : dict,
    frame_width  : int = 1280,
    frame_height : int = 720,
) -> HeatmapGenerator | None:
    """
    Build from tracking_config.yaml dict.
    Returns None if heatmap is disabled in config.
    """
    h = cfg.get("heatmap", {})
    if not h.get("enabled", True):
        return None
    return HeatmapGenerator(
        frame_width        = frame_width,
        frame_height       = frame_height,
        blur_kernel        = h.get("blur_kernel", 25),
        colormap           = h.get("colormap", "HOT"),
        alpha              = h.get("alpha", 0.5),
        accumulate_every_n = h.get("accumulate_every_n", 1),
    )