"""
tracking/detector.py
────────────────────
YOLOv8 detection wrapper.

Outputs a standardised numpy array per frame:
    [[x1, y1, x2, y2, confidence, class_id], ...]

This is the exact format ByteTrack expects as input.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from ultralytics import YOLO


class Detector:
    """
    Thin wrapper around YOLOv8 that handles device selection,
    class filtering, and output normalisation.

    Parameters
    ----------
    weights : str | Path
        Path to .pt weights file, or a model name like 'yolov8n.pt'
        (auto-downloaded on first use).
    confidence : float
        Minimum detection confidence to keep (0–1).
    iou : float
        NMS IOU threshold (0–1).
    classes : list[int] | None
        COCO class IDs to keep. None = keep all classes.
    device : str
        'mps' | 'cpu' | 'cuda'
    imgsz : int
        Inference image size (pixels). Must be a multiple of 32.
    """

    def __init__(
        self,
        weights: str | Path = "yolov8n.pt",
        confidence: float = 0.4,
        iou: float = 0.5,
        classes: list[int] | None = None,
        device: str = "mps",
        imgsz: int = 640,
    ) -> None:
        self.confidence = confidence
        self.iou = iou
        self.classes = classes
        self.imgsz = imgsz
        self.device = self._resolve_device(device)

        self.model = YOLO(str(weights))
        # Warm-up: run one dummy inference so the first real frame isn't slow
        self._warmup()

    # ── Public API ────────────────────────────────────────────────────────────

    def detect(self, frame: np.ndarray) -> np.ndarray:
        """
        Run detection on a single BGR frame (as returned by cv2.VideoCapture).

        Returns
        -------
        np.ndarray  shape (N, 6)  dtype float32
            Columns: [x1, y1, x2, y2, confidence, class_id]
            Returns an empty (0, 6) array when nothing is detected.
        """
        results = self.model.predict(
            source=frame,
            conf=self.confidence,
            iou=self.iou,
            classes=self.classes,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
        )

        return self._parse(results)

    @property
    def class_names(self) -> dict[int, str]:
        """COCO class id → name mapping from the loaded model."""
        return self.model.names  # type: ignore[return-value]

    # ── Internals ─────────────────────────────────────────────────────────────

    def _warmup(self) -> None:
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        self.model.predict(
            source=dummy,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
        )

    @staticmethod
    def _resolve_device(requested: str) -> str:
        """
        Fall back gracefully: mps → cpu if MPS is unavailable,
        cuda → cpu if CUDA is unavailable.
        """
        if requested == "mps":
            if torch.backends.mps.is_available():
                return "mps"
            print("[Detector] MPS not available, falling back to CPU.")
            return "cpu"
        if requested == "cuda":
            if torch.cuda.is_available():
                return "cuda"
            print("[Detector] CUDA not available, falling back to CPU.")
            return "cpu"
        return "cpu"

    @staticmethod
    def _parse(results) -> np.ndarray:
        """
        Convert ultralytics Results → (N, 6) float32 array.
        Handles the case where results[0].boxes is empty or None.
        """
        empty = np.zeros((0, 6), dtype=np.float32)

        if not results or results[0].boxes is None:
            return empty

        boxes = results[0].boxes
        if len(boxes) == 0:
            return empty

        xyxy  = boxes.xyxy.cpu().numpy()   # (N, 4)
        conf  = boxes.conf.cpu().numpy()   # (N,)
        cls   = boxes.cls.cpu().numpy()    # (N,)

        return np.column_stack([xyxy, conf, cls]).astype(np.float32)


# ── Convenience factory ───────────────────────────────────────────────────────

def build_detector(cfg: dict) -> Detector:
    """
    Instantiate a Detector from the 'model' + 'classes' sections
    of tracking_config.yaml (already loaded as a dict).

    Example
    -------
    import yaml
    cfg = yaml.safe_load(open("configs/tracking_config.yaml"))
    detector = build_detector(cfg)
    """
    m = cfg["model"]
    return Detector(
        weights=m.get("weights", "yolov8n.pt"),
        confidence=m.get("confidence", 0.4),
        iou=m.get("iou", 0.5),
        classes=cfg.get("classes", {}).get("filter", None),
        device=m.get("device", "mps"),
        imgsz=m.get("imgsz", 640),
    )