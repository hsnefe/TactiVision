"""Ultralytics YOLO wrapper — stub until model wiring is implemented."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from tactivision.config.settings import Settings


@dataclass
class FrameDetections:
    """Single-frame detection tensors in pixel space (xyxy)."""

    xyxy: np.ndarray  # (N, 4) float32, LTRB
    conf: np.ndarray  # (N,) float32
    cls: np.ndarray  # (N,) int32, class indices

    @staticmethod
    def empty() -> FrameDetections:
        return FrameDetections(
            xyxy=np.zeros((0, 4), dtype=np.float32),
            conf=np.zeros((0,), dtype=np.float32),
            cls=np.zeros((0,), dtype=np.int32),
        )


class Detector:
    """Loads a YOLO model and runs per-frame inference."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._model = None  # TODO: ultralytics YOLO(settings.model_path)

    def load(self) -> None:
        """Load weights into memory. No-op in stub."""
        # TODO: from ultralytics import YOLO
        # self._model = YOLO(self._settings.model_path)
        return

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def predict(self, frame: np.ndarray) -> FrameDetections:
        """
        Run detection on one BGR frame.

        Returns empty detections until Ultralytics is wired.
        """
        # TODO: results = self._model.predict(frame, conf=..., iou=..., verbose=False)
        # TODO: parse results[0].boxes to FrameDetections
        return FrameDetections.empty()

    def close(self) -> None:
        """Release model resources."""
        self._model = None
