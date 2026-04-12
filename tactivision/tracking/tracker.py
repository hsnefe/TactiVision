"""Multi-object tracking — stub; will wrap Ultralytics track() or external tracker."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from tactivision.config.settings import Settings
from tactivision.detection.detector import Detector


@dataclass
class TrackedFrame:
    """Detections augmented with persistent track IDs."""

    xyxy: np.ndarray  # (N, 4)
    conf: np.ndarray  # (N,)
    cls: np.ndarray  # (N,)
    track_ids: np.ndarray  # (N,) int64; -1 = no ID

    @staticmethod
    def empty() -> TrackedFrame:
        return TrackedFrame(
            xyxy=np.zeros((0, 4), dtype=np.float32),
            conf=np.zeros((0,), dtype=np.float32),
            cls=np.zeros((0,), dtype=np.int32),
            track_ids=np.zeros((0,), dtype=np.int64),
        )


class Tracker:
    """
    Maintains identities across frames.

    Intended use: internal Ultralytics YOLO with ``model.track(..., persist=True)``,
    or a separate tracker config (e.g. ByteTrack).
    """

    def __init__(self, settings: Settings, detector: Detector) -> None:
        self._settings = settings
        self._detector = detector
        self._frame_index = 0

    def reset(self) -> None:
        self._frame_index = 0

    def update(self, frame: np.ndarray) -> TrackedFrame:
        """
        Run tracking for one frame.

        Returns empty tracks until tracker + detector are implemented.
        """
        # TODO: call detector model in track mode, or fuse detect + ByteTrack
        self._frame_index += 1
        return TrackedFrame.empty()

    def close(self) -> None:
        pass
