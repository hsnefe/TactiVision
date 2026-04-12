"""Structured detection and tracking types for one video frame."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable

import numpy as np


class ObjectRole(str, Enum):
    """Semantic role mapped from YOLO class names (or user overrides)."""

    PLAYER = "player"
    REFEREE = "referee"
    BALL = "ball"
    OTHER = "other"


@dataclass(frozen=True, slots=True)
class TrackedInstance:
    """One tracked object with bbox, YOLO class, and semantic role."""

    track_id: int
    """Tracker ID; ``-1`` if the backend did not assign an ID this frame."""

    xyxy: tuple[float, float, float, float]
    """Bounding box left, top, right, bottom in pixel coordinates."""

    confidence: float
    yolo_class_id: int
    yolo_name: str
    role: ObjectRole


@dataclass(frozen=True, slots=True)
class FrameTracks:
    """All tracked instances for a single frame."""

    frame_index: int
    timestamp_sec: float | None
    instances: tuple[TrackedInstance, ...]

    @staticmethod
    def empty(frame_index: int, timestamp_sec: float | None = None) -> FrameTracks:
        return FrameTracks(frame_index=frame_index, timestamp_sec=timestamp_sec, instances=())

    def filter_by_role(self, roles: Iterable[ObjectRole]) -> tuple[TrackedInstance, ...]:
        """Return instances whose role is in the given set."""
        want = {r for r in roles}
        return tuple(inst for inst in self.instances if inst.role in want)

    def as_legacy_arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Stack fields into NumPy arrays for legacy code paths.

        Returns ``(xyxy, conf, cls, track_ids)`` with shapes (N,4), (N,), (N,), (N,).
        ``cls`` holds YOLO class indices. ``track_ids`` uses ``-1`` when missing.
        """
        n = len(self.instances)
        if n == 0:
            return (
                np.zeros((0, 4), dtype=np.float32),
                np.zeros((0,), dtype=np.float32),
                np.zeros((0,), dtype=np.int32),
                np.zeros((0,), dtype=np.int64),
            )
        xyxy = np.array([inst.xyxy for inst in self.instances], dtype=np.float32)
        conf = np.array([inst.confidence for inst in self.instances], dtype=np.float32)
        cls_arr = np.array([inst.yolo_class_id for inst in self.instances], dtype=np.int32)
        ids = np.array([inst.track_id for inst in self.instances], dtype=np.int64)
        return xyxy, conf, cls_arr, ids
