"""Per-frame ball debug and temporal overlay types."""

from __future__ import annotations

from dataclasses import dataclass

from VisionEngine.schemas.schema import FrameTracks


@dataclass(frozen=True, slots=True)
class RawBallDetection:
    """Single raw YOLO ball-only detection (no tracker ID)."""

    xyxy: tuple[float, float, float, float]
    confidence: float


@dataclass(frozen=True, slots=True)
class FrameTrackingOutput:
    """Tracker output plus a low-threshold ball-only ``predict`` pass for recall / debug."""

    tracks: FrameTracks
    raw_ball_detections: tuple[RawBallDetection, ...]
