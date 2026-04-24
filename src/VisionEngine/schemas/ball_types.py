"""Per-frame ball debug and temporal overlay types."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

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
    pre_pitch_filter_tracks: Optional[FrameTracks] = None
    """Copy of tracks before pitch / min-area filtering (for debug overlay)."""

    pitch_contour: Optional[np.ndarray] = None
    """Largest grass contour in full-frame coordinates, shape ``(N, 1, 2)`` float32."""

    pitch_mask: Optional[np.ndarray] = None
    """Filled pitch mask ``uint8`` (H, W), 255 inside contour; optional debug / reuse."""
