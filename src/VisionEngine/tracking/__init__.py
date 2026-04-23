"""Multi-object tracking (Ultralytics YOLO + ByteTrack-style backends).

Heavy imports (:class:`ObjectTracker`) are lazy to avoid circular imports with
``config.class_mapping`` (``class_mapping`` imports ``schema``; loading
``object_tracker`` early would re-enter ``config`` while it is still loading).
"""

from __future__ import annotations

from VisionEngine.io.track_log import TrackingJsonlWriter, frame_tracks_to_record
from VisionEngine.schemas.ball_types import FrameTrackingOutput, RawBallDetection
from VisionEngine.schemas.schema import FrameTracks, ObjectRole, TrackedInstance

__all__ = [
    "ObjectTracker",
    "Tracker",
    "FrameTracks",
    "FrameTrackingOutput",
    "RawBallDetection",
    "TrackedInstance",
    "ObjectRole",
    "TrackingJsonlWriter",
    "frame_tracks_to_record",
]


def __getattr__(name: str):
    if name == "ObjectTracker":
        from VisionEngine.tracking.object_tracker import ObjectTracker

        return ObjectTracker
    if name == "Tracker":
        from VisionEngine.tracking.tracker import Tracker

        return Tracker
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
