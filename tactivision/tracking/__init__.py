"""Multi-object tracking (Ultralytics YOLO + ByteTrack-style backends).

Heavy imports (:class:`ObjectTracker`) are lazy to avoid circular imports with
``config.class_mapping`` (``class_mapping`` imports ``schema``; loading
``object_tracker`` early would re-enter ``config`` while it is still loading).
"""

from __future__ import annotations

from tactivision.tracking.schema import FrameTracks, ObjectRole, TrackedInstance
from tactivision.tracking.track_log import TrackingJsonlWriter, frame_tracks_to_record

__all__ = [
    "ObjectTracker",
    "Tracker",
    "FrameTracks",
    "TrackedInstance",
    "ObjectRole",
    "TrackingJsonlWriter",
    "frame_tracks_to_record",
]


def __getattr__(name: str):
    if name == "ObjectTracker":
        from tactivision.tracking.object_tracker import ObjectTracker

        return ObjectTracker
    if name == "Tracker":
        from tactivision.tracking.tracker import Tracker

        return Tracker
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
