"""Compatibility exports for tracking; prefer :class:`ObjectTracker`."""

from __future__ import annotations

from VisionEngine.schemas.schema import FrameTracks, ObjectRole, TrackedInstance
from VisionEngine.tracking.object_tracker import ObjectTracker

# Historical alias — prefer ObjectTracker in new code.
Tracker = ObjectTracker

__all__ = [
    "ObjectTracker",
    "Tracker",
    "FrameTracks",
    "ObjectRole",
    "TrackedInstance",
]
