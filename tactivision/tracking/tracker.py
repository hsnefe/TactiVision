"""Compatibility exports for tracking; prefer :class:`ObjectTracker`."""

from __future__ import annotations

from tactivision.tracking.object_tracker import ObjectTracker
from tactivision.tracking.schema import FrameTracks, ObjectRole, TrackedInstance

# Historical alias — prefer ObjectTracker in new code.
Tracker = ObjectTracker

__all__ = [
    "ObjectTracker",
    "Tracker",
    "FrameTracks",
    "ObjectRole",
    "TrackedInstance",
]
