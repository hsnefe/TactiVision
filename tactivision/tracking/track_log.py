"""Append per-frame tracking results as JSON Lines."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional, TextIO

from tactivision.tracking.schema import FrameTracks, TrackedInstance

SCHEMA_VERSION = 1


def _instance_to_dict(inst: TrackedInstance) -> dict[str, Any]:
    return {
        "id": int(inst.track_id),
        "role": inst.role.value,
        "yolo_class": int(inst.yolo_class_id),
        "name": inst.yolo_name,
        "conf": float(inst.confidence),
        "xyxy": [float(inst.xyxy[0]), float(inst.xyxy[1]), float(inst.xyxy[2]), float(inst.xyxy[3])],
    }


def frame_tracks_to_record(frame: FrameTracks) -> dict[str, Any]:
    """Build one JSON-serializable dict for a single frame."""
    tracks = [_instance_to_dict(inst) for inst in frame.instances]
    rec: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "frame": int(frame.frame_index),
        "tracks": tracks,
    }
    if frame.timestamp_sec is not None:
        rec["timestamp_sec"] = float(frame.timestamp_sec)
    return rec


class TrackingJsonlWriter:
    """
    Write one JSON object per line (``.jsonl``) for streaming consumption.

    Each line matches :func:`frame_tracks_to_record`.
    """

    def __init__(self, path: Path) -> None:
        self._path = Path(path)
        self._fp: Optional[TextIO] = None

    def open(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._fp = self._path.open("w", encoding="utf-8")

    def write_frame(self, frame: FrameTracks) -> None:
        if self._fp is None:
            raise RuntimeError("TrackingJsonlWriter.open() must be called first.")
        rec = frame_tracks_to_record(frame)
        self._fp.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def close(self) -> None:
        if self._fp is not None:
            self._fp.close()
            self._fp = None

    def __enter__(self) -> TrackingJsonlWriter:
        self.open()
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


# Re-export for type checkers
__all__ = ["TrackingJsonlWriter", "frame_tracks_to_record", "SCHEMA_VERSION"]
