"""Ultralytics YOLO multi-object tracking with semantic role mapping."""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np

from tactivision.config.class_mapping import build_role_map_from_model, merge_role_overrides
from tactivision.config.settings import Settings
from tactivision.tracking.schema import FrameTracks, ObjectRole, TrackedInstance

logger = logging.getLogger(__name__)


class ObjectTracker:
    """
    Runs ``YOLO.track`` on each frame with a configured ByteTrack/BoT-SORT YAML,
    maps YOLO classes to ``ObjectRole``, and returns a :class:`FrameTracks` result.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._model: Any = None
        self._role_by_class_id: dict[int, ObjectRole] = {}
        self._class_names: dict[int, str] = {}

    def load(self) -> None:
        """Load Ultralytics weights and build the class-id -> role table."""
        try:
            from ultralytics import YOLO
        except ImportError as e:
            raise ImportError(
                "ultralytics is required for ObjectTracker. "
                "Install dependencies: pip install ultralytics"
            ) from e

        self._model = YOLO(self._settings.model_path)
        names = getattr(self._model, "names", None)
        if not isinstance(names, dict):
            raise RuntimeError("YOLO model has no valid .names dict.")
        # Ultralytics may use str keys; normalize to int -> str
        self._class_names = {int(k): str(v) for k, v in names.items()}

        base = build_role_map_from_model(self._class_names, self._settings.class_mapping_preset)
        self._role_by_class_id = merge_role_overrides(base, self._settings.class_role_overrides)

        if self._settings.debug_tracking:
            logger.info(
                "ObjectTracker loaded model=%s names=%s preset=%s",
                self._settings.model_path,
                self._class_names,
                self._settings.class_mapping_preset,
            )

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def reset(self) -> None:
        """Call at the start of each new video so tracker state is fresh."""
        # Ultralytics persists tracker state on the model; creating a new ObjectTracker
        # per pipeline run is the main reset. This hook remains for explicit clears.
        if self._settings.debug_tracking:
            logger.debug("ObjectTracker.reset()")

    def update(self, frame: np.ndarray, frame_index: int, timestamp_sec: float | None = None) -> FrameTracks:
        """
        Run tracking on one BGR frame.

        Parameters
        ----------
        frame :
            HxWx3 BGR image (OpenCV convention).
        frame_index :
            Zero-based index in the video.
        timestamp_sec :
            Optional time in seconds for logging (e.g. frame_index / fps).
        """
        if self._model is None:
            raise RuntimeError("ObjectTracker.load() must be called before update().")

        try:
            # Single-frame inference; stream=False returns a list of Results
            results = self._model.track(
                source=frame,
                conf=self._settings.conf_threshold,
                iou=self._settings.iou_threshold,
                tracker=self._settings.tracker_config,
                persist=True,
                verbose=False,
                stream=False,
            )
        except Exception as e:
            logger.exception("YOLO track() failed at frame %s", frame_index)
            raise RuntimeError(f"Tracking failed at frame {frame_index}: {e}") from e

        if not results:
            return FrameTracks.empty(frame_index, timestamp_sec=timestamp_sec)

        result = results[0]
        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return FrameTracks.empty(frame_index, timestamp_sec=timestamp_sec)

        xyxy_t = boxes.xyxy
        conf_t = boxes.conf
        cls_t = boxes.cls
        id_t = boxes.id

        xyxy_np = xyxy_t.cpu().numpy().astype(np.float32)
        conf_np = conf_t.cpu().numpy().astype(np.float32)
        cls_np = cls_t.cpu().numpy().astype(np.int32)

        if id_t is not None:
            tid_np = id_t.cpu().numpy().astype(np.int64).reshape(-1)
        else:
            tid_np = np.full(len(cls_np), -1, dtype=np.int64)

        instances: list[TrackedInstance] = []
        n = int(cls_np.shape[0])
        for i in range(n):
            cid = int(cls_np[i])
            yolo_name = self._class_names.get(cid, str(cid))
            role = self._role_by_class_id.get(cid, ObjectRole.OTHER)

            if self._settings.only_mapped_classes and role is ObjectRole.OTHER:
                continue

            x1, y1, x2, y2 = (float(xyxy_np[i, j]) for j in range(4))
            tid = int(tid_np[i]) if i < len(tid_np) else -1

            instances.append(
                TrackedInstance(
                    track_id=tid,
                    xyxy=(x1, y1, x2, y2),
                    confidence=float(conf_np[i]),
                    yolo_class_id=cid,
                    yolo_name=yolo_name,
                    role=role,
                )
            )

        if self._settings.debug_tracking and n > 0:
            logger.debug(
                "frame=%s tracks=%s (after filter=%s)",
                frame_index,
                n,
                len(instances),
            )

        return FrameTracks(
            frame_index=frame_index,
            timestamp_sec=timestamp_sec,
            instances=tuple(instances),
        )

    def close(self) -> None:
        """Release the model reference."""
        self._model = None
        self._role_by_class_id.clear()
        self._class_names.clear()
