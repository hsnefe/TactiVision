"""Ultralytics YOLO multi-object tracking with semantic role mapping."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from tactivision.config.class_mapping import build_role_map_from_model, merge_role_overrides
from tactivision.config.settings import Settings
from tactivision.tracking.ball_types import FrameTrackingOutput, RawBallDetection
from tactivision.tracking.schema import FrameTracks, ObjectRole, TrackedInstance

logger = logging.getLogger(__name__)


class ObjectTracker:
    """
    Runs ``YOLO.track`` on each frame with a configured ByteTrack/BoT-SORT YAML,
    maps YOLO classes to ``ObjectRole``, and returns a :class:`FrameTrackingOutput`.

    For ball recall diagnostics, also runs a ball-class-only ``predict`` pass at
    ``Settings.ball_conf_threshold`` (typically lower than the main ``conf``).
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._model: Any = None
        self._role_by_class_id: dict[int, ObjectRole] = {}
        self._class_names: dict[int, str] = {}
        self._ball_class_ids: list[int] = []
        self._track_class_ids: list[int] = []

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
        self._class_names = {int(k): str(v) for k, v in names.items()}

        base = build_role_map_from_model(self._class_names, self._settings.class_mapping_preset)
        self._role_by_class_id = merge_role_overrides(base, self._settings.class_role_overrides)

        self._ball_class_ids = [
            cid for cid, role in self._role_by_class_id.items() if role is ObjectRole.BALL
        ]
        self._track_class_ids = [
            cid
            for cid, role in self._role_by_class_id.items()
            if role in (ObjectRole.PLAYER, ObjectRole.REFEREE, ObjectRole.BALL)
        ]

        if self._settings.debug_tracking:
            logger.info(
                "ObjectTracker loaded model=%s imgsz=%s ball_classes=%s track_classes=%s preset=%s",
                self._settings.model_path,
                self._settings.inference_imgsz,
                self._ball_class_ids,
                self._track_class_ids,
                self._settings.class_mapping_preset,
            )

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def reset(self) -> None:
        if self._settings.debug_tracking:
            logger.debug("ObjectTracker.reset()")

    def _raw_ball_predict(self, frame: np.ndarray) -> tuple[RawBallDetection, ...]:
        """Low-threshold, ball-class-only detections for recall + debug (yellow overlay)."""
        if not self._ball_class_ids or self._model is None:
            return ()

        pred = self._model.predict(
            source=frame,
            conf=self._settings.ball_conf_threshold,
            iou=self._settings.iou_threshold,
            imgsz=self._settings.inference_imgsz,
            classes=self._ball_class_ids,
            verbose=False,
            stream=False,
        )
        if not pred or len(pred[0].boxes) == 0:
            return ()

        boxes = pred[0].boxes
        xyxy_np = boxes.xyxy.cpu().numpy().astype(np.float32)
        conf_np = boxes.conf.cpu().numpy().astype(np.float32)
        out: list[RawBallDetection] = []
        for i in range(len(conf_np)):
            x1, y1, x2, y2 = (float(xyxy_np[i, j]) for j in range(4))
            out.append(
                RawBallDetection(
                    xyxy=(x1, y1, x2, y2),
                    confidence=float(conf_np[i]),
                )
            )
        return tuple(out)

    def update(
        self,
        frame: np.ndarray,
        frame_index: int,
        timestamp_sec: float | None = None,
    ) -> FrameTrackingOutput:
        """
        Run ball-only raw ``predict`` (diagnostic), then full ``track`` for IDs.

        Returns :class:`~tactivision.tracking.ball_types.FrameTrackingOutput`.
        """
        if self._model is None:
            raise RuntimeError("ObjectTracker.load() must be called before update().")

        raw_balls = self._raw_ball_predict(frame)

        try:
            results = self._model.track(
                source=frame,
                conf=self._settings.conf_threshold,
                iou=self._settings.iou_threshold,
                imgsz=self._settings.inference_imgsz,
                tracker=self._settings.tracker_config,
                classes=self._track_class_ids or None,
                persist=True,
                verbose=False,
                stream=False,
            )
        except Exception as e:
            logger.exception("YOLO track() failed at frame %s", frame_index)
            raise RuntimeError(f"Tracking failed at frame {frame_index}: {e}") from e

        tracks = self._boxes_to_frame_tracks(
            results, frame_index, timestamp_sec
        )
        return FrameTrackingOutput(tracks=tracks, raw_ball_detections=raw_balls)

    def _boxes_to_frame_tracks(
        self,
        results: list | None,
        frame_index: int,
        timestamp_sec: float | None,
    ) -> FrameTracks:
        if not results:
            return FrameTracks.empty(frame_index, timestamp_sec=timestamp_sec)

        result = results[0]
        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return FrameTracks.empty(frame_index, timestamp_sec=timestamp_sec)

        xyxy_np = boxes.xyxy.cpu().numpy().astype(np.float32)
        conf_np = boxes.conf.cpu().numpy().astype(np.float32)
        cls_np = boxes.cls.cpu().numpy().astype(np.int32)
        id_t = boxes.id

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
        self._model = None
        self._role_by_class_id.clear()
        self._class_names.clear()
        self._ball_class_ids.clear()
        self._track_class_ids.clear()
