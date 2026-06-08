"""Ultralytics YOLO multi-object tracking with semantic role mapping."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from VisionEngine.config.class_mapping import build_role_map_from_model, merge_role_overrides
from VisionEngine.config.settings import Settings
from VisionEngine.schemas.ball_types import FrameTrackingOutput, RawBallDetection
from VisionEngine.schemas.schema import FrameTracks, ObjectRole, TrackedInstance
from VisionEngine.tracking.roi_recovery import (
    bbox_iou,
    center_distance,
    expand_xyxy,
    is_near_frame_border,
    pick_best_by_iou,
)

logger = logging.getLogger(__name__)


class ObjectTracker:
    """
    Runs ``YOLO.track`` on each frame with a configured ByteTrack/BoT-SORT YAML,
    maps YOLO classes to ``ObjectRole``, and returns a :class:`FrameTrackingOutput`.

    For ball recall diagnostics, also runs a ball-class-only ``predict`` pass at
    ``Settings.ball_conf_threshold`` (typically lower than the main ``conf``).

    Optional ROI recovery (``Settings.roi_recovery_enabled``) runs a low-threshold
    person-only ``predict`` on crops around players the main tracker missed and
    injects synthetic :class:`TrackedInstance` rows with the same ``track_id``.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._model: Any = None
        self._aux_model: Any = None
        self._role_by_class_id: dict[int, ObjectRole] = {}
        self._class_names: dict[int, str] = {}
        self._ball_class_ids: list[int] = []
        self._track_class_ids: list[int] = []
        self._player_class_ids: list[int] = []
        self._prev_player_by_id: dict[int, TrackedInstance] = {}
        self._roi_lost_streak: dict[int, int] = {}
        self._last_ball_bbox: tuple[float, float, float, float] | None = None
        self._prev_ball_bbox: tuple[float, float, float, float] | None = None
        self._ball_lost_streak: int = 0
        self._ball_static_streak: int = 0
        self._static_ball_pos: Optional[tuple[float, float]] = None
        self._known_static_positions: list[tuple[float, float]] = []

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
        # Separate model instance for auxiliary ``predict`` calls (ball-only
        # raw predict and ROI player recovery). We MUST NOT reuse ``self._model``
        # for plain ``predict`` calls: ``YOLO.track(persist=True)`` registers
        # ``on_predict_postprocess_end`` as a model-level callback which fires
        # on *every* ``predict`` on the same instance, feeding the partial
        # detections into the ByteTrack state and corrupting player track IDs
        # (lost players never recover, even with a new ID). Using a second
        # instance keeps the main tracker state clean.
        if not self._settings.debug_persons:
            self._aux_model = YOLO(self._settings.model_path)
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
        self._player_class_ids = [
            cid for cid, role in self._role_by_class_id.items() if role is ObjectRole.PLAYER
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
        self._prev_player_by_id.clear()
        self._roi_lost_streak.clear()
        self._last_ball_bbox = None
        self._prev_ball_bbox = None
        self._ball_lost_streak = 0
        self._ball_static_streak = 0
        self._static_ball_pos = None
        self._known_static_positions = []
        if self._settings.debug_tracking:
            logger.debug("ObjectTracker.reset()")

    def _raw_ball_predict(self, frame: np.ndarray) -> tuple[RawBallDetection, ...]:
        """Low-threshold, ball-class-only detections for recall + debug (yellow overlay).

        Runs on a dedicated ``self._aux_model`` instance so this ``predict``
        never triggers the ByteTrack ``on_predict_postprocess_end`` callback
        registered on ``self._model`` by ``track(persist=True)``. Firing that
        callback with a ball-only detection set would mark every player track
        as lost on every frame and break new-ID recovery after a track loss.
        """
        aux_model = self._aux_model or self._model
        if not self._ball_class_ids or aux_model is None:
            return ()

        pred = aux_model.predict(
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

    def _scan_ball_roi(
        self,
        frame: np.ndarray,
        last_known_bbox: tuple[float, float, float, float]
    ) -> tuple[tuple[float, float, float, float], float] | None:
        """Performs ROI search around the last position if the ball is not found in general search."""
        print(f"BİLGİ: Top için ROI taraması tetiklendi! Conf: {getattr(self._settings, 'ball_roi_conf', 0.15)}")

        aux_model = self._aux_model or self._model
        if not self._ball_class_ids or aux_model is None:
            return None

        img_h, img_w = frame.shape[:2]

        roi_size = getattr(self._settings, "ball_roi_size", 300)
        cx = 0.5 * (last_known_bbox[0] + last_known_bbox[2])
        cy = 0.5 * (last_known_bbox[1] + last_known_bbox[3])
        x1 = max(0, int(cx - roi_size // 2))
        y1 = max(0, int(cy - roi_size // 2))
        x2 = min(img_w, int(cx + roi_size // 2))
        y2 = min(img_h, int(cy + roi_size // 2))

        if x2 - x1 < 10 or y2 - y1 < 10:
            return None

        roi_img = frame[y1:y2, x1:x2]
        roi_conf = getattr(self._settings, "ball_roi_conf", 0.15)

        pred = aux_model.predict(
            source=roi_img,
            conf=roi_conf,
            iou=self._settings.iou_threshold,
            classes=self._ball_class_ids,
            verbose=False,
            stream=False,
        )

        if not pred or len(pred[0].boxes) == 0:
            return None

        boxes = pred[0].boxes
        xyxy_np = boxes.xyxy.cpu().numpy().astype(np.float32)
        conf_np = boxes.conf.cpu().numpy().astype(np.float32)

        best_box = None
        highest_conf = 0.0

        for i in range(len(conf_np)):
            conf = float(conf_np[i])
            bx1, by1, bx2, by2 = float(xyxy_np[i, 0]), float(xyxy_np[i, 1]), float(xyxy_np[i, 2]), float(xyxy_np[i, 3])

            box_w = bx2 - bx1
            box_h = by2 - by1

            if box_w <= 3 or box_h <= 3 or box_w > 65 or box_h > 65:
                continue

            aspect_ratio = box_w / box_h
            if aspect_ratio < 0.3 or aspect_ratio > 3.3:
                continue

            if conf > highest_conf:
                highest_conf = conf
                best_box = (bx1 + x1, by1 + y1, bx2 + x1, by2 + y1)

        if best_box is not None:
            return best_box, highest_conf
        return None

    def update(
        self,
        frame: np.ndarray,
        frame_index: int,
        timestamp_sec: float | None = None,
    ) -> FrameTrackingOutput:
        """
        Run ball-only raw ``predict`` (diagnostic), then full ``track`` for IDs.

        Returns :class:`~VisionEngine.schemas.ball_types.FrameTrackingOutput`.
        """
        if self._model is None:
            raise RuntimeError("ObjectTracker.load() must be called before update().")

        self._prev_ball_bbox = self._last_ball_bbox
        raw_balls: tuple[RawBallDetection, ...] = ()
        if not self._settings.debug_persons:
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

        primary = self._boxes_to_frame_tracks(results, frame_index, timestamp_sec)

        ball_found_in_primary = False
        for inst in primary.instances:
            if inst.role is ObjectRole.BALL:
                self._last_ball_bbox = inst.xyxy
                ball_found_in_primary = True
                break

        if not ball_found_in_primary and raw_balls:
            if self._last_ball_bbox is not None:
                last_cx = 0.5 * (self._last_ball_bbox[0] + self._last_ball_bbox[2])
                last_cy = 0.5 * (self._last_ball_bbox[1] + self._last_ball_bbox[3])
                
                candidates = []
                for rb in raw_balls:
                    rcx = 0.5 * (rb.xyxy[0] + rb.xyxy[2])
                    rcy = 0.5 * (rb.xyxy[1] + rb.xyxy[3])
                    dist = float(np.hypot(rcx - last_cx, rcy - last_cy))
                    
                    if dist < 250.0 or rb.confidence >= 0.25:
                        score = rb.confidence - 0.001 * dist
                        candidates.append((score, rb))
                
                if candidates:
                    best_raw = max(candidates, key=lambda c: c[0])[1]
                else:
                    best_raw = max(raw_balls, key=lambda b: b.confidence)
            else:
                best_raw = max(raw_balls, key=lambda b: b.confidence)

            self._last_ball_bbox = best_raw.xyxy
            ball_found_in_primary = True

            cid = self._ball_class_ids[0] if self._ball_class_ids else -1
            yolo_name = self._class_names.get(cid, str(cid))

            recovered_from_raw = TrackedInstance(
                track_id=-1,
                xyxy=best_raw.xyxy,
                confidence=best_raw.confidence,
                yolo_class_id=cid,
                yolo_name=yolo_name,
                role=ObjectRole.BALL
            )
            primary = FrameTracks(
                frame_index=primary.frame_index,
                timestamp_sec=primary.timestamp_sec,
                instances=primary.instances + (recovered_from_raw,)
            )

        if not ball_found_in_primary and getattr(self._settings, "ball_roi_recovery", False) and self._last_ball_bbox is not None:
            roi_result = self._scan_ball_roi(frame, self._last_ball_bbox)
            if roi_result is not None:
                best_box, max_conf = roi_result
                self._last_ball_bbox = best_box
                ball_found_in_primary = True

                cid = self._ball_class_ids[0] if self._ball_class_ids else -1
                yolo_name = self._class_names.get(cid, str(cid))

                recovered_ball_inst = TrackedInstance(
                    track_id=-1,
                    xyxy=best_box,
                    confidence=max_conf,
                    yolo_class_id=cid,
                    yolo_name=yolo_name,
                    role=ObjectRole.BALL
                )
                primary = FrameTracks(
                    frame_index=primary.frame_index,
                    timestamp_sec=primary.timestamp_sec,
                    instances=primary.instances + (recovered_ball_inst,)
                )

        if ball_found_in_primary:
            self._ball_lost_streak = 0
        else:
            self._ball_lost_streak += 1

        if not ball_found_in_primary and self._last_ball_bbox is not None and self._ball_lost_streak <= 4:
            cid = self._ball_class_ids[0] if self._ball_class_ids else -1
            yolo_name = self._class_names.get(cid, str(cid))

            memory_ball_inst = TrackedInstance(
                track_id=-1,
                xyxy=self._last_ball_bbox,
                confidence=0.1,
                yolo_class_id=cid,
                yolo_name=yolo_name,
                role=ObjectRole.BALL
            )
            primary = FrameTracks(
                frame_index=primary.frame_index,
                timestamp_sec=primary.timestamp_sec,
                instances=primary.instances + (memory_ball_inst,)
            )

        # 1. Estimate camera pan displacement from prev frame to current frame using players
        pan_dx = 0.0
        pan_dy = 0.0
        player_displacements = []
        for inst in primary.instances:
            if inst.role is ObjectRole.PLAYER and inst.track_id >= 0:
                if inst.track_id in self._prev_player_by_id:
                    prev_box = self._prev_player_by_id[inst.track_id].xyxy
                    curr_box = inst.xyxy
                    prev_cx = 0.5 * (prev_box[0] + prev_box[2])
                    prev_cy = 0.5 * (prev_box[1] + prev_box[3])
                    curr_cx = 0.5 * (curr_box[0] + curr_box[2])
                    curr_cy = 0.5 * (curr_box[1] + curr_box[3])
                    player_displacements.append((curr_cx - prev_cx, curr_cy - prev_cy))
        
        if len(player_displacements) >= 3:
            pan_dx = float(np.median([d[0] for d in player_displacements]))
            pan_dy = float(np.median([d[1] for d in player_displacements]))

        # Update all known static positions with camera pan
        if self._static_ball_pos is not None:
            self._static_ball_pos = (self._static_ball_pos[0] + pan_dx, self._static_ball_pos[1] + pan_dy)
        self._known_static_positions = [
            (x + pan_dx, y + pan_dy) for x, y in getattr(self, '_known_static_positions', [])
        ]

        # 2. Filter out static ball false positives (like penalty spots)
        ball_inst = None
        for inst in primary.instances:
            if inst.role is ObjectRole.BALL:
                ball_inst = inst
                break

        if ball_inst is not None:
            curr_box = ball_inst.xyxy
            curr_cx = 0.5 * (curr_box[0] + curr_box[2])
            curr_cy = 0.5 * (curr_box[1] + curr_box[3])
            
            # Check against all previously known static positions first
            known_statics = getattr(self, '_known_static_positions', [])
            is_known_static = False
            for sx, sy in known_statics:
                if float(np.hypot(curr_cx - sx, curr_cy - sy)) < 15.0:
                    is_known_static = True
                    break

            if is_known_static:
                # Immediately discard — this is a known static false positive
                logger.info(
                    "FILTER: Instantly discarded known static false positive at frame %d pos=(%.0f,%.0f)",
                    frame_index, curr_cx, curr_cy
                )
                new_instances = tuple(inst for inst in primary.instances if inst.role is not ObjectRole.BALL)
                primary = FrameTracks(
                    frame_index=primary.frame_index,
                    timestamp_sec=primary.timestamp_sec,
                    instances=new_instances
                )
                self._last_ball_bbox = None
            else:
                # Check if the ball is stationary (building up streak)
                matched_static = False
                if self._static_ball_pos is not None:
                    dist_to_static = float(np.hypot(curr_cx - self._static_ball_pos[0], curr_cy - self._static_ball_pos[1]))
                    if dist_to_static < 8.0:
                        self._ball_static_streak += 1
                        self._static_ball_pos = (curr_cx, curr_cy)
                        matched_static = True

                if not matched_static:
                    if self._prev_ball_bbox is not None:
                        prev_cx = 0.5 * (self._prev_ball_bbox[0] + self._prev_ball_bbox[2])
                        prev_cy = 0.5 * (self._prev_ball_bbox[1] + self._prev_ball_bbox[3])
                        ball_dx = curr_cx - prev_cx
                        ball_dy = curr_cy - prev_cy
                        comp_dx = ball_dx - pan_dx
                        comp_dy = ball_dy - pan_dy
                        comp_dist = float(np.hypot(comp_dx, comp_dy))
                        if comp_dist < 5.0:
                            self._ball_static_streak += 1
                            self._static_ball_pos = (curr_cx, curr_cy)
                        else:
                            self._ball_static_streak = 0
                            self._static_ball_pos = None
                    else:
                        self._ball_static_streak = 0
                        self._static_ball_pos = None

                # If ball has been static for 3+ frames, it's a pitch marking
                if self._ball_static_streak > 3:
                    logger.info(
                        "FILTER: Discarded static ball false positive (penalty spot?) at frame %d, streak=%d",
                        frame_index, self._ball_static_streak
                    )
                    new_instances = tuple(inst for inst in primary.instances if inst.role is not ObjectRole.BALL)
                    primary = FrameTracks(
                        frame_index=primary.frame_index,
                        timestamp_sec=primary.timestamp_sec,
                        instances=new_instances
                    )
                    self._last_ball_bbox = None
                    # Remember this position permanently
                    if not hasattr(self, '_known_static_positions'):
                        self._known_static_positions = []
                    already_known = any(
                        float(np.hypot(curr_cx - sx, curr_cy - sy)) < 20.0
                        for sx, sy in self._known_static_positions
                    )
                    if not already_known:
                        self._known_static_positions.append((curr_cx, curr_cy))
                        logger.info("FILTER: Registered new known static position at (%.0f, %.0f)", curr_cx, curr_cy)
        else:
            self._ball_static_streak = 0

        if self._settings.debug_persons or not self._settings.roi_recovery_enabled:
            # Like --debug_persons: emit raw tracker output so that a player
            # reappearing with a new ByteTrack ID is rendered immediately on
            # the next frame instead of being suppressed by the identity
            # preservation pipeline below.
            return FrameTrackingOutput(tracks=primary, raw_ball_detections=raw_balls)

        primary = self._relabel_lost_players_from_primary_nearby(primary, frame_index)
        self._update_roi_lost_streak(primary)
        merged = self._merge_roi_player_recovery(
            frame, frame_index, timestamp_sec, primary
        )
        self._refresh_prev_player_state(merged)
        return FrameTrackingOutput(tracks=merged, raw_ball_detections=raw_balls)

    def _update_roi_lost_streak(self, primary: FrameTracks) -> None:
        """Primary missed tracks increment streak; primary hits reset it."""
        primary_pids = {
            inst.track_id
            for inst in primary.instances
            if inst.role is ObjectRole.PLAYER and inst.track_id >= 0
        }
        for tid in primary_pids:
            self._roi_lost_streak[tid] = 0
        for tid in list(self._prev_player_by_id.keys()):
            if tid not in primary_pids:
                self._roi_lost_streak[tid] = self._roi_lost_streak.get(tid, 0) + 1

    def _refresh_prev_player_state(self, merged: FrameTracks) -> None:
        """Keep last bboxes for ROI while a track is missing (until pruned by streak)."""
        new_prev: dict[int, TrackedInstance] = {}
        for inst in merged.instances:
            if inst.role is ObjectRole.PLAYER and inst.track_id >= 0:
                new_prev[inst.track_id] = inst
        for tid, old in self._prev_player_by_id.items():
            if tid in new_prev:
                continue
            if self._roi_lost_streak.get(tid, 0) <= self._settings.roi_max_lost_streak:
                new_prev[tid] = old
        self._prev_player_by_id = new_prev
        for tid in list(self._roi_lost_streak.keys()):
            if tid not in self._prev_player_by_id:
                del self._roi_lost_streak[tid]

    def _relabel_lost_players_from_primary_nearby(
        self,
        primary: FrameTracks,
        frame_index: int,
    ) -> FrameTracks:
        """
        Rebind nearby visible players to recently lost IDs before ROI predict.

        This handles the case where a person is still visible near the last known
        location, but tracker identity continuity breaks. We relabel that visible
        person instance with the lost ``track_id`` and keep downstream continuity.
        """
        if not self._prev_player_by_id:
            return primary

        instances = list(primary.instances)
        primary_player_ids = {
            inst.track_id
            for inst in instances
            if inst.role is ObjectRole.PLAYER and inst.track_id >= 0
        }
        lost_ids = [
            tid
            for tid in sorted(self._prev_player_by_id.keys())
            if tid not in primary_player_ids
            and self._roi_lost_streak.get(tid, 0) <= self._settings.roi_max_lost_streak
        ]
        if not lost_ids:
            return primary

        player_indices = [
            i
            for i, inst in enumerate(instances)
            if inst.role is ObjectRole.PLAYER and inst.track_id >= 0
        ]
        if not player_indices:
            return primary

        # Avoid stealing IDs that were already known and still active from history.
        active_historical_ids = primary_player_ids & set(self._prev_player_by_id.keys())
        used_candidate_indices: set[int] = set()
        changed = False

        for lost_tid in lost_ids:
            last = self._prev_player_by_id[lost_tid]
            last_box = last.xyxy
            last_w = max(1.0, last_box[2] - last_box[0])
            last_h = max(1.0, last_box[3] - last_box[1])
            max_center_dist = (1.0 + self._settings.roi_margin_ratio) * max(last_w, last_h)
            best_idx: int | None = None
            best_score = float("-inf")

            for idx in player_indices:
                if idx in used_candidate_indices:
                    continue
                cand = instances[idx]
                if cand.track_id in active_historical_ids:
                    continue
                iou = bbox_iou(cand.xyxy, last_box)
                dist = center_distance(cand.xyxy, last_box)
                near_enough = (
                    iou >= self._settings.roi_min_iou_with_last
                    or dist <= max_center_dist
                )
                if not near_enough:
                    continue
                norm_dist = dist / max(1.0, max_center_dist)
                score = iou - 0.25 * norm_dist
                if score > best_score:
                    best_score = score
                    best_idx = idx

            if best_idx is None:
                continue

            cand = instances[best_idx]
            if self._settings.debug_tracking:
                logger.debug(
                    "roi_relabel_primary_nearby frame=%s old_id=%s -> lost_id=%s iou=%.3f dist=%.1f",
                    frame_index,
                    cand.track_id,
                    lost_tid,
                    bbox_iou(cand.xyxy, last_box),
                    center_distance(cand.xyxy, last_box),
                )
            instances[best_idx] = TrackedInstance(
                track_id=lost_tid,
                xyxy=cand.xyxy,
                confidence=cand.confidence,
                yolo_class_id=cand.yolo_class_id,
                yolo_name=cand.yolo_name,
                role=cand.role,
            )
            used_candidate_indices.add(best_idx)
            changed = True

        if not changed:
            return primary
        return FrameTracks(
            frame_index=primary.frame_index,
            timestamp_sec=primary.timestamp_sec,
            instances=tuple(instances),
        )

    def _merge_roi_player_recovery(
        self,
        frame: np.ndarray,
        frame_index: int,
        timestamp_sec: float | None,
        primary: FrameTracks,
    ) -> FrameTracks:
        """
        If ByteTrack misses a player id that existed before, run a low-threshold
        person-only ``predict`` on an expanded crop and inject a synthetic
        :class:`TrackedInstance` with the **same** ``track_id`` so downstream
        stages can keep continuity until the main tracker sees the player again.

        Uses ``self._aux_model`` so this ``predict`` does not trigger the
        ByteTrack tracker callback registered on ``self._model`` (which would
        corrupt player tracks).
        """
        aux_model = self._aux_model or self._model
        if not self._player_class_ids or aux_model is None:
            return primary

        h, w = int(frame.shape[0]), int(frame.shape[1])
        primary_pids = {
            inst.track_id
            for inst in primary.instances
            if inst.role is ObjectRole.PLAYER and inst.track_id >= 0
        }

        candidates: list[int] = []
        for tid, inst in self._prev_player_by_id.items():
            if tid in primary_pids:
                continue
            if self._roi_lost_streak.get(tid, 0) > self._settings.roi_max_lost_streak:
                continue
            if self._settings.roi_skip_near_border and is_near_frame_border(
                inst.xyxy,
                w,
                h,
                self._settings.roi_border_margin_px,
            ):
                continue
            candidates.append(tid)

        candidates.sort()
        candidates = candidates[: self._settings.roi_max_per_frame]

        extra: list[TrackedInstance] = []
        for tid in candidates:
            last = self._prev_player_by_id[tid]
            rx1, ry1, rx2, ry2 = expand_xyxy(
                last.xyxy,
                self._settings.roi_margin_ratio,
                w,
                h,
            )
            roi = frame[ry1:ry2, rx1:rx2]
            if roi.size == 0:
                continue

            pred = aux_model.predict(
                source=roi,
                conf=self._settings.roi_conf_threshold,
                iou=self._settings.iou_threshold,
                imgsz=self._settings.roi_inference_imgsz,
                classes=self._player_class_ids,
                verbose=False,
                stream=False,
            )
            if not pred or len(pred[0].boxes) == 0:
                continue

            boxes_t = pred[0].boxes
            xyxy_np = boxes_t.xyxy.cpu().numpy().astype(np.float32)
            conf_np = boxes_t.conf.cpu().numpy().astype(np.float32)
            cls_np = boxes_t.cls.cpu().numpy().astype(np.int32)
            n = int(cls_np.shape[0])
            glob_boxes: list[tuple[float, float, float, float]] = []
            meta: list[tuple[float, int]] = []
            for i in range(n):
                cid = int(cls_np[i])
                if self._role_by_class_id.get(cid, ObjectRole.OTHER) is not ObjectRole.PLAYER:
                    continue
                gx1 = float(xyxy_np[i, 0]) + float(rx1)
                gy1 = float(xyxy_np[i, 1]) + float(ry1)
                gx2 = float(xyxy_np[i, 2]) + float(rx1)
                gy2 = float(xyxy_np[i, 3]) + float(ry1)
                glob_boxes.append((gx1, gy1, gx2, gy2))
                meta.append((float(conf_np[i]), cid))

            if not glob_boxes:
                continue

            bi = pick_best_by_iou(glob_boxes, last.xyxy)
            if bi is None:
                continue
            best_box = glob_boxes[bi]
            if bbox_iou(best_box, last.xyxy) < self._settings.roi_min_iou_with_last:
                continue

            conflict = False
            for inst in primary.instances:
                if inst.track_id == tid:
                    continue
                if bbox_iou(best_box, inst.xyxy) > self._settings.roi_max_iou_with_other_track:
                    conflict = True
                    break
            if conflict:
                continue
            for prev_extra in extra:
                if bbox_iou(best_box, prev_extra.xyxy) > self._settings.roi_max_iou_with_other_track:
                    conflict = True
                    break
            if conflict:
                continue

            conf_v, cid = meta[bi]
            yolo_name = self._class_names.get(cid, str(cid))
            extra.append(
                TrackedInstance(
                    track_id=tid,
                    xyxy=best_box,
                    confidence=conf_v,
                    yolo_class_id=cid,
                    yolo_name=yolo_name,
                    role=ObjectRole.PLAYER,
                )
            )
            if self._settings.debug_tracking:
                logger.debug(
                    "roi_player_recovery frame=%s track_id=%s iou_vs_last=%.3f",
                    frame_index,
                    tid,
                    bbox_iou(best_box, last.xyxy),
                )

        if not extra:
            return primary

        return FrameTracks(
            frame_index=frame_index,
            timestamp_sec=timestamp_sec,
            instances=primary.instances + tuple(extra),
        )

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
        self._aux_model = None
        self._role_by_class_id.clear()
        self._class_names.clear()
        self._ball_class_ids.clear()
        self._track_class_ids.clear()
        self._player_class_ids.clear()
        self._prev_player_by_id.clear()
        self._roi_lost_streak.clear()
        self._last_ball_bbox = None
        self._prev_ball_bbox = None
        self._ball_lost_streak = 0
        self._ball_static_streak = 0
        self._static_ball_pos = None
        self._known_static_positions = []