"""Orchestrates video I/O, perception, analytics, and rendering."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from VisionEngine.EventAnalytics.camera_pan import CameraPanEstimator
from VisionEngine.EventAnalytics.duel import DuelEstimator
from VisionEngine.EventAnalytics.possession import PossessionEstimator
from VisionEngine.clustering.team_assigner import TeamAssigner
from VisionEngine.config.settings import Settings
from VisionEngine.debug.player_tracking_debugger import PlayerTrackingDebugger
from VisionEngine.io.track_log import TrackingJsonlWriter
from VisionEngine.io.video import VideoProperties, VideoReader, VideoWriter
from VisionEngine.metrics.homography import FieldMapper
from VisionEngine.schemas.schema import FrameTracks, ObjectRole, TrackedInstance
from VisionEngine.tracking.ball_temporal import BallTemporalBridge
from VisionEngine.tracking.object_tracker import ObjectTracker
from VisionEngine.visualization.renderer import AnnotationRenderer

logger = logging.getLogger(__name__)


class AnalysisPipeline:
    """Wires all stages and runs the main frame loop."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._tracker = ObjectTracker(settings)
        self._teams = TeamAssigner(settings)
        self._possession = PossessionEstimator(settings)
        self._duel_estimator = DuelEstimator(settings)
        self._camera_pan = CameraPanEstimator(settings)
        self._field = FieldMapper(settings)
        self._renderer = AnnotationRenderer()
        self._ball_bridge = BallTemporalBridge(settings.ball_max_gap_frames)
        self._player_debugger = PlayerTrackingDebugger(
            only_one_player=settings.debug_only1_player,
            scope_margin_px=settings.debug_player_scope_margin_px,
        )

        self._field_kp_detector = None
        self._field_mask_filter = None
        self._pitch_cfg = None
        if self._settings.field_detection_enabled:
            from VisionEngine.field.keypoint_detector import FieldKeypointDetector
            from VisionEngine.field.mask_filter import FieldMaskFilter
            from VisionEngine.field.pitch_geometry import SoccerPitchConfiguration

            self._field_kp_detector = FieldKeypointDetector(settings)
            self._field_mask_filter = FieldMaskFilter(
                expand_ratio=settings.field_mask_expand_ratio,
            )
            self._pitch_cfg = SoccerPitchConfiguration()

    def run(self) -> Path:
        """
        Process the full input video and write ``settings.output_video``.

        Uses :class:`~VisionEngine.tracking.object_tracker.ObjectTracker` for
        YOLO detection + multi-object tracking; optionally writes JSONL sidecar.
        """
        inp = self._settings.input_video
        out = self._settings.output_video

        self._tracker.load()
        self._tracker.reset()
        self._ball_bridge.reset()
        self._player_debugger.reset()
        self._teams.reset()
        self._possession.reset()
        self._duel_estimator.reset()
        self._camera_pan.reset()
        self._field.reset()
        self._field.configure_from_settings()
        if self._field_mask_filter is not None:
            self._field_mask_filter.reset()
        if self._field_kp_detector is not None:
            self._field_kp_detector.load()

        log_writer: TrackingJsonlWriter | None = None
        if self._settings.tracks_log_path is not None:
            log_writer = TrackingJsonlWriter(self._settings.tracks_log_path)
            log_writer.open()

        reader = VideoReader(inp)
        reader.open()
        try:
            props = reader.properties
            fps = max(props.fps, 1e-6)
            writer = VideoWriter(
                out,
                width=props.width,
                height=props.height,
                fps=props.fps,
            )
            writer.open()
            try:
                self._process_frames(reader, writer, props, fps, log_writer)
            finally:
                writer.release()
        finally:
            reader.release()
            if log_writer is not None:
                log_writer.close()
            self._tracker.close()

        return out

    def _process_frames(
        self,
        reader: VideoReader,
        writer: VideoWriter,
        props: VideoProperties,
        fps: float,
        log_writer: TrackingJsonlWriter | None,
    ) -> None:
        frame_index = 0
        for frame in reader.frames():
            t_sec = frame_index / fps
            self._maybe_update_field_calibration(frame, frame_index)
            ft_out = self._tracker.update(
                frame, frame_index=frame_index, timestamp_sec=t_sec
            )
            tracks = ft_out.tracks
            if self._field_mask_filter is not None and self._field_mask_filter.has_polygon:
                tracks = self._field_mask_filter.filter_tracks(tracks)
            if self._settings.debug_player_tracking:
                h, w = frame.shape[:2]
                self._player_debugger.update(
                    frame_tracks=tracks,
                    frame_width=w,
                    frame_height=h,
                )
            if log_writer is not None:
                log_writer.write_frame(tracks)

            tracked_balls = tracks.filter_by_role({ObjectRole.BALL})
            if self._settings.debug_persons:
                estimated_ball_center = None
            else:
                est_xy, is_estimated = self._ball_bridge.update(
                    frame_index, tracked_balls
                )
                estimated_ball_center = est_xy if is_estimated else None

            if self._settings.debug_persons:
                track_to_team: dict[int, int] = {}
                poss = self._possession.update(frame, tracks, track_to_team)
                duel_state = self._duel_estimator.update(frame, tracks, track_to_team, poss, estimated_ball_center)
            else:
                track_to_team = self._teams.update(frame, tracks)
                id_remap = self._teams.track_id_remap
                if id_remap:
                    tracks = _remap_track_ids(tracks, id_remap)
                poss = self._possession.update(frame, tracks, track_to_team)
                duel_state = self._duel_estimator.update(frame, tracks, track_to_team, poss, estimated_ball_center)
            team_enabled = (
                self._settings.team_classification_enabled
                and not self._settings.debug_persons
            )
            team_draw = track_to_team if team_enabled else None
            _pan = self._camera_pan.update(frame)
            if self._settings.enable_top_down:
                _ = self._field.image_to_field((0.0, 0.0))

            hud = f"frame {frame_index}"
            if self._settings.debug_persons:
                hud = f"{hud} | debug_persons"
            if self._settings.field_detection_enabled:
                hud = f"{hud} | field"
            ball_debug = (
                self._settings.ball_debug_overlay and not self._settings.debug_persons
            )
            annotated = self._renderer.render(
                frame,
                tracks,
                team_draw,
                poss,
                duel_state,
                hud_text=hud,
                raw_ball_detections=ft_out.raw_ball_detections,
                estimated_ball_center=estimated_ball_center,
                ball_debug_overlay=ball_debug,
            )
            annotated = self._draw_field_overlay(annotated)
            writer.write(annotated)
            frame_index += 1

    def _maybe_update_field_calibration(self, frame: np.ndarray, frame_index: int) -> None:
        """Re-detect pitch keypoints periodically and refresh polygon + homography."""
        if self._field_kp_detector is None or self._pitch_cfg is None:
            return
        every = max(1, int(self._settings.field_recompute_every_n_frames))
        if frame_index != 0 and frame_index % every != 0:
            return

        from VisionEngine.field.pitch_geometry import (
            back_project_polygon,
            compute_homography,
        )

        try:
            kps = self._field_kp_detector.detect(frame)
        except Exception:
            logger.exception(
                "Pitch keypoint detection failed at frame %s; reusing previous calibration",
                frame_index,
            )
            return

        if kps.num_visible == 0:
            return

        pitch_xy = self._pitch_cfg.vertices
        if kps.xy.shape[0] != pitch_xy.shape[0]:
            n = min(kps.xy.shape[0], pitch_xy.shape[0])
            image_xy = kps.xy[:n]
            confidence = kps.confidence[:n]
            pitch_xy = pitch_xy[:n]
        else:
            image_xy = kps.xy
            confidence = kps.confidence

        H = compute_homography(
            image_xy=image_xy,
            confidence=confidence,
            pitch_xy=pitch_xy,
            min_keypoint_conf=self._settings.field_kp_min_conf,
        )
        if H is None:
            return

        self._field.set_homography(H)
        polygon_image = back_project_polygon(H, self._pitch_cfg.boundary_polygon)
        if polygon_image is not None and self._field_mask_filter is not None:
            self._field_mask_filter.update_polygon(polygon_image)
        if self._settings.debug_tracking:
            logger.debug(
                "field calibration refreshed at frame %s (visible_kps=%s)",
                frame_index,
                int((confidence >= self._settings.field_kp_min_conf).sum()),
            )

    def _draw_field_overlay(self, frame: np.ndarray) -> np.ndarray:
        """Draw the pitch polygon outline when field detection is enabled."""
        if self._field_mask_filter is None or not self._field_mask_filter.has_polygon:
            return frame
        try:
            import cv2
        except ImportError:
            return frame
        poly = self._field_mask_filter.polygon
        if poly is None or len(poly) < 3:
            return frame
        overlay = frame
        pts = poly.astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(overlay, [pts], isClosed=True, color=(0, 255, 255), thickness=2)
        return overlay


def _remap_track_ids(tracks: FrameTracks, remap: dict[int, int]) -> FrameTracks:
    """Return a new ``FrameTracks`` with raw track IDs replaced by effective IDs where remapped."""
    remapped: list[TrackedInstance] = []
    for inst in tracks.instances:
        effective_id = remap.get(inst.track_id, inst.track_id)
        if effective_id != inst.track_id:
            inst = TrackedInstance(
                track_id=effective_id,
                xyxy=inst.xyxy,
                confidence=inst.confidence,
                yolo_class_id=inst.yolo_class_id,
                yolo_name=inst.yolo_name,
                role=inst.role,
            )
        remapped.append(inst)
    return FrameTracks(
        frame_index=tracks.frame_index,
        timestamp_sec=tracks.timestamp_sec,
        instances=tuple(remapped),
    )
