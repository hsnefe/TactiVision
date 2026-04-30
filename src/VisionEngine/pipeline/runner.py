"""Orchestrates video I/O, perception, analytics, and rendering."""

from __future__ import annotations

from pathlib import Path

from VisionEngine.EventAnalytics.camera_pan import CameraPanEstimator
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


class AnalysisPipeline:
    """Wires all stages and runs the main frame loop."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._tracker = ObjectTracker(settings)
        self._teams = TeamAssigner(settings)
        self._possession = PossessionEstimator(settings)
        self._camera_pan = CameraPanEstimator(settings)
        self._field = FieldMapper(settings)
        self._renderer = AnnotationRenderer()
        self._ball_bridge = BallTemporalBridge(settings.ball_max_gap_frames)
        self._player_debugger = PlayerTrackingDebugger(
            only_one_player=settings.debug_only1_player,
            scope_margin_px=settings.debug_player_scope_margin_px,
        )

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
        self._camera_pan.reset()
        self._field.reset()
        self._field.configure_from_settings()

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
            ft_out = self._tracker.update(
                frame, frame_index=frame_index, timestamp_sec=t_sec
            )
            tracks = ft_out.tracks
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
            else:
                track_to_team = self._teams.update(frame, tracks)
                id_remap = self._teams.track_id_remap
                if id_remap:
                    tracks = _remap_track_ids(tracks, id_remap)
                poss = self._possession.update(frame, tracks, track_to_team)
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
            ball_debug = (
                self._settings.ball_debug_overlay and not self._settings.debug_persons
            )
            annotated = self._renderer.render(
                frame,
                tracks,
                team_draw,
                poss,
                hud_text=hud,
                raw_ball_detections=ft_out.raw_ball_detections,
                estimated_ball_center=estimated_ball_center,
                ball_debug_overlay=ball_debug,
            )
            writer.write(annotated)
            frame_index += 1


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
