"""Orchestrates video I/O, perception, analytics, and rendering."""

from __future__ import annotations

from pathlib import Path

from tactivision.analytics.camera_pan import CameraPanEstimator
from tactivision.analytics.possession import PossessionEstimator
from tactivision.config.settings import Settings
from tactivision.field.homography import FieldMapper
from tactivision.io.video import VideoReader, VideoWriter, VideoProperties
from tactivision.teams.team_assigner import TeamAssigner
from tactivision.tracking.object_tracker import ObjectTracker
from tactivision.tracking.track_log import TrackingJsonlWriter
from tactivision.visualization.renderer import AnnotationRenderer


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

    def run(self) -> Path:
        """
        Process the full input video and write ``settings.output_video``.

        Uses :class:`~tactivision.tracking.object_tracker.ObjectTracker` for
        YOLO detection + multi-object tracking; optionally writes JSONL sidecar.
        """
        inp = self._settings.input_video
        out = self._settings.output_video

        self._tracker.load()
        self._tracker.reset()
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
            tracks = self._tracker.update(frame, frame_index=frame_index, timestamp_sec=t_sec)
            if log_writer is not None:
                log_writer.write_frame(tracks)

            track_to_team = self._teams.update(frame, tracks)
            poss = self._possession.update(frame, tracks, track_to_team)
            _pan = self._camera_pan.update(frame)
            if self._settings.enable_top_down:
                _ = self._field.image_to_field((0.0, 0.0))

            hud = f"frame {frame_index}"
            annotated = self._renderer.render(
                frame,
                tracks,
                track_to_team,
                poss,
                hud_text=hud,
            )
            writer.write(annotated)
            frame_index += 1
