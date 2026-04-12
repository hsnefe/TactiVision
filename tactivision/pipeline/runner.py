"""Orchestrates video I/O, perception stubs, analytics, and rendering."""

from __future__ import annotations

from pathlib import Path

from tactivision.analytics.camera_pan import CameraPanEstimator
from tactivision.analytics.possession import PossessionEstimator
from tactivision.config.settings import Settings
from tactivision.detection.detector import Detector
from tactivision.field.homography import FieldMapper
from tactivision.io.video import VideoReader, VideoWriter, VideoProperties
from tactivision.teams.team_assigner import TeamAssigner
from tactivision.tracking.tracker import Tracker
from tactivision.visualization.renderer import AnnotationRenderer


class AnalysisPipeline:
    """Wires all stages and runs the main frame loop."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._detector = Detector(settings)
        self._tracker = Tracker(settings, self._detector)
        self._teams = TeamAssigner(settings)
        self._possession = PossessionEstimator(settings)
        self._camera_pan = CameraPanEstimator(settings)
        self._field = FieldMapper(settings)
        self._renderer = AnnotationRenderer()

    def run(self) -> Path:
        """
        Process the full input video and write ``settings.output_video``.

        Detection/tracking are stubs: output is a passthrough with a small HUD.
        """
        inp = self._settings.input_video
        out = self._settings.output_video

        self._detector.load()
        self._tracker.reset()
        self._teams.reset()
        self._possession.reset()
        self._camera_pan.reset()
        self._field.reset()
        self._field.configure_from_settings()

        reader = VideoReader(inp)
        reader.open()
        try:
            props = reader.properties
            writer = VideoWriter(
                out,
                width=props.width,
                height=props.height,
                fps=props.fps,
            )
            writer.open()
            try:
                self._process_frames(reader, writer, props)
            finally:
                writer.release()
        finally:
            reader.release()
            self._tracker.close()
            self._detector.close()

        return out

    def _process_frames(
        self,
        reader: VideoReader,
        writer: VideoWriter,
        props: VideoProperties,
    ) -> None:
        frame_index = 0
        for frame in reader.frames():
            tracked = self._tracker.update(frame)
            track_to_team = self._teams.update(frame, tracked)
            poss = self._possession.update(frame, tracked, track_to_team)
            _pan = self._camera_pan.update(frame)
            if self._settings.enable_top_down:
                _ = self._field.image_to_field((0.0, 0.0))

            hud = f"frame {frame_index} | stub pipeline"
            annotated = self._renderer.render(
                frame,
                tracked,
                track_to_team,
                poss,
                hud_text=hud,
            )
            writer.write(annotated)
            frame_index += 1
