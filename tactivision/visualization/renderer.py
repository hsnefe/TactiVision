"""Draw boxes, IDs, team colors, HUD text, and optional ball debug layers."""

from __future__ import annotations

from typing import Optional

import cv2

from tactivision.analytics.possession import PossessionState
from tactivision.tracking.ball_types import RawBallDetection
from tactivision.tracking.schema import FrameTracks
from tactivision.visualization.ball_debug_draw import (
    draw_ball_debug_legend,
    draw_estimated_ball,
    draw_raw_ball_detections,
    draw_tracked_ball_highlight,
)
from tactivision.visualization.track_draw import draw_frame_tracks


class AnnotationRenderer:
    """Compose overlays for the output video."""

    def __init__(self) -> None:
        pass

    def render(
        self,
        frame: np.ndarray,
        tracks: FrameTracks,
        track_to_team: dict[int, int],
        possession: PossessionState,
        hud_text: Optional[str] = None,
        *,
        raw_ball_detections: tuple[RawBallDetection, ...] = (),
        estimated_ball_center: Optional[tuple[float, float]] = None,
        ball_debug_overlay: bool = False,
    ) -> np.ndarray:
        """
        Draw tracks; optionally ball debug layers (yellow raw, red tracked emphasis, blue estimate).

        When ``ball_debug_overlay`` is True, raw ``predict`` ball boxes are drawn first (yellow),
        then all tracks, then a thick red outline on tracked ball(s), then an estimated center
        if provided (blue).
        """
        out = frame.copy()
        if ball_debug_overlay:
            out = draw_raw_ball_detections(out, raw_ball_detections)
        out = draw_frame_tracks(out, tracks, thickness=2, show_role=True)
        if ball_debug_overlay:
            out = draw_tracked_ball_highlight(out, tracks)
            out = draw_estimated_ball(out, estimated_ball_center)
            out = draw_ball_debug_legend(out)
        if hud_text:
            cv2.putText(
                out,
                hud_text,
                (16, 32),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
        if possession.team_id is not None:
            pos_txt = f"possession: team {possession.team_id}"
            cv2.putText(
                out,
                pos_txt,
                (16, 64),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 0),
                2,
                cv2.LINE_AA,
            )
        return out
