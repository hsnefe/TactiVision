"""Draw boxes, IDs, team colors, HUD text, and optional ball debug layers."""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from VisionEngine.EventAnalytics.duel import DuelState
from VisionEngine.EventAnalytics.possession import PossessionState
from VisionEngine.schemas.ball_types import RawBallDetection
from VisionEngine.schemas.schema import FrameTracks
from VisionEngine.visualization.ball_debug_draw import (
    draw_ball_debug_legend,
    draw_estimated_ball,
    draw_raw_ball_detections,
    draw_tracked_ball_highlight,
)
from VisionEngine.visualization.track_draw import draw_frame_tracks


class AnnotationRenderer:
    """Compose overlays for the output video."""

    def __init__(self) -> None:
        pass

    def render(
        self,
        frame: np.ndarray,
        tracks: FrameTracks,
        track_to_team: Optional[dict[int, int]],
        possession: PossessionState,
        duel_state: DuelState,
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
        out = draw_frame_tracks(
            out,
            tracks,
            thickness=2,
            show_role=True,
            track_to_team=track_to_team,
        )
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
            
        if duel_state.is_duel:
            duel_txt = f"DUEL! T{duel_state.team_1_id}:{duel_state.player_1_id} vs T{duel_state.team_2_id}:{duel_state.player_2_id}"
            cv2.putText(
                out,
                duel_txt,
                (16, 96),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 0, 255),
                3,
                cv2.LINE_AA,
            )
            p1_center = None
            p2_center = None
            for inst in tracks.instances:
                if inst.track_id == duel_state.player_1_id:
                    x1, y1, x2, y2 = inst.xyxy
                    p1_center = (int((x1+x2)/2), int((y1+y2)/2))
                elif inst.track_id == duel_state.player_2_id:
                    x1, y1, x2, y2 = inst.xyxy
                    p2_center = (int((x1+x2)/2), int((y1+y2)/2))
            
            if p1_center and p2_center:
                cv2.line(out, p1_center, p2_center, (0, 255, 255), 4, cv2.LINE_AA)
                if duel_state.center_xy:
                    cx, cy = int(duel_state.center_xy[0]), int(duel_state.center_xy[1])
                    cv2.circle(out, (cx, cy), 40, (0, 0, 255), 3, cv2.LINE_AA)

        return out
