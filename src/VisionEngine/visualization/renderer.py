"""Draw boxes, IDs, team colors, HUD text, and optional ball debug layers."""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from VisionEngine.EventAnalytics.duel import DuelState
from VisionEngine.EventAnalytics.possession import PossessionState
from VisionEngine.EventAnalytics.shot import ShotState
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
        shot_state: ShotState,
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

        if shot_state.is_shot:
            # 1. Draw Shot Alert HUD (drop-shadow effect)
            shot_txt = f"SHOT!"
            if shot_state.team_id is not None and shot_state.player_id is not None:
                shot_txt += f" T{shot_state.team_id}:{shot_state.player_id}"
            if shot_state.shot_speed_kph is not None:
                shot_txt += f" | {shot_state.shot_speed_kph:.1f} km/h"
            elif shot_state.direction:
                shot_txt += f" | Attack {shot_state.direction.upper()}"

            # Shadow
            cv2.putText(
                out,
                shot_txt,
                (18, 134),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 0, 0),
                4,
                cv2.LINE_AA,
            )
            # Text (bright neon orange-red)
            cv2.putText(
                out,
                shot_txt,
                (16, 132),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (30, 80, 255),
                3,
                cv2.LINE_AA,
            )

            # 2. Draw Kicker Highlight
            kicker_center = None
            for inst in tracks.instances:
                if inst.track_id == shot_state.player_id:
                    x1, y1, x2, y2 = inst.xyxy
                    kicker_center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
                    # Draw a nice dashed-looking circle or double circle at their feet
                    feet_y = int(y2)
                    cv2.circle(out, (kicker_center[0], feet_y), 25, (0, 0, 0), 4, cv2.LINE_AA)
                    cv2.circle(out, (kicker_center[0], feet_y), 25, (0, 165, 255), 2, cv2.LINE_AA)
                    cv2.putText(
                        out,
                        "SHOOTER",
                        (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 165, 255),
                        2,
                        cv2.LINE_AA,
                    )
                    break

            # 3. Draw Neon Trajectory Tail
            traj = shot_state.trajectory
            if len(traj) >= 2:
                for i in range(1, len(traj)):
                    pt1 = (int(traj[i-1][0]), int(traj[i-1][1]))
                    pt2 = (int(traj[i][0]), int(traj[i][1]))
                    
                    # Gradient effect based on position in trajectory list
                    alpha = i / len(traj)
                    # Interpolate color from dark red (0, 0, 150) to bright orange/yellow (0, 200, 255)
                    color = (0, int(200 * alpha), int(150 + 105 * alpha))
                    thickness = int(2 + 4 * alpha)
                    
                    cv2.line(out, pt1, pt2, (0, 0, 0), thickness + 2, cv2.LINE_AA) # shadow/outline
                    cv2.line(out, pt1, pt2, color, thickness, cv2.LINE_AA)

                # Draw a final glowing circle at the end of the trajectory
                end_pt = (int(traj[-1][0]), int(traj[-1][1]))
                cv2.circle(out, end_pt, 8, (0, 255, 255), -1, cv2.LINE_AA)
                cv2.circle(out, end_pt, 12, (0, 165, 255), 2, cv2.LINE_AA)

        return out
