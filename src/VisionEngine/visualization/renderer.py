"""Draw boxes, IDs, team colors, HUD text, and optional ball debug layers."""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from VisionEngine.EventAnalytics.duel import DuelState
from VisionEngine.EventAnalytics.goal import GoalState
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
        goal_state: GoalState,
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

        # ── GOAL OVERLAY ──
        if goal_state.is_goal:
            out = self._draw_goal_overlay(out, goal_state, tracks)

        return out

    def _draw_goal_overlay(
        self,
        frame: np.ndarray,
        goal_state: GoalState,
        tracks: FrameTracks,
    ) -> np.ndarray:
        """Draw a premium 'GOAL!' overlay with dark band, large text, and scorer info."""
        h, w = frame.shape[:2]
        out = frame.copy()

        # Semi-transparent dark band across the center
        band_h = 120
        band_y1 = (h // 2) - (band_h // 2)
        band_y2 = band_y1 + band_h
        overlay = out.copy()
        cv2.rectangle(overlay, (0, band_y1), (w, band_y2), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, out, 0.4, 0, out)

        # Top and bottom gold accent lines
        cv2.line(out, (0, band_y1), (w, band_y1), (0, 215, 255), 3, cv2.LINE_AA)
        cv2.line(out, (0, band_y2), (w, band_y2), (0, 215, 255), 3, cv2.LINE_AA)

        # Large "GOAL!" text centered
        goal_text = "GOAL!"
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 3.0
        thickness = 10

        (tw, th), baseline = cv2.getTextSize(goal_text, font, scale, thickness)
        text_x = (w - tw) // 2
        text_y = (h // 2) + (th // 2)

        # Shadow / outline layers for depth
        cv2.putText(out, goal_text, (text_x + 3, text_y + 3), font, scale, (0, 0, 0), thickness + 4, cv2.LINE_AA)
        # Outer glow (gold)
        cv2.putText(out, goal_text, (text_x, text_y), font, scale, (0, 180, 255), thickness + 2, cv2.LINE_AA)
        # Inner bright white-gold
        cv2.putText(out, goal_text, (text_x, text_y), font, scale, (0, 255, 255), thickness, cv2.LINE_AA)

        # Scorer info below the band
        if goal_state.team_id is not None and goal_state.scorer_id is not None:
            scorer_txt = f"Scorer: Team {goal_state.team_id} | Player #{goal_state.scorer_id}"
        elif goal_state.team_id is not None:
            scorer_txt = f"Scorer: Team {goal_state.team_id}"
        else:
            scorer_txt = ""

        if scorer_txt:
            s_scale = 0.8
            s_thick = 2
            (sw, sh), _ = cv2.getTextSize(scorer_txt, font, s_scale, s_thick)
            sx = (w - sw) // 2
            sy = band_y2 + 35

            cv2.putText(out, scorer_txt, (sx + 2, sy + 2), font, s_scale, (0, 0, 0), s_thick + 2, cv2.LINE_AA)
            cv2.putText(out, scorer_txt, (sx, sy), font, s_scale, (0, 215, 255), s_thick, cv2.LINE_AA)

        # Highlight scorer on the field
        if goal_state.scorer_id is not None:
            for inst in tracks.instances:
                if inst.track_id == goal_state.scorer_id:
                    x1, y1, x2, y2 = inst.xyxy
                    cx = int((x1 + x2) / 2)
                    cy = int((y1 + y2) / 2)
                    # Triple ring spotlight
                    cv2.circle(out, (cx, cy), 45, (0, 0, 0), 4, cv2.LINE_AA)
                    cv2.circle(out, (cx, cy), 45, (0, 215, 255), 2, cv2.LINE_AA)
                    cv2.circle(out, (cx, cy), 55, (0, 180, 255), 1, cv2.LINE_AA)
                    # Label
                    cv2.putText(
                        out,
                        "SCORER",
                        (int(x1) - 5, int(y1) - 15),
                        font,
                        0.7,
                        (0, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )
                    break

        return out
