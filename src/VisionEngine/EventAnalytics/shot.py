"""Detect shot events (şut tespiti) and track their trajectory towards the goal."""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
from typing import Optional
import numpy as np

logger = logging.getLogger(__name__)

from VisionEngine.config.settings import Settings
from VisionEngine.schemas.schema import FrameTracks, ObjectRole
from VisionEngine.EventAnalytics.possession import PossessionState


@dataclass
class ShotState:
    """Represents a shot event currently being visualized."""

    is_shot: bool = False
    player_id: Optional[int] = None
    team_id: Optional[int] = None
    frame_started: Optional[int] = None
    shot_speed_kph: Optional[float] = None
    direction: Optional[str] = None  # "left" or "right"
    trajectory: list[tuple[float, float]] = field(default_factory=list)


class ShotEstimator:
    """Tracks player positions and ball movements to detect shots towards the goal."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

        # Settings variables
        self._shot_min_speed_kph = getattr(settings, "shot_min_speed_kph", 50.0)
        self._shot_min_rel_speed = getattr(settings, "shot_min_rel_speed", 0.30)
        self._shot_display_frames = getattr(settings, "shot_display_frames", 30)
        self._shot_max_start_dist_ratio = getattr(settings, "shot_max_start_dist_ratio", 1.2)

        # State management
        self._active_shot: Optional[ShotState] = None
        self._shot_countdown: int = 0

        # Ball history: only REAL (non-estimated) positions are stored
        # (frame_idx, ball_x, ball_y, pitch_xy_or_None, is_real)
        self._ball_history: list[tuple[int, float, float, Optional[tuple[float, float]]]] = []

        # Possession history
        self._possession_history: list[tuple[int, PossessionState]] = []

        # Accumulators for determining attack directions
        self._team_x_accum: dict[int, list[float]] = {0: [], 1: []}
        self._attack_direction: dict[int, int] = {}  # Team ID -> +1 (right) or -1 (left)

        # Cooldown to prevent double-triggering
        self._last_kicked_frame: int = -100

        # Player tracks history
        self._last_frame_players: dict[int, tuple[float, float, float, float]] = {}
        self._player_history: list[tuple[int, dict[int, tuple[float, float, float, float]]]] = []

    def reset(self) -> None:
        self._active_shot = None
        self._shot_countdown = 0
        self._ball_history.clear()
        self._possession_history.clear()
        self._team_x_accum = {0: [], 1: []}
        self._attack_direction.clear()
        self._last_kicked_frame = -100
        self._last_frame_players.clear()
        self._player_history.clear()

    def _bbox_center(self, xyxy: tuple[float, float, float, float]) -> tuple[float, float]:
        x1, y1, x2, y2 = xyxy
        return (0.5 * (x1 + x2), 0.5 * (y1 + y2))

    def update(
        self,
        frame: np.ndarray,
        tracks: FrameTracks,
        track_to_team: dict[int, int],
        possession: PossessionState,
        estimated_ball_center: Optional[tuple[float, float]] = None,
        field_mapper=None,
        ball_is_estimated: bool = False,
    ) -> ShotState:
        """Update shot estimation state for the current frame."""
        frame_idx = tracks.frame_index
        fps = 30.0

        # 1. Update team X coordinates history to determine attack directions
        players = [inst for inst in tracks.instances if inst.role is ObjectRole.PLAYER and inst.track_id >= 0]

        for p in players:
            tid = track_to_team.get(p.track_id)
            if tid is not None and tid in (0, 1):
                p_cx, p_cy = self._bbox_center(p.xyxy)
                val_x = p_cx
                if field_mapper is not None and field_mapper.is_configured:
                    pitch_pos = field_mapper.image_to_field((p_cx, p.xyxy[3]))
                    if pitch_pos is not None:
                        val_x = pitch_pos[0]
                self._team_x_accum[tid].append(val_x)

        # Dynamically determine attack directions (require enough data)
        if len(self._team_x_accum[0]) > 30 and len(self._team_x_accum[1]) > 30:
            avg_x0 = np.median(self._team_x_accum[0])
            avg_x1 = np.median(self._team_x_accum[1])
            if avg_x0 < avg_x1:
                self._attack_direction[0] = 1
                self._attack_direction[1] = -1
            else:
                self._attack_direction[0] = -1
                self._attack_direction[1] = 1

        # 2. Get REAL ball position (do NOT use estimated_ball_center for shot history)
        real_ball_xy = possession.ball_xy  # Only from actual detections
        pitch_ball_xy = None
        if real_ball_xy is not None and field_mapper is not None and field_mapper.is_configured:
            pitch_ball_xy = field_mapper.image_to_field(real_ball_xy)

        # Only record REAL ball detections into history (not temporal extrapolations)
        if real_ball_xy is not None:
            # Check for position jump (e.g. penalty spot -> real ball switch)
            # If the ball "teleports" more than 3x avg player height, clear history
            if len(self._ball_history) >= 1:
                last_f, last_bx, last_by, _ = self._ball_history[-1]
                df_hist = frame_idx - last_f
                if df_hist > 0 and df_hist <= 5:
                    jump_dist = float(np.hypot(real_ball_xy[0] - last_bx, real_ball_xy[1] - last_by))
                    avg_h = 100.0
                    if players:
                        avg_h = float(np.mean([p.xyxy[3] - p.xyxy[1] for p in players]))
                    if jump_dist > avg_h * 3.0:
                        logger.info(
                            "SHOT_GUARD: Ball position jumped %.0fpx at frame %d (limit=%.0f), clearing history",
                            jump_dist, frame_idx, avg_h * 3.0
                        )
                        self._ball_history.clear()

            self._ball_history.append((frame_idx, real_ball_xy[0], real_ball_xy[1], pitch_ball_xy))
            if len(self._ball_history) > 60:
                self._ball_history.pop(0)

        # Possession history for context
        self._possession_history.append((frame_idx, possession))
        if len(self._possession_history) > 60:
            self._possession_history.pop(0)

        ret_val = None

        # 3. Handle active shot lifecycle
        if self._active_shot is not None:
            self._shot_countdown -= 1
            if real_ball_xy is not None:
                self._active_shot.trajectory.append(real_ball_xy)

            # Terminate if another player gains possession
            if possession.track_id is not None and possession.track_id != self._active_shot.player_id:
                self._active_shot = None
                self._shot_countdown = 0
            elif self._shot_countdown <= 0:
                self._active_shot = None

            if self._active_shot is not None:
                ret_val = self._active_shot

        # 4. Check for new shot triggers — only with REAL ball and established attack direction
        cooldown_ok = (frame_idx - self._last_kicked_frame) > 30
        attack_direction_known = len(self._attack_direction) == 2

        if (
            ret_val is None
            and real_ball_xy is not None
            and len(self._ball_history) >= 3
            and cooldown_ok
            and attack_direction_known
        ):
            # Need at least 2 consecutive REAL detections close in time
            # Find the most recent previous real ball position
            prev_entry = None
            for entry in reversed(self._ball_history[:-1]):
                prev_f, prev_bx, prev_by, prev_pitch = entry
                df = frame_idx - prev_f
                # Only use frames that are recent (within 5 frames) to avoid
                # false spikes when tracking resumes after a gap
                if df <= 5 and df > 0:
                    prev_entry = entry
                    break

            if prev_entry is None:
                # No recent enough real detection to compute reliable velocity
                self._last_frame_players = {p.track_id: p.xyxy for p in players}
                self._player_history.append((frame_idx, self._last_frame_players))
                if len(self._player_history) > 60:
                    self._player_history.pop(0)
                return ret_val if ret_val is not None else ShotState()

            prev_f, prev_bx, prev_by, prev_pitch = prev_entry
            df = frame_idx - prev_f
            dx = real_ball_xy[0] - prev_bx
            dy = real_ball_xy[1] - prev_by
            v_x = dx / df
            v_y = dy / df
            speed_px = float(np.hypot(v_x, v_y))

            # Calculate average player height for scaling
            avg_player_h = 100.0
            if players:
                avg_player_h = float(np.mean([p.xyxy[3] - p.xyxy[1] for p in players]))

            # Calculate speed metric
            speed_kph = None
            is_speed_high = False

            if pitch_ball_xy is not None and prev_pitch is not None:
                dist_cm = float(np.hypot(pitch_ball_xy[0] - prev_pitch[0], pitch_ball_xy[1] - prev_pitch[1]))
                dist_m = dist_cm / 100.0
                dt = df / fps
                speed_mps = dist_m / dt
                speed_kph = speed_mps * 3.6
                is_speed_high = speed_kph >= self._shot_min_speed_kph
            else:
                rel_speed = speed_px / avg_player_h
                is_speed_high = rel_speed >= self._shot_min_rel_speed

            if is_speed_high:
                # --- Kicker detection: temporal scan ---
                ball_positions = {f: (bx, by) for f, bx, by, _ in self._ball_history}
                player_positions_by_frame = {f_idx: plist for f_idx, plist in self._player_history}

                track_min_dists: dict[int, float] = {}

                scan_start = max(0, prev_f - 15)
                scan_end = prev_f

                for f in range(scan_start, scan_end + 1):
                    if f in ball_positions and f in player_positions_by_frame:
                        bx_f, by_f = ball_positions[f]
                        for tid, bbox in player_positions_by_frame[f].items():
                            x1, y1, x2, y2 = bbox
                            feet_x = 0.5 * (x1 + x2)
                            feet_y = y2
                            dist = float(np.hypot(feet_x - bx_f, feet_y - by_f))
                            h = y2 - y1
                            # Strict threshold: must be within 1.2x player height
                            if dist < h * 1.2:
                                if tid not in track_min_dists or dist < track_min_dists[tid]:
                                    track_min_dists[tid] = dist

                kicker_info = None
                if track_min_dists:
                    best_tid = min(track_min_dists, key=track_min_dists.get)
                    kicker_team = track_to_team.get(best_tid)
                    logger.info(
                        "SHOT_CANDIDATE frame=%d prev_f=%d speed=%.2f dists=%s kicker=%d team=%s",
                        frame_idx, prev_f, speed_px / avg_player_h,
                        {k: round(v, 1) for k, v in track_min_dists.items()},
                        best_tid, kicker_team
                    )
                    kicker_info = (best_tid, kicker_team)
                else:
                    logger.info(
                        "SHOT_REJECT frame=%d: no player found near ball in window [%d,%d] (speed=%.2f h/f)",
                        frame_idx, scan_start, scan_end, speed_px / avg_player_h
                    )

                # --- Direction validation: only proceed with established direction ---
                if kicker_info is not None:
                    kicker_id, kicker_team = kicker_info
                    if kicker_team in (0, 1):
                        dir_sign = self._attack_direction.get(kicker_team, 0)

                        # If attack direction still unknown, reject shot to avoid false positives
                        if dir_sign == 0:
                            logger.info(
                                "SHOT_REJECT frame=%d: attack direction unknown for team %s",
                                frame_idx, kicker_team
                            )
                        else:
                            is_correct_direction = False
                            if pitch_ball_xy is not None and prev_pitch is not None:
                                p_vx = (pitch_ball_xy[0] - prev_pitch[0]) / df
                                is_correct_direction = (p_vx * dir_sign) > 0
                            else:
                                is_correct_direction = (v_x * dir_sign) > 0

                            if not is_correct_direction:
                                logger.info(
                                    "SHOT_REJECT frame=%d: wrong direction (v_x=%.1f dir_sign=%d)",
                                    frame_idx, v_x, dir_sign
                                )
                            else:
                                # Field-mode: goal line intersection check
                                is_shot_valid = True
                                if pitch_ball_xy is not None and prev_pitch is not None:
                                    p_vx = (pitch_ball_xy[0] - prev_pitch[0]) / df
                                    p_vy = (pitch_ball_xy[1] - prev_pitch[1]) / df
                                    goal_x = 10500.0 if dir_sign == 1 else 0.0
                                    if abs(p_vx) > 1e-3:
                                        t_goal = (goal_x - pitch_ball_xy[0]) / p_vx
                                        if t_goal > 0:
                                            y_intercept = pitch_ball_xy[1] + p_vy * t_goal
                                            is_shot_valid = (1384.0 <= y_intercept <= 5416.0)
                                        else:
                                            is_shot_valid = False
                                    else:
                                        is_shot_valid = False

                                if is_shot_valid:
                                    speed_str = (
                                        f"{speed_kph:.1f} km/h" if speed_kph is not None
                                        else f"rel {speed_px / avg_player_h:.2f} h/f"
                                    )
                                    logger.info(
                                        "SHOT DETECTED! Frame: %d, Player ID: %d, Team: %d, Speed: %s, Direction: %s",
                                        frame_idx, kicker_id, kicker_team, speed_str,
                                        "right" if dir_sign == 1 else "left"
                                    )
                                    self._active_shot = ShotState(
                                        is_shot=True,
                                        player_id=kicker_id,
                                        team_id=kicker_team,
                                        frame_started=frame_idx,
                                        shot_speed_kph=speed_kph,
                                        direction="right" if dir_sign == 1 else "left",
                                        trajectory=[(prev_bx, prev_by), real_ball_xy],
                                    )
                                    self._shot_countdown = self._shot_display_frames
                                    self._last_kicked_frame = frame_idx
                                    ret_val = self._active_shot

        # Store player tracks for the next frame
        self._last_frame_players = {p.track_id: p.xyxy for p in players}
        self._player_history.append((frame_idx, self._last_frame_players))
        if len(self._player_history) > 60:
            self._player_history.pop(0)

        return ret_val if ret_val is not None else ShotState()
