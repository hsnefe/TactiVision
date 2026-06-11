"""Detect goal events by combining shot detection, ball disappearance, and celebration signals."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
import logging
from typing import Optional

import numpy as np

from VisionEngine.config.settings import Settings
from VisionEngine.schemas.schema import FrameTracks, ObjectRole
from VisionEngine.EventAnalytics.possession import PossessionState
from VisionEngine.EventAnalytics.shot import ShotState

logger = logging.getLogger(__name__)


@dataclass
class GoalState:
    """Represents a goal event currently being visualized."""

    is_goal: bool = False
    scorer_id: Optional[int] = None
    team_id: Optional[int] = None
    frame_detected: Optional[int] = None


class _Phase(Enum):
    """Internal state machine phases."""
    IDLE = auto()
    MONITORING = auto()
    DISPLAYING = auto()


class GoalEstimator:
    """Detects goals by monitoring signals after a shot is triggered.

    Signals:
        1. Shot trigger (required gate from ShotEstimator)
        2. Ball disappearance after the shot
        3. Player celebration clustering (same-team players grouping)
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._monitor_window = getattr(settings, "goal_monitor_window", 60)
        self._ball_missing_ratio = getattr(settings, "goal_ball_missing_ratio", 0.5)
        self._display_frames = getattr(settings, "goal_display_frames", 90)
        self._celebration_min_players = getattr(settings, "goal_celebration_min_players", 3)

        # State machine
        self._phase = _Phase.IDLE

        # Monitoring state (active during MONITORING phase)
        self._shot_frame: int = -1
        self._shot_player_id: Optional[int] = None
        self._shot_team_id: Optional[int] = None
        self._monitor_start_frame: int = -1
        self._ball_present_count: int = 0
        self._ball_missing_count: int = 0
        self._celebration_detected: bool = False

        # Baseline team spread (average pairwise distance before the shot)
        self._pre_shot_team_spread: Optional[float] = None

        # Display state (active during DISPLAYING phase)
        self._active_goal: Optional[GoalState] = None
        self._display_countdown: int = 0

    def reset(self) -> None:
        self._phase = _Phase.IDLE
        self._shot_frame = -1
        self._shot_player_id = None
        self._shot_team_id = None
        self._monitor_start_frame = -1
        self._ball_present_count = 0
        self._ball_missing_count = 0
        self._celebration_detected = False
        self._pre_shot_team_spread = None
        self._active_goal = None
        self._display_countdown = 0

    def _compute_team_spread(
        self,
        tracks: FrameTracks,
        track_to_team: dict[int, int],
        team_id: int,
    ) -> Optional[float]:
        """Compute average pairwise distance between players of a given team."""
        positions: list[tuple[float, float]] = []
        for inst in tracks.instances:
            if inst.role is ObjectRole.PLAYER and inst.track_id >= 0:
                if track_to_team.get(inst.track_id) == team_id:
                    cx = 0.5 * (inst.xyxy[0] + inst.xyxy[2])
                    cy = 0.5 * (inst.xyxy[1] + inst.xyxy[3])
                    positions.append((cx, cy))

        if len(positions) < 2:
            return None

        total_dist = 0.0
        count = 0
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                dx = positions[i][0] - positions[j][0]
                dy = positions[i][1] - positions[j][1]
                total_dist += float(np.hypot(dx, dy))
                count += 1

        return total_dist / count if count > 0 else None

    def _count_celebration_cluster(
        self,
        tracks: FrameTracks,
        track_to_team: dict[int, int],
        team_id: int,
    ) -> int:
        """Count the max number of same-team players within close proximity of each other."""
        positions: list[tuple[float, float]] = []
        for inst in tracks.instances:
            if inst.role is ObjectRole.PLAYER and inst.track_id >= 0:
                if track_to_team.get(inst.track_id) == team_id:
                    cx = 0.5 * (inst.xyxy[0] + inst.xyxy[2])
                    cy = 0.5 * (inst.xyxy[1] + inst.xyxy[3])
                    positions.append((cx, cy))

        if len(positions) < 2:
            return 0

        # Calculate average player height for proximity threshold
        player_heights: list[float] = []
        for inst in tracks.instances:
            if inst.role is ObjectRole.PLAYER and inst.track_id >= 0:
                h = inst.xyxy[3] - inst.xyxy[1]
                if h > 0:
                    player_heights.append(h)
        avg_h = float(np.mean(player_heights)) if player_heights else 100.0

        # Proximity threshold: 2x player height
        threshold = avg_h * 2.0

        # Find the largest cluster using greedy approach
        best_cluster = 0
        for i, (cx, cy) in enumerate(positions):
            cluster_count = 1
            for j, (ox, oy) in enumerate(positions):
                if i == j:
                    continue
                if float(np.hypot(cx - ox, cy - oy)) < threshold:
                    cluster_count += 1
            best_cluster = max(best_cluster, cluster_count)

        return best_cluster

    def update(
        self,
        frame: np.ndarray,
        tracks: FrameTracks,
        track_to_team: dict[int, int],
        possession: PossessionState,
        shot_state: ShotState,
        field_mapper=None,
    ) -> GoalState:
        """Update goal detection state for the current frame."""
        frame_idx = tracks.frame_index

        # ── PHASE: DISPLAYING ──
        if self._phase is _Phase.DISPLAYING:
            self._display_countdown -= 1
            if self._display_countdown <= 0:
                self._active_goal = None
                self._phase = _Phase.IDLE
            if self._active_goal is not None:
                return self._active_goal

        # ── PHASE: IDLE → check for new shot trigger ──
        if self._phase is _Phase.IDLE:
            if shot_state.is_shot and shot_state.frame_started == frame_idx:
                # A new shot just started — begin monitoring
                self._phase = _Phase.MONITORING
                self._shot_frame = frame_idx
                self._shot_player_id = shot_state.player_id
                self._shot_team_id = shot_state.team_id
                self._monitor_start_frame = frame_idx
                self._ball_present_count = 0
                self._ball_missing_count = 0
                self._celebration_detected = False
                self._stationary_detected = False
                self._ball_positions = []

                # Record pre-shot team spread as baseline
                if self._shot_team_id is not None:
                    self._pre_shot_team_spread = self._compute_team_spread(
                        tracks, track_to_team, self._shot_team_id
                    )
                logger.info(
                    "GOAL_MONITOR: Started monitoring after shot at frame %d (player=%s team=%s spread=%.0f)",
                    frame_idx,
                    self._shot_player_id,
                    self._shot_team_id,
                    self._pre_shot_team_spread or 0,
                )

        # ── PHASE: MONITORING ──
        if self._phase is _Phase.MONITORING:
            frames_elapsed = frame_idx - self._monitor_start_frame

            # Track ball presence/absence
            ball_detected = possession.ball_xy is not None
            if ball_detected:
                self._ball_present_count += 1
                self._ball_positions.append(possession.ball_xy)
                
                # Check stationary ball (trapped in net)
                if len(self._ball_positions) >= 30 and not self._stationary_detected:
                    recent_positions = self._ball_positions[-30:]
                    xs = [p[0] for p in recent_positions]
                    ys = [p[1] for p in recent_positions]
                    if (max(xs) - min(xs)) < 60 and (max(ys) - min(ys)) < 60:
                        self._stationary_detected = True
                        logger.info("GOAL_SIGNAL: Stationary ball detected at frame %d", frame_idx)

                # Check if ball is physically inside the goal net via homography
                if field_mapper is not None and field_mapper.is_configured:
                    pitch_ball_xy = field_mapper.image_to_field(possession.ball_xy)
                    if pitch_ball_xy is not None:
                        bx, by = pitch_ball_xy
                        # Pitch width is 6800. Goal center is 3400. Frame is 732cm. Use 2900-3900 margin.
                        if frame_idx % 5 == 0:
                            logger.info("GOAL_GEOMETRY DEBUG frame=%d: bx=%.0f, by=%.0f", frame_idx, bx, by)
                        if 2900 <= by <= 3900 and (bx < 0 or bx > 10500):
                            logger.info(
                                "GOAL DETECTED (GEOMETRY)! Frame: %d, Scorer: %s, Team: %s, Pos: (%.0f, %.0f)",
                                frame_idx, self._shot_player_id, self._shot_team_id, bx, by
                            )
                            self._active_goal = GoalState(
                                is_goal=True,
                                scorer_id=self._shot_player_id,
                                team_id=self._shot_team_id,
                                frame_detected=frame_idx,
                            )
                            self._display_countdown = self._display_frames
                            self._phase = _Phase.DISPLAYING
                            return self._active_goal
            else:
                self._ball_missing_count += 1

            # Check celebration clustering
            if self._shot_team_id is not None and not self._celebration_detected:
                cluster_size = self._count_celebration_cluster(
                    tracks, track_to_team, self._shot_team_id
                )
                if cluster_size >= self._celebration_min_players:
                    # Also check if spread has decreased compared to baseline
                    current_spread = self._compute_team_spread(
                        tracks, track_to_team, self._shot_team_id
                    )
                    spread_decreased = True
                    if self._pre_shot_team_spread is not None and current_spread is not None:
                        spread_decreased = current_spread < self._pre_shot_team_spread * 0.85
                    
                    if spread_decreased:
                        self._celebration_detected = True
                        logger.info(
                            "GOAL_SIGNAL: Celebration cluster detected at frame %d (cluster=%d spread=%.0f->%.0f)",
                            frame_idx, cluster_size,
                            self._pre_shot_team_spread or 0,
                            current_spread or 0,
                        )

            # Check if monitoring window expired
            if frames_elapsed >= self._monitor_window:
                total_frames = self._ball_present_count + self._ball_missing_count
                missing_ratio = (
                    self._ball_missing_count / total_frames if total_frames > 0 else 0
                )

                ball_signal = missing_ratio >= self._ball_missing_ratio
                celebration_signal = self._celebration_detected

                logger.info(
                    "GOAL_EVAL frame=%d: ball_missing=%.1f%% (need %.0f%%) celebration=%s",
                    frame_idx,
                    missing_ratio * 100,
                    self._ball_missing_ratio * 100,
                    celebration_signal,
                )

                # Goal confirmed if: celebration detected, stationary ball in net, or missing ball
                if celebration_signal or self._stationary_detected or missing_ratio >= 0.80:
                    logger.info(
                        "GOAL DETECTED! Frame: %d, Scorer: %s, Team: %s",
                        frame_idx, self._shot_player_id, self._shot_team_id,
                    )
                    self._active_goal = GoalState(
                        is_goal=True,
                        scorer_id=self._shot_player_id,
                        team_id=self._shot_team_id,
                        frame_detected=frame_idx,
                    )
                    self._display_countdown = self._display_frames
                    self._phase = _Phase.DISPLAYING
                    return self._active_goal
                else:
                    logger.info("GOAL_REJECT: Signals insufficient at frame %d", frame_idx)
                    self._phase = _Phase.IDLE

        return GoalState()
