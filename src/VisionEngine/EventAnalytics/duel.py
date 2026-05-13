"""Detect duels (ikili mücadele) between opposing players near the ball."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from VisionEngine.config.settings import Settings
from VisionEngine.schemas.schema import FrameTracks, ObjectRole
from VisionEngine.EventAnalytics.possession import PossessionState


@dataclass
class DuelState:
    """Represents an ongoing duel."""

    is_duel: bool = False
    player_1_id: Optional[int] = None
    player_2_id: Optional[int] = None
    team_1_id: Optional[int] = None
    team_2_id: Optional[int] = None
    center_xy: Optional[tuple[float, float]] = None


class DuelEstimator:
    """Finds situations where opposing players fight for the ball."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        # Distance between two players to be considered in a physical duel (tightened to 75)
        self._duel_max_player_dist_px = getattr(settings, "duel_max_player_dist_px", 75.0)
        # Base allowable distance from the duel center to the ball (tightened to 90)
        self._duel_max_ball_dist_px = getattr(settings, "duel_max_ball_dist_px", 90.0)
        
        # Stateful ball tracking memory
        self._last_known_ball_xy: Optional[tuple[float, float]] = None
        self._frames_since_ball_seen: int = 0
        self._max_ball_memory_frames = 60  # Forget the ball location after 2 seconds at 30fps
        
        # Temporal smoothing to prevent transient overlap (e.g., pass & run) false positives
        self._active_pair: Optional[tuple[int, int]] = None
        self._consecutive_frames: int = 0
        self._min_duel_frames = 4  # Require 4 consecutive frames (~0.13 sec) of clashing to confirm

    def reset(self) -> None:
        self._last_known_ball_xy = None
        self._frames_since_ball_seen = 0
        self._active_pair = None
        self._consecutive_frames = 0

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
    ) -> DuelState:
        """Identify if there is an active duel this frame near the ball."""
        # 1. Update stateful ball tracking memory
        ball_xy = None
        if possession.ball_xy is not None:
            ball_xy = possession.ball_xy
        elif estimated_ball_center is not None:
            ball_xy = estimated_ball_center

        if ball_xy is not None:
            self._last_known_ball_xy = ball_xy
            self._frames_since_ball_seen = 0
        else:
            self._frames_since_ball_seen += 1

        # Helper to reset state and return empty
        def _no_duel() -> DuelState:
            self._active_pair = None
            self._consecutive_frames = 0
            return DuelState()

        # If the ball is lost for too long or was never seen, we cannot declare a duel
        if self._last_known_ball_xy is None or self._frames_since_ball_seen > self._max_ball_memory_frames:
            return _no_duel()

        # 2. Gather players by team
        players = [i for i in tracks.instances if i.role is ObjectRole.PLAYER and i.track_id >= 0]
        team_0_players = []
        team_1_players = []
        for p in players:
            tid = track_to_team.get(p.track_id)
            if tid == 0:
                team_0_players.append(p)
            elif tid == 1:
                team_1_players.append(p)

        if not team_0_players or not team_1_players:
            return _no_duel()

        active_ball_xy = self._last_known_ball_xy

        best_candidate = None
        min_score = float("inf")

        # 3. Find opposing players who are physically clashing NEAR the active ball location
        for p0 in team_0_players:
            # Get height for perspective-aware dynamic thresholding
            h0 = p0.xyxy[3] - p0.xyxy[1]
            c0 = self._bbox_center(p0.xyxy)
            
            for p1 in team_1_players:
                h1 = p1.xyxy[3] - p1.xyxy[1]
                c1 = self._bbox_center(p1.xyxy)
                
                avg_h = 0.5 * (h0 + h1)
                
                # DYNAMIC THRESHOLDS based on perspective (player height)
                # A player is close to the camera -> height is large -> larger pixel threshold.
                # A player is far -> height is small -> smaller pixel threshold.
                dynamic_max_player_dist = avg_h * 0.85  # Max dist is ~85% of their projected height
                dynamic_max_ball_dist = avg_h * 1.1    # Base ball dist is ~110% of their projected height
                
                # Add ball memory uncertainty decay (2 pixels per missing frame)
                effective_max_ball_dist = dynamic_max_ball_dist + (self._frames_since_ball_seen * 2.0)
                
                p2p_dist = float(np.hypot(c0[0] - c1[0], c0[1] - c1[1]))
                
                # First condition: Opponents must be physically clashing
                if p2p_dist <= dynamic_max_player_dist:
                    duel_cx = (c0[0] + c1[0]) / 2.0
                    duel_cy = (c0[1] + c1[1]) / 2.0
                    
                    dist_to_ball = float(np.hypot(duel_cx - active_ball_xy[0], duel_cy - active_ball_xy[1]))
                    
                    # Second condition: The clash center must be near the ball memory
                    if dist_to_ball <= effective_max_ball_dist:
                        # Normalize the score by local scale so far-away and near duels are evaluated fairly
                        score = (p2p_dist / avg_h) + (dist_to_ball / avg_h * 0.5)
                        
                        if score < min_score:
                            min_score = score
                            best_candidate = DuelState(
                                is_duel=True,
                                player_1_id=p0.track_id,
                                team_1_id=0,
                                player_2_id=p1.track_id,
                                team_2_id=1,
                                center_xy=(duel_cx, duel_cy),
                            )

        # 4. Apply Temporal Smoothing to filter out transient/passing overlaps
        if best_candidate is not None:
            current_pair = tuple(sorted([best_candidate.player_1_id, best_candidate.player_2_id]))
            
            if current_pair == self._active_pair:
                self._consecutive_frames += 1
            else:
                self._active_pair = current_pair
                self._consecutive_frames = 1
                
            if self._consecutive_frames >= self._min_duel_frames:
                return best_candidate
            else:
                return DuelState(is_duel=False)
        else:
            return _no_duel()
