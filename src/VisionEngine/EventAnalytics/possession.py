"""Ball possession by nearest player and team."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from VisionEngine.config.settings import Settings
from VisionEngine.schemas.schema import FrameTracks, ObjectRole


@dataclass
class PossessionState:
    """Who has the ball this frame (optional temporal smoothing later)."""

    team_id: Optional[int] = None
    """0 or 1 for team A/B; None if unknown."""

    track_id: Optional[int] = None
    """Closest player track to the ball, if any."""

    ball_xy: Optional[tuple[float, float]] = None
    """Ball center in image coordinates, if detected."""


def _bbox_center(xyxy: tuple[float, float, float, float]) -> tuple[float, float]:
    x1, y1, x2, y2 = xyxy
    return (0.5 * (x1 + x2), 0.5 * (y1 + y2))


class PossessionEstimator:
    """Find ball detection, nearest player within radius, map to team."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def reset(self) -> None:
        pass

    def update(
        self,
        _frame: np.ndarray,
        tracks: FrameTracks,
        track_to_team: dict[int, int],
    ) -> PossessionState:
        """
        Compute possession for the current frame using semantic roles.

        Uses ``BALL`` for the ball hypothesis and possesses among ``PLAYER``
        tracks only (excludes ``REFEREE``, ``OTHER``). Goalkeepers modeled as ``PLAYER``.
        """
        balls = [i for i in tracks.instances if i.role is ObjectRole.BALL]
        # Only players can possess; exclude referee (and implicitly goalkeepers would use PLAYER).
        _poss_roles = frozenset({ObjectRole.PLAYER})
        candidates = [i for i in tracks.instances if i.role in _poss_roles]

        if not balls:
            return PossessionState()

        # If multiple ball hypotheses, take highest confidence
        ball = max(balls, key=lambda b: b.confidence)
        bx, by = _bbox_center(ball.xyxy)
        ball_xy = (bx, by)

        best_tid: Optional[int] = None
        best_d2 = float(self._settings.possession_proximity_px) ** 2

        for pl in candidates:
            if pl.track_id < 0:
                continue
            px, py = _bbox_center(pl.xyxy)
            dx, dy = px - bx, py - by
            d2 = dx * dx + dy * dy
            if d2 <= best_d2:
                best_d2 = d2
                best_tid = pl.track_id

        if best_tid is None:
            return PossessionState(ball_xy=ball_xy)

        team = track_to_team.get(best_tid)
        return PossessionState(team_id=team, track_id=best_tid, ball_xy=ball_xy)
