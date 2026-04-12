"""Ball possession by nearest player and team."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from tactivision.config.settings import Settings
from tactivision.tracking.tracker import TrackedFrame


@dataclass
class PossessionState:
    """Who has the ball this frame (optional temporal smoothing later)."""

    team_id: Optional[int] = None
    """0 or 1 for team A/B; None if unknown."""

    track_id: Optional[int] = None
    """Closest player track to the ball, if any."""

    ball_xy: Optional[tuple[float, float]] = None
    """Ball center in image coordinates, if detected."""


class PossessionEstimator:
    """Find ball detection, nearest player within radius, map to team."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def reset(self) -> None:
        pass

    def update(
        self,
        frame: np.ndarray,
        tracked: TrackedFrame,
        track_to_team: dict[int, int],
        ball_class_id: int = 32,
    ) -> PossessionState:
        """
        Compute possession for the current frame.

        ``ball_class_id`` is COCO ball class when using default YOLO COCO weights;
        override when using a custom football dataset.
        """
        # TODO: filter tracked.cls for ball, compute centroid, find nearest player bbox center
        return PossessionState()
