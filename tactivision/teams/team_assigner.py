"""Assign each track to team A/B using jersey color clustering (K-means)."""

from __future__ import annotations

from typing import Optional

import numpy as np

from tactivision.config.settings import Settings
from tactivision.tracking.tracker import TrackedFrame


class TeamAssigner:
    """
    Samples pixels inside player bounding boxes, clusters into two dominant colors,
    maps clusters to team IDs, and returns a stable track_id -> team_id mapping.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._track_team: dict[int, int] = {}

    def reset(self) -> None:
        self._track_team.clear()

    def update(self, frame: np.ndarray, tracked: TrackedFrame) -> dict[int, int]:
        """
        Update team labels from current frame and tracks.

        Returns a copy of the current mapping ``track_id -> team_id`` (0/1).
        """
        # TODO: crop ROIs, KMeans(n_clusters=2), assign clusters to teams, smooth over time
        return dict(self._track_team)

    def team_for_track(self, track_id: int) -> Optional[int]:
        return self._track_team.get(track_id)
