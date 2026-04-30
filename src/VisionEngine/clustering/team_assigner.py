"""Facade for jersey-based team assignment; delegates to :class:`TeamClassifier`."""

from __future__ import annotations

from typing import Optional

import numpy as np
from sklearn.cluster import KMeans

from VisionEngine.clustering.team_classifier import TeamClassifier
from VisionEngine.config.settings import Settings
from VisionEngine.schemas.schema import FrameTracks, TrackedInstance


class TeamAssigner:
    """
    Samples pixels inside player bounding boxes, clusters into two dominant colors,
    maps clusters to team IDs, and returns a stable track_id -> team_id mapping.

    Implementation lives in :class:`~VisionEngine.clustering.team_classifier.TeamClassifier`.
    """

    def __init__(self, settings: Settings) -> None:
        self._classifier = TeamClassifier(settings)
        self._last_teams: dict[int, int] = {}

    def reset(self) -> None:
        self._classifier.reset()
        self._last_teams.clear()

    def update(self, frame: np.ndarray, tracks: FrameTracks) -> dict[int, int]:
        """
        Update team labels from current frame and tracks.

        Only :class:`~VisionEngine.schemas.schema.ObjectRole` ``PLAYER`` instances
        are classified; referees and ball are ignored.
        """
        self._last_teams = self._classifier.update(frame, tracks)
        return dict(self._last_teams)

    def get_clustering_model(
        self,
        pixels: np.ndarray,
        n_clusters: int = 2,
    ) -> Optional[KMeans]:
        """Expose clustering helper used by the classifier pipeline."""
        return self._classifier.get_clustering_model(pixels, n_clusters=n_clusters)

    def get_player_color(self, frame: np.ndarray, inst: TrackedInstance) -> Optional[np.ndarray]:
        """Expose per-player jersey feature extraction helper."""
        return self._classifier.get_player_color(frame, inst)

    def assign_team_color(self, player_colors: list[np.ndarray]) -> bool:
        """Expose global team-color model fitting helper."""
        return self._classifier.assign_team_color(player_colors)

    def get_player_team(self, track_id: int, player_color: Optional[np.ndarray]) -> Optional[int]:
        """Expose stable team lookup/assignment helper."""
        return self._classifier.get_player_team(track_id, player_color)

    def team_for_track(self, track_id: int) -> Optional[int]:
        return self._last_teams.get(track_id)

    @property
    def track_id_remap(self) -> dict[int, int]:
        """Return ``raw_track_id -> effective_track_id`` map built by lost-track recovery."""
        return self._classifier.track_id_remap
