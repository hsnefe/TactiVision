"""Facade for jersey-based team assignment; delegates to :class:`TeamClassifier`."""

from __future__ import annotations

from typing import Optional

from tactivision.config.settings import Settings
from tactivision.teams.team_classifier import TeamClassifier
from tactivision.tracking.schema import FrameTracks


class TeamAssigner:
    """
    Samples pixels inside player bounding boxes, clusters into two dominant colors,
    maps clusters to team IDs, and returns a stable track_id -> team_id mapping.

    Implementation lives in :class:`~tactivision.teams.team_classifier.TeamClassifier`.
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

        Only :class:`~tactivision.tracking.schema.ObjectRole` ``PLAYER`` instances
        are classified; referees and ball are ignored.
        """
        self._last_teams = self._classifier.update(frame, tracks)
        return dict(self._last_teams)

    def team_for_track(self, track_id: int) -> Optional[int]:
        return self._last_teams.get(track_id)
