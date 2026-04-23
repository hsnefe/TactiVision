"""Tutorial-style two-stage KMeans team assignment for tracked players.

Pipeline (per frame):

1. Keep only tracked :class:`~VisionEngine.schemas.schema.ObjectRole` ``PLAYER``
   instances with ``track_id >= 0``.
2. Crop each player bbox from the frame and use only the top half.
3. Run local ``KMeans(n_clusters=2)`` on top-half pixels.
4. Use the four corner labels to identify the background cluster
   (majority corner label). The other cluster is treated as jersey.
5. Use the jersey-cluster center as that player's color feature.
6. Collect player features and fit a second global ``KMeans(n_clusters=2)``
   team-color model.
7. Predict and persist ``track_id -> team_id``. Once assigned, a track id is
   not reassigned until ``reset()``.
"""

from __future__ import annotations

from collections import Counter
from typing import Optional

import numpy as np
from sklearn.cluster import KMeans

from VisionEngine.config.settings import Settings
from VisionEngine.schemas.schema import FrameTracks, ObjectRole, TrackedInstance


class TeamClassifier:
    """Assign stable team ``0``/``1`` labels from jersey colors."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._team_model: Optional[KMeans] = None
        self._track_to_team: dict[int, int] = {}
        self._track_to_color: dict[int, np.ndarray] = {}

    def reset(self) -> None:
        self._team_model = None
        self._track_to_team.clear()
        self._track_to_color.clear()

    def _player_instances(self, tracks: FrameTracks) -> list[TrackedInstance]:
        out: list[TrackedInstance] = []
        for inst in tracks.instances:
            if inst.role is not ObjectRole.PLAYER:
                continue
            if inst.track_id < 0:
                continue
            out.append(inst)
        return out

    def _crop_player(self, frame: np.ndarray, xyxy: tuple[float, float, float, float]) -> Optional[np.ndarray]:
        """Return full player crop from frame, or ``None`` when invalid/tiny."""
        h_img, w_img = frame.shape[:2]
        x1, y1, x2, y2 = xyxy
        xa, ya = max(0, int(x1)), max(0, int(y1))
        xb, yb = min(w_img, int(x2)), min(h_img, int(y2))
        if xb <= xa or yb <= ya:
            return None
        # Defensive minimum size checks.
        if (xb - xa) < 2 or (yb - ya) < 4:
            return None
        crop = frame[ya:yb, xa:xb]
        if crop.size == 0:
            return None
        return crop.copy()

    def _top_half(self, crop: np.ndarray) -> Optional[np.ndarray]:
        if crop.size == 0:
            return None
        h, w = crop.shape[:2]
        if h < 2 or w < 2:
            return None
        half_h = h // 2
        if half_h < 1:
            return None
        top = crop[:half_h, :]
        if top.size == 0 or top.shape[0] < 1 or top.shape[1] < 2:
            return None
        return top

    def get_clustering_model(
        self,
        pixels: np.ndarray,
        n_clusters: int = 2,
    ) -> Optional[KMeans]:
        """Fit KMeans defensively; return ``None`` on invalid data/failure."""
        if pixels.size == 0:
            return None
        if pixels.ndim != 2:
            return None
        if pixels.shape[0] < n_clusters:
            return None
        if pixels.shape[1] != 3:
            return None
        try:
            model = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
            model.fit(pixels)
        except Exception:
            return None
        if not hasattr(model, "cluster_centers_") or model.cluster_centers_.shape[0] != n_clusters:
            return None
        return model

    def get_player_color(self, frame: np.ndarray, inst: TrackedInstance) -> Optional[np.ndarray]:
        """Extract one player's jersey color via top-half local KMeans."""
        crop = self._crop_player(frame, inst.xyxy)
        if crop is None:
            return None
        top = self._top_half(crop)
        if top is None:
            return None

        h, w = top.shape[:2]
        pixels = top.reshape(-1, 3).astype(np.float32)
        model = self.get_clustering_model(pixels, n_clusters=2)
        if model is None:
            return None

        labels = model.labels_.reshape(h, w)
        corner_labels = [
            int(labels[0, 0]),
            int(labels[0, w - 1]),
            int(labels[h - 1, 0]),
            int(labels[h - 1, w - 1]),
        ]
        bg_cluster = Counter(corner_labels).most_common(1)[0][0]
        if bg_cluster not in (0, 1):
            return None
        jersey_cluster = 1 - bg_cluster
        if jersey_cluster not in (0, 1):
            return None

        jersey_mask = model.labels_ == jersey_cluster
        if int(np.count_nonzero(jersey_mask)) == 0:
            return None

        center = model.cluster_centers_[jersey_cluster].astype(np.float32)
        if center.shape != (3,):
            return None
        return center

    def assign_team_color(self, player_colors: list[np.ndarray]) -> bool:
        """Fit global team-color KMeans from collected player jersey colors."""
        if len(player_colors) < 2:
            return False
        X = np.stack(player_colors, axis=0).astype(np.float32)
        model = self.get_clustering_model(X, n_clusters=2)
        if model is None:
            return False
        self._team_model = model
        return True

    def get_player_team(self, track_id: int, player_color: Optional[np.ndarray]) -> Optional[int]:
        """Return stable team assignment for a track id."""
        if track_id in self._track_to_team:
            return self._track_to_team[track_id]
        if player_color is None:
            return None
        if self._team_model is None:
            return None
        try:
            label = int(self._team_model.predict(player_color.reshape(1, -1))[0])
        except Exception:
            return None
        self._track_to_team[track_id] = label
        return label

    def update(self, frame: np.ndarray, tracks: FrameTracks) -> dict[int, int]:
        """Return stable ``track_id -> team_id`` mapping for tracked players."""
        if not self._settings.team_classification_enabled:
            return {}

        players = self._player_instances(tracks)

        for inst in players:
            if inst.track_id in self._track_to_team:
                continue
            color = self.get_player_color(frame, inst)
            if color is None:
                continue
            self._track_to_color[inst.track_id] = color

        # Fit global team model once enough player features are collected.
        if self._team_model is None:
            self.assign_team_color(list(self._track_to_color.values()))

        # Assign only unassigned visible players; previously assigned track ids are kept.
        if self._team_model is not None:
            for inst in players:
                if inst.track_id in self._track_to_team:
                    continue
                player_color = self._track_to_color.get(inst.track_id)
                self.get_player_team(inst.track_id, player_color)

        # Return known team assignments. Renderer/possession consume visible track ids.
        return dict(self._track_to_team)
