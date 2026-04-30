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
from dataclasses import dataclass
from typing import Optional

import numpy as np
from sklearn.cluster import KMeans

from VisionEngine.config.settings import Settings
from VisionEngine.schemas.schema import FrameTracks, ObjectRole, TrackedInstance


@dataclass(slots=True)
class LostTrackRecord:
    lost_track_id: int
    team_id: int
    last_center_xy: tuple[float, float]
    lost_at_frame_index: int


class TeamClassifier:
    """Assign stable team ``0``/``1`` labels from jersey colors."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._team_model: Optional[KMeans] = None
        self._track_to_team: dict[int, int] = {}
        self._track_to_color: dict[int, np.ndarray] = {}
        # Persistent map: raw ByteTrack ID → effective (restored) track ID.
        # Entries are added when a new raw ID is matched to a lost-track record
        # and removed when that raw ID disappears from tracking.
        self._raw_to_effective: dict[int, int] = {}
        self._prev_visible_effective_ids: set[int] = set()
        self._last_center_by_effective: dict[int, tuple[float, float]] = {}
        self._lost_track_cache: list[LostTrackRecord] = []
        self._lost_track_ttl_frames = 5
        self._lost_track_match_distance_px = 150.0

    def reset(self) -> None:
        self._team_model = None
        self._track_to_team.clear()
        self._track_to_color.clear()
        self._raw_to_effective.clear()
        self._prev_visible_effective_ids.clear()
        self._last_center_by_effective.clear()
        self._lost_track_cache.clear()

    @property
    def track_id_remap(self) -> dict[int, int]:
        """Return current ``raw_track_id -> effective_track_id`` map for display relabeling."""
        return dict(self._raw_to_effective)

    def _bbox_center(self, xyxy: tuple[float, float, float, float]) -> tuple[float, float]:
        x1, y1, x2, y2 = xyxy
        return ((x1 + x2) * 0.5, (y1 + y2) * 0.5)

    def _predict_team_label(self, player_color: Optional[np.ndarray]) -> Optional[int]:
        if player_color is None or self._team_model is None:
            return None
        try:
            return int(self._team_model.predict(player_color.reshape(1, -1))[0])
        except Exception:
            return None

    def _prune_lost_track_cache(self, frame_index: int) -> None:
        self._lost_track_cache = [
            rec
            for rec in self._lost_track_cache
            if (frame_index - rec.lost_at_frame_index) <= self._lost_track_ttl_frames
        ]

    def _capture_newly_lost_tracks(self, current_effective_ids: set[int], frame_index: int) -> None:
        """Add records for players that were visible last frame but are gone now (by effective ID)."""
        newly_lost = self._prev_visible_effective_ids - current_effective_ids
        for eff_id in sorted(newly_lost):
            team_id = self._track_to_team.get(eff_id)
            center = self._last_center_by_effective.get(eff_id)
            if team_id is None or center is None:
                continue
            # Avoid duplicate entries for the same effective ID.
            already = any(r.lost_track_id == eff_id for r in self._lost_track_cache)
            if not already:
                self._lost_track_cache.append(
                    LostTrackRecord(
                        lost_track_id=eff_id,
                        team_id=team_id,
                        last_center_xy=center,
                        lost_at_frame_index=frame_index,
                    )
                )

    def _match_lost_track_id(
        self,
        *,
        team_id: int,
        center_xy: tuple[float, float],
        active_effective_ids: set[int],
        consumed_lost_ids: set[int],
    ) -> Optional[int]:
        """Return the closest lost-track ID of the same team within the distance threshold."""
        best_id: Optional[int] = None
        best_dist = float("inf")
        cx, cy = center_xy
        for rec in self._lost_track_cache:
            if rec.team_id != team_id:
                continue
            if rec.lost_track_id in active_effective_ids:
                continue
            if rec.lost_track_id in consumed_lost_ids:
                continue
            dx = cx - rec.last_center_xy[0]
            dy = cy - rec.last_center_xy[1]
            dist = float(np.hypot(dx, dy))
            if dist > self._lost_track_match_distance_px:
                continue
            if dist < best_dist:
                best_dist = dist
                best_id = rec.lost_track_id
        return best_id

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
        """Return stable ``effective_track_id -> team_id`` mapping for tracked players."""
        if not self._settings.team_classification_enabled:
            return {}

        frame_index = int(tracks.frame_index)
        self._prune_lost_track_cache(frame_index)
        players = self._player_instances(tracks)
        current_raw_ids: set[int] = {inst.track_id for inst in players}

        # --- Step 1: extract jersey colors for raw IDs not yet in any color store ---
        for inst in players:
            raw_id = inst.track_id
            eff_id = self._raw_to_effective.get(raw_id, raw_id)
            if eff_id in self._track_to_team:
                continue
            if raw_id in self._track_to_color:
                continue
            color = self.get_player_color(frame, inst)
            if color is None:
                continue
            self._track_to_color[raw_id] = color

        # --- Step 2: fit global team model once enough features are ready ---
        if self._team_model is None:
            self.assign_team_color(list(self._track_to_color.values()))

        # --- Step 3: assign teams; resolve lost-track matches for new players ---
        current_effective_ids: set[int] = set()
        consumed_lost_ids: set[int] = set()

        if self._team_model is not None:
            for inst in players:
                raw_id = inst.track_id
                center_xy = self._bbox_center(inst.xyxy)
                eff_id = self._raw_to_effective.get(raw_id, raw_id)

                if eff_id in self._track_to_team:
                    # Already assigned — just update position tracking.
                    current_effective_ids.add(eff_id)
                    self._last_center_by_effective[eff_id] = center_xy
                    continue

                # New player (not yet assigned under any effective ID).
                player_color = self._track_to_color.get(raw_id)
                predicted_team = self._predict_team_label(player_color)
                if predicted_team is None:
                    continue

                # Try to recover a lost-track ID from cache (skip on frame 0).
                if frame_index > 0:
                    matched_lost_id = self._match_lost_track_id(
                        team_id=predicted_team,
                        center_xy=center_xy,
                        active_effective_ids=current_effective_ids,
                        consumed_lost_ids=consumed_lost_ids,
                    )
                    if matched_lost_id is not None:
                        eff_id = matched_lost_id
                        self._raw_to_effective[raw_id] = eff_id
                        consumed_lost_ids.add(matched_lost_id)
                        # Carry color over to effective ID so future frames keep it.
                        if player_color is not None:
                            self._track_to_color[eff_id] = player_color

                # Persist team under effective ID only (NOT raw ID when remapped).
                self._track_to_team[eff_id] = predicted_team
                current_effective_ids.add(eff_id)
                self._last_center_by_effective[eff_id] = center_xy

        # Ensure all already-assigned players (team model warming-up path) are
        # counted in current_effective_ids.
        for inst in players:
            raw_id = inst.track_id
            eff_id = self._raw_to_effective.get(raw_id, raw_id)
            if eff_id in self._track_to_team:
                current_effective_ids.add(eff_id)
                self._last_center_by_effective[eff_id] = self._bbox_center(inst.xyxy)

        # Remove consumed lost records so they cannot be reused.
        if consumed_lost_ids:
            self._lost_track_cache = [
                rec for rec in self._lost_track_cache if rec.lost_track_id not in consumed_lost_ids
            ]

        # Capture tracks that disappeared this frame (keyed by effective ID).
        self._capture_newly_lost_tracks(current_effective_ids, frame_index)
        self._prev_visible_effective_ids = set(current_effective_ids)

        # Expire remap entries whose raw ID is no longer tracked.
        for raw_id in list(self._raw_to_effective.keys()):
            if raw_id not in current_raw_ids:
                del self._raw_to_effective[raw_id]

        # Return effective_id -> team mapping. Renderer/possession consume this.
        return dict(self._track_to_team)
