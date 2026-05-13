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

import cv2
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


def _upper_torso_crop(
    frame: np.ndarray,
    xyxy: tuple[float, float, float, float],
    y0_ratio: float,
    y1_ratio: float,
) -> Optional[np.ndarray]:
    """Return BGR crop of upper torso region; None if invalid."""
    h_img, w_img = frame.shape[:2]
    x1, y1, x2, y2 = xyxy
    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2, y2 = min(w_img - 1, int(x2)), min(h_img - 1, int(y2))
    if x2 <= x1 or y2 <= y1:
        return None
    bh = y2 - y1
    ya = int(y1 + bh * y0_ratio)
    yb = int(y1 + bh * y1_ratio)
    ya = max(y1, min(ya, y2 - 1))
    yb = max(ya + 1, min(yb, y2))
    return frame[ya:yb, x1:x2].copy()


def _grass_suppressed_mean_lab(
    bgr_crop: np.ndarray,
    lower_hsv: tuple[int, int, int],
    upper_hsv: tuple[int, int, int],
    morph_kernel: int,
) -> Optional[np.ndarray]:
    """Mean Lab vector (3,) from non-grass pixels; fallback to full crop."""
    if bgr_crop.size == 0 or bgr_crop.shape[0] < 2 or bgr_crop.shape[1] < 2:
        return None
    hsv = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2HSV)
    grass = cv2.inRange(hsv, np.array(lower_hsv), np.array(upper_hsv))
    mask = cv2.bitwise_not(grass)
    if morph_kernel >= 3 and morph_kernel % 2 == 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_kernel, morph_kernel))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    lab = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2LAB)
    if cv2.countNonZero(mask) < 0.15 * mask.size:
        pixels = lab.reshape(-1, 3).astype(np.float32)
    else:
        pixels = lab[mask > 0].astype(np.float32)
        if pixels.shape[0] < 10:
            pixels = lab.reshape(-1, 3).astype(np.float32)
    return np.mean(pixels, axis=0)


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
        # Retention window in frames. The original spec said "5 frames" but in
        # practice ByteTrack can drop a track for 7-15 frames before assigning
        # a new id (especially under occlusion). We keep records ~1s long so
        # the recovery has a chance; combined with the team+distance match
        # this stays selective enough to avoid wrong reassignments.
        self._lost_track_ttl_frames = 30
        self._lost_track_match_distance_px = 150.0
        # Tight fallback distance: when a new detection is this close to a
        # lost-track record we recover its ID even if the freshly predicted
        # jersey team differs (predicted team can be noisy across frames).
        self._lost_track_close_match_distance_px = 50.0
        # Verbose recovery diagnostics; prints cache state + match decisions.
        self._debug_recovery = True
        # Temporal smoothing for team predictions
        self._history: dict[int, list[int]] = {}
        self._history_len = 21

    def reset(self) -> None:
        self._team_model = None
        self._track_to_team.clear()
        self._track_to_color.clear()
        self._raw_to_effective.clear()
        self._prev_visible_effective_ids.clear()
        self._last_center_by_effective.clear()
        self._lost_track_cache.clear()
        self._history.clear()

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

    def _add_to_history(self, track_id: int, team: int) -> None:
        if track_id not in self._history:
            self._history[track_id] = []
        self._history[track_id].append(team)
        if len(self._history[track_id]) > self._history_len:
            self._history[track_id].pop(0)

    def _get_stable_team(self, track_id: int) -> int:
        hist = self._history.get(track_id, [])
        if not hist:
            return self._track_to_team.get(track_id, 0)
        return Counter(hist).most_common(1)[0][0]

    def _prune_lost_track_cache(self, frame_index: int) -> None:
        """Drop entries older than the retention window.

        ``lost_at_frame_index`` is refreshed when a player flickers in/out; the
        window is measured in **frames since that last refresh**. A small +1
        buffer avoids dropping the record on the frame right after the nominal
        ``_lost_track_ttl_frames`` limit (common when matching runs one frame
        later than expected).
        """
        max_age = self._lost_track_ttl_frames + 1
        self._lost_track_cache = [
            rec
            for rec in self._lost_track_cache
            if (frame_index - rec.lost_at_frame_index) <= max_age
        ]

    def _log_player_lost(
        self, player_id: int, center_xy: tuple[float, float], frame_index: int
    ) -> None:
        cx, cy = center_xy
        print(
            f"Player Lost ID: {player_id} CorX: {cx:.1f} CorY: {cy:.1f} "
            f"frame={frame_index}"
        )

    def _log_player_detected(
        self, center_xy: tuple[float, float], final_id: int, frame_index: int
    ) -> None:
        cx, cy = center_xy
        print(
            f"Player Detected: CorX: {cx:.1f} CorY: {cy:.1f} ID: {final_id} "
            f"frame={frame_index}"
        )

    def _log_recovery_attempt(
        self,
        *,
        frame_index: int,
        raw_id: int,
        center_xy: tuple[float, float],
        predicted_team: int,
        active_effective_ids: set[int],
        consumed_lost_ids: set[int],
    ) -> None:
        """Print full lost-track cache state for a single recovery attempt."""
        cx, cy = center_xy
        print(
            f"[recovery] frame={frame_index} raw={raw_id} center=({cx:.1f},{cy:.1f}) "
            f"predicted_team={predicted_team} cache_size={len(self._lost_track_cache)}"
        )
        if not self._lost_track_cache:
            return
        for rec in self._lost_track_cache:
            rcx, rcy = rec.last_center_xy
            dist = float(np.hypot(cx - rcx, cy - rcy))
            age = frame_index - rec.lost_at_frame_index
            reasons: list[str] = []
            if rec.lost_track_id in active_effective_ids:
                reasons.append("ACTIVE_ID")
            if rec.lost_track_id in consumed_lost_ids:
                reasons.append("CONSUMED")
            if dist > self._lost_track_match_distance_px:
                reasons.append("DIST>MAX")
            same_team = rec.team_id == predicted_team
            tier1 = same_team and dist <= self._lost_track_match_distance_px
            tier2 = (not same_team) and dist <= self._lost_track_close_match_distance_px
            tag = "TIER1" if tier1 else ("TIER2" if tier2 else "REJECT")
            if reasons:
                tag = f"REJECT({','.join(reasons)})"
            print(
                f"[recovery]   rec id={rec.lost_track_id} team={rec.team_id} "
                f"center=({rcx:.1f},{rcy:.1f}) age={age} dist={dist:.1f} -> {tag}"
            )

    def _capture_newly_lost_tracks(self, current_effective_ids: set[int], frame_index: int) -> None:
        """Add or refresh records for players that were visible last frame but are gone now."""
        newly_lost = self._prev_visible_effective_ids - current_effective_ids
        for eff_id in sorted(newly_lost):
            team_id = self._track_to_team.get(eff_id)
            center = self._last_center_by_effective.get(eff_id)
            if team_id is None or center is None:
                continue
            # If a stale record for this effective id exists (e.g. from a prior
            # disappearance), refresh it with the latest center & frame so we
            # match against the most recent known position.
            existing = next(
                (r for r in self._lost_track_cache if r.lost_track_id == eff_id),
                None,
            )
            if existing is not None:
                existing.team_id = team_id
                existing.last_center_xy = center
                existing.lost_at_frame_index = frame_index
            else:
                self._lost_track_cache.append(
                    LostTrackRecord(
                        lost_track_id=eff_id,
                        team_id=team_id,
                        last_center_xy=center,
                        lost_at_frame_index=frame_index,
                    )
                )
            self._log_player_lost(eff_id, center, frame_index)

    def _match_lost_track_id(
        self,
        *,
        team_id: int,
        center_xy: tuple[float, float],
        active_effective_ids: set[int],
        consumed_lost_ids: set[int],
    ) -> Optional[LostTrackRecord]:
        """Return the best matching lost-track record (or ``None``).

        Two-tier matching:
        1. **Same-team within full distance** (``_lost_track_match_distance_px``).
        2. **Close-distance fallback** (``_lost_track_close_match_distance_px``):
           when a candidate is very close to a lost record, recover the ID even
           if the freshly predicted team differs. The record's stored team is
           used downstream so identity stays stable through prediction noise.
        """
        cx, cy = center_xy
        best_team_match: Optional[LostTrackRecord] = None
        best_team_dist = float("inf")
        best_close_match: Optional[LostTrackRecord] = None
        best_close_dist = float("inf")

        for rec in self._lost_track_cache:
            if rec.lost_track_id in active_effective_ids:
                continue
            if rec.lost_track_id in consumed_lost_ids:
                continue
            dx = cx - rec.last_center_xy[0]
            dy = cy - rec.last_center_xy[1]
            dist = float(np.hypot(dx, dy))
            if dist > self._lost_track_match_distance_px:
                continue

            if rec.team_id == team_id and dist < best_team_dist:
                best_team_dist = dist
                best_team_match = rec

            if (
                dist <= self._lost_track_close_match_distance_px
                and dist < best_close_dist
            ):
                best_close_dist = dist
                best_close_match = rec

        if best_team_match is not None:
            return best_team_match
        return best_close_match

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
        """Extract one player's jersey color via torso crop & grass suppressed LAB mean."""
        crop = _upper_torso_crop(
            frame,
            inst.xyxy,
            self._settings.team_torso_y0_ratio,
            self._settings.team_torso_y1_ratio,
        )
        if crop is None:
            return None
        feat = _grass_suppressed_mean_lab(
            crop,
            self._settings.team_grass_hsv_lower,
            self._settings.team_grass_hsv_upper,
            self._settings.team_grass_morph_kernel,
        )
        return feat

    def assign_team_color(self, player_colors: list[np.ndarray]) -> bool:
        """Fit global team-color KMeans from collected player jersey colors."""
        min_samples = max(self._settings.team_kmeans_min_samples, 8)
        if len(player_colors) < min_samples:
            return False
        X = np.stack(player_colors, axis=0).astype(np.float32)
        model = self.get_clustering_model(X, n_clusters=2)
        if model is None:
            return False
            
        # IMPORTANT: Sort clusters by LAB 'A' channel to stabilize team 0 vs team 1
        centers = model.cluster_centers_
        order = np.argsort(centers[:, 1])
        model.cluster_centers_ = centers[order]
        
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

        # --- Step 1: extract jersey colors for visible IDs ---
        frame_colors: dict[int, np.ndarray] = {}
        for inst in players:
            raw_id = inst.track_id
            color = self.get_player_color(frame, inst)
            if color is not None:
                frame_colors[raw_id] = color
                if self._team_model is None:
                    # Accumulate for warm-up
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
                player_color = frame_colors.get(raw_id)
                predicted_team = self._predict_team_label(player_color)

                if eff_id in self._track_to_team:
                    # Already assigned — update tracking & history
                    current_effective_ids.add(eff_id)
                    self._last_center_by_effective[eff_id] = center_xy
                    if predicted_team is not None:
                        self._add_to_history(eff_id, predicted_team)
                        self._track_to_team[eff_id] = self._get_stable_team(eff_id)
                    continue

                # New player (not yet assigned under any effective ID).
                if predicted_team is None:
                    continue

                # Try to recover a lost-track ID from cache
                final_team = predicted_team
                if frame_index > 0:
                    if self._debug_recovery:
                        self._log_recovery_attempt(
                            frame_index=frame_index,
                            raw_id=raw_id,
                            center_xy=center_xy,
                            predicted_team=predicted_team,
                            active_effective_ids=current_effective_ids,
                            consumed_lost_ids=consumed_lost_ids,
                        )
                    matched_rec = self._match_lost_track_id(
                        team_id=predicted_team,
                        center_xy=center_xy,
                        active_effective_ids=current_effective_ids,
                        consumed_lost_ids=consumed_lost_ids,
                    )
                    if matched_rec is not None:
                        eff_id = matched_rec.lost_track_id
                        self._raw_to_effective[raw_id] = eff_id
                        consumed_lost_ids.add(eff_id)
                        if self._debug_recovery:
                            print(
                                f"[recovery] frame={frame_index} raw={raw_id} "
                                f"-> MATCHED lost_id={eff_id} team={matched_rec.team_id}"
                            )
                        # Use the recovered record's team to keep identity stable
                        final_team = matched_rec.team_id
                        self._add_to_history(eff_id, final_team)
                    elif self._debug_recovery:
                        print(
                            f"[recovery] frame={frame_index} raw={raw_id} -> NO MATCH"
                        )

                # Persist team under effective ID only
                if eff_id not in self._track_to_team:
                    self._add_to_history(eff_id, final_team)
                self._track_to_team[eff_id] = self._get_stable_team(eff_id)
                
                current_effective_ids.add(eff_id)
                self._last_center_by_effective[eff_id] = center_xy
                self._log_player_detected(center_xy, eff_id, frame_index)

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
