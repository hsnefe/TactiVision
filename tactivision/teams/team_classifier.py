"""
Jersey-based two-team classification for tracked players.

Pipeline (per frame):

1. Keep only :class:`~tactivision.tracking.schema.ObjectRole` ``PLAYER`` with
   ``track_id >= 0``. Referees and ball are ignored (referees need a referee
   class in the detector; COCO ``person`` cannot be separated without a custom
   model).

2. Crop the **upper torso** band inside each player box (configurable vertical
   slice) to reduce shorts/socks/grass at the feet.

3. Suppress typical **pitch green** in HSV; if too few non-grass pixels remain,
   fall back to the full crop.

4. Compute mean **Lab** color (perceptual; more stable than RGB under lighting).

5. Run **KMeans(k=2)** on all player features in the frame when enough samples
   exist; otherwise label by distance to **EMA-smoothed** team Lab centroids
   from earlier frames.

6. Map sklearn cluster id → team ``0`` / ``1`` by sorting cluster centers by
   Lab ``a`` (green–red axis), so the same physical kit stays the same team id
   across frames even if sklearn permutes cluster labels.

7. **Temporal stabilization**: per ``track_id``, keep a deque of recent raw
   team ids and output the **majority vote**.

See :class:`~tactivision.config.settings.Settings` for tunables.
"""

from __future__ import annotations

from collections import Counter, deque
from typing import Optional

import cv2
import numpy as np
from sklearn.cluster import KMeans

from tactivision.config.settings import Settings
from tactivision.tracking.schema import FrameTracks, ObjectRole, TrackedInstance


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
        # Too much green removed — use full patch
        pixels = lab.reshape(-1, 3).astype(np.float32)
    else:
        pixels = lab[mask > 0].astype(np.float32)
        if pixels.shape[0] < 10:
            pixels = lab.reshape(-1, 3).astype(np.float32)
    return np.mean(pixels, axis=0)


class TeamClassifier:
    """
    Assign team ``0`` or ``1`` to each player track using jersey color clustering.

    Parameters are read from :class:`~tactivision.config.settings.Settings`.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._ema_center0: Optional[np.ndarray] = None  # Lab (3,)
        self._ema_center1: Optional[np.ndarray] = None
        self._history: dict[int, deque[int]] = {}

    def reset(self) -> None:
        self._ema_center0 = self._ema_center1 = None
        self._history.clear()

    def _player_instances(self, tracks: FrameTracks) -> list[TrackedInstance]:
        out: list[TrackedInstance] = []
        for inst in tracks.instances:
            if inst.role is not ObjectRole.PLAYER:
                continue
            if inst.track_id < 0:
                continue
            out.append(inst)
        return out

    def _feature_for_player(self, frame: np.ndarray, inst: TrackedInstance) -> Optional[np.ndarray]:
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

    def _assign_teams_raw(
        self,
        track_ids: list[int],
        X: np.ndarray,
    ) -> dict[int, int]:
        """Map each track_id to raw team 0/1 for this frame."""
        n = X.shape[0]
        if n == 0:
            return {}

        min_s = self._settings.team_kmeans_min_samples
        alpha = self._settings.team_center_ema_alpha

        # Single player: nearest EMA center only
        if n == 1:
            if self._ema_center0 is None or self._ema_center1 is None:
                return {}
            tid = track_ids[0]
            d0 = float(np.sum((X[0] - self._ema_center0) ** 2))
            d1 = float(np.sum((X[0] - self._ema_center1) ** 2))
            return {tid: (0 if d0 < d1 else 1)}

        # Enough samples for a fresh KMeans
        if n >= min_s:
            km = KMeans(n_clusters=2, n_init=10, random_state=42).fit(X)
            centers = km.cluster_centers_
            labels = km.labels_
            # Stable team id: sort cluster indices by Lab 'a' (column 1)
            order = np.argsort(centers[:, 1])
            cluster_to_team = {int(order[0]): 0, int(order[1]): 1}
            raw = {track_ids[i]: cluster_to_team[int(labels[i])] for i in range(n)}

            sorted_centers = centers[order]
            if self._ema_center0 is None:
                self._ema_center0 = sorted_centers[0].copy()
                self._ema_center1 = sorted_centers[1].copy()
            else:
                self._ema_center0 = (1 - alpha) * self._ema_center0 + alpha * sorted_centers[0]
                self._ema_center1 = (1 - alpha) * self._ema_center1 + alpha * sorted_centers[1]
            return raw

        # Few players: assign by distance to EMA centers if available
        if self._ema_center0 is not None and self._ema_center1 is not None:
            out: dict[int, int] = {}
            c0, c1 = self._ema_center0, self._ema_center1
            for i, tid in enumerate(track_ids):
                d0 = float(np.sum((X[i] - c0) ** 2))
                d1 = float(np.sum((X[i] - c1) ** 2))
                out[tid] = 0 if d0 < d1 else 1
            return out

        # Cold start with 2–3 players: still run KMeans if possible
        if n >= 2:
            km = KMeans(n_clusters=2, n_init=10, random_state=42).fit(X)
            centers = km.cluster_centers_
            labels = km.labels_
            order = np.argsort(centers[:, 1])
            cluster_to_team = {int(order[0]): 0, int(order[1]): 1}
            raw = {track_ids[i]: cluster_to_team[int(labels[i])] for i in range(n)}
            sorted_centers = centers[order]
            if self._ema_center0 is None:
                self._ema_center0 = sorted_centers[0].copy()
                self._ema_center1 = sorted_centers[1].copy()
            else:
                self._ema_center0 = (1 - alpha) * self._ema_center0 + alpha * sorted_centers[0]
                self._ema_center1 = (1 - alpha) * self._ema_center1 + alpha * sorted_centers[1]
            return raw

        return {}

    def _smooth(self, track_id: int, team: int) -> int:
        w = self._settings.team_history_frames
        if track_id not in self._history:
            self._history[track_id] = deque(maxlen=w)
        self._history[track_id].append(team)
        hist = self._history[track_id]
        return Counter(hist).most_common(1)[0][0]

    def update(self, frame: np.ndarray, tracks: FrameTracks) -> dict[int, int]:
        """
        Return stabilized ``track_id -> team_id`` (0 or 1) for visible players.

        Players without enough signal keep prior smoothed label if present; new
        tracks may be omitted until features stabilize.
        """
        if not self._settings.team_classification_enabled:
            return {}

        players = self._player_instances(tracks)
        track_ids: list[int] = []
        feats: list[np.ndarray] = []
        for inst in players:
            f = self._feature_for_player(frame, inst)
            if f is None:
                continue
            track_ids.append(inst.track_id)
            feats.append(f)

        if not feats:
            return {
                tid: Counter(h).most_common(1)[0][0]
                for tid, h in self._history.items()
                if len(h) > 0
            }

        X = np.stack(feats, axis=0)
        raw_map = self._assign_teams_raw(track_ids, X)

        result: dict[int, int] = {}
        for tid in track_ids:
            if tid not in raw_map:
                continue
            team = self._smooth(tid, raw_map[tid])
            result[tid] = team

        # Carry forward tracks that were seen before but missing this frame (hold last vote)
        for tid, hist in self._history.items():
            if tid not in result and len(hist) > 0:
                result[tid] = Counter(hist).most_common(1)[0][0]

        return result
