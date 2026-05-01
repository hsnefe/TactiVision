"""Drop player / referee detections that fall outside the pitch polygon."""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from VisionEngine.schemas.schema import FrameTracks, ObjectRole, TrackedInstance


class FieldMaskFilter:
    """Cache an image-space pitch polygon and filter ``FrameTracks`` against it.

    BALL detections are always preserved (the ball can legitimately leave the
    pitch on a clearance), only PLAYER and REFEREE roles are filtered out.
    """

    def __init__(self, expand_ratio: float = 0.05) -> None:
        self._expand_ratio = float(expand_ratio)
        self._polygon: Optional[np.ndarray] = None

    @property
    def has_polygon(self) -> bool:
        return self._polygon is not None

    def reset(self) -> None:
        self._polygon = None

    def update_polygon(self, polygon_xy: Optional[np.ndarray]) -> None:
        """Replace the cached polygon (``None`` clears it).

        The polygon is expanded by ``expand_ratio`` of its bbox diagonal,
        so people right on the touchline are not dropped.
        """
        if polygon_xy is None or len(polygon_xy) < 3:
            self._polygon = None
            return
        poly = np.asarray(polygon_xy, dtype=np.float32).reshape(-1, 2)
        if self._expand_ratio > 0.0:
            poly = self._expand_polygon(poly, self._expand_ratio)
        self._polygon = poly

    def is_inside(self, point_xy: tuple[float, float]) -> bool:
        """Point-in-polygon test; returns True when no polygon is cached."""
        if self._polygon is None:
            return True
        x, y = float(point_xy[0]), float(point_xy[1])
        return cv2.pointPolygonTest(self._polygon, (x, y), False) >= 0

    def filter_tracks(self, tracks: FrameTracks) -> FrameTracks:
        """Return a new ``FrameTracks`` with off-pitch player/referee dropped.

        If no polygon is cached, the input is returned unchanged.
        """
        if self._polygon is None or not tracks.instances:
            return tracks
        kept: list[TrackedInstance] = []
        for inst in tracks.instances:
            if inst.role in (ObjectRole.PLAYER, ObjectRole.REFEREE):
                foot_xy = self._foot_point(inst.xyxy)
                if not self.is_inside(foot_xy):
                    continue
            kept.append(inst)
        if len(kept) == len(tracks.instances):
            return tracks
        return FrameTracks(
            frame_index=tracks.frame_index,
            timestamp_sec=tracks.timestamp_sec,
            instances=tuple(kept),
        )

    @property
    def polygon(self) -> Optional[np.ndarray]:
        return self._polygon

    @staticmethod
    def _foot_point(xyxy: tuple[float, float, float, float]) -> tuple[float, float]:
        x1, y1, x2, y2 = xyxy
        return (0.5 * (x1 + x2), float(y2))

    @staticmethod
    def _expand_polygon(poly: np.ndarray, expand_ratio: float) -> np.ndarray:
        center = poly.mean(axis=0)
        bbox_w = float(poly[:, 0].max() - poly[:, 0].min())
        bbox_h = float(poly[:, 1].max() - poly[:, 1].min())
        diag = float(np.hypot(bbox_w, bbox_h))
        if diag <= 1e-3:
            return poly
        offsets = poly - center
        offset_norms = np.linalg.norm(offsets, axis=1, keepdims=True)
        offset_norms = np.where(offset_norms < 1e-3, 1.0, offset_norms)
        directions = offsets / offset_norms
        return (poly + directions * (expand_ratio * diag)).astype(np.float32)
