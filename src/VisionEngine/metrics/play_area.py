"""Restrict detections to the visible pitch (field of play) using a polygon mask."""

from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple

import cv2
import numpy as np

from VisionEngine.schemas.ball_types import RawBallDetection
from VisionEngine.schemas.schema import FrameTracks, ObjectRole, TrackedInstance

# Order matches Settings.pitch_corners_image: TL, TR, BR, BL.
Point = Tuple[float, float]


def _denormalize_corners(
    corners_norm: Sequence[Sequence[float]],
    width: int,
    height: int,
) -> list[Point]:
    w, h = float(max(width, 1)), float(max(height, 1))
    out: list[Point] = []
    for c in corners_norm:
        if len(c) < 2:
            continue
        xn, yn = float(c[0]), float(c[1])
        out.append((xn * w, yn * h))
    return out


def _bbox_diag_area(xyxy: tuple[float, float, float, float], w: int, h: int) -> tuple[float, float]:
    x1, y1, x2, y2 = xyxy
    bw = max(0.0, x2 - x1)
    bh = max(0.0, y2 - y1)
    diag = math.hypot(bw, bh)
    area_frac = (bw * bh) / float(max(w * h, 1))
    return diag, area_frac


class PitchPlayArea:
    """
    Keep objects whose geometry lies inside a user-defined pitch quadrilateral.

    Corners are ``(x, y)`` in pixel space, or normalized ``[0, 1]`` if
    ``Settings.play_area_corners_normalized`` is True. Persons (players/referees)
    are kept when the bbox center lies inside the polygon. Ball hypotheses are
    additionally filtered by maximum diagonal and area relative to the frame
    to drop large sideline objects (bottles, mics) that are sometimes misclassified.
    """

    def __init__(
        self,
        *,
        corners: Sequence[Point] | Sequence[Sequence[float]] | None,
        corners_normalized: bool,
        enabled: bool,
        ball_max_diag_frac: float,
        ball_max_area_frac: float,
    ) -> None:
        self._raw_corners = corners
        self._corners_normalized = corners_normalized
        self._enabled = enabled and corners is not None and len(corners) >= 3
        self._ball_max_diag_frac = max(1e-6, float(ball_max_diag_frac))
        self._ball_max_area_frac = max(1e-9, float(ball_max_area_frac))
        self._poly: Optional[np.ndarray] = None  # (K, 1, 2) float32 for OpenCV

    def _ensure_poly(self, width: int, height: int) -> Optional[np.ndarray]:
        if not self._enabled or self._raw_corners is None:
            return None
        if self._poly is not None:
            return self._poly
        pts: list[Point]
        if self._corners_normalized:
            pts = _denormalize_corners(self._raw_corners, width, height)  # type: ignore[arg-type]
        else:
            pts = [(float(p[0]), float(p[1])) for p in self._raw_corners]
        if len(pts) < 3:
            self._enabled = False
            return None
        arr = np.array(pts, dtype=np.float32).reshape(-1, 1, 2)
        self._poly = arr
        return self._poly

    def reset(self) -> None:
        self._poly = None

    def center_inside(self, xyxy: tuple[float, float, float, float], width: int, height: int) -> bool:
        poly = self._ensure_poly(width, height)
        if poly is None:
            return True
        x1, y1, x2, y2 = xyxy
        cx = 0.5 * (x1 + x2)
        cy = 0.5 * (y1 + y2)
        val = cv2.pointPolygonTest(poly, (float(cx), float(cy)), measureDist=False)
        return val >= 0

    def ball_plausible(self, xyxy: tuple[float, float, float, float], width: int, height: int) -> bool:
        if not self.center_inside(xyxy, width, height):
            return False
        diag, area_frac = _bbox_diag_area(xyxy, width, height)
        min_side = float(min(max(width, 1), max(height, 1)))
        if diag > self._ball_max_diag_frac * min_side:
            return False
        if area_frac > self._ball_max_area_frac:
            return False
        return True

    def filter_tracks(self, tracks: FrameTracks, frame_shape: tuple[int, ...]) -> FrameTracks:
        if not self._enabled or len(frame_shape) < 2:
            return tracks
        h, w = int(frame_shape[0]), int(frame_shape[1])
        kept: list[TrackedInstance] = []
        for inst in tracks.instances:
            if inst.role is ObjectRole.BALL:
                if self.ball_plausible(inst.xyxy, w, h):
                    kept.append(inst)
                continue
            if inst.role in (ObjectRole.PLAYER, ObjectRole.REFEREE):
                if self.center_inside(inst.xyxy, w, h):
                    kept.append(inst)
                continue
            kept.append(inst)
        return FrameTracks(
            frame_index=tracks.frame_index,
            timestamp_sec=tracks.timestamp_sec,
            instances=tuple(kept),
        )

    def filter_raw_balls(
        self,
        raw: tuple[RawBallDetection, ...],
        frame_shape: tuple[int, ...],
    ) -> tuple[RawBallDetection, ...]:
        if not self._enabled or len(frame_shape) < 2:
            return raw
        h, w = int(frame_shape[0]), int(frame_shape[1])
        out: list[RawBallDetection] = []
        for det in raw:
            if self.ball_plausible(det.xyxy, w, h):
                out.append(det)
        return tuple(out)
