"""Soccer pitch geometry and keypoint <-> homography helpers.

The 32-point layout mirrors the canonical ``SoccerPitchConfiguration`` from
Roboflow's ``sports`` repository, which is the same layout the
``football-field-detection-f07vi`` keypoint pose model is trained on. All
vertices are expressed in centimeters with the origin at one of the corner
flags, x along the length of the pitch and y along the width.

The package only depends on NumPy and OpenCV; it does not require Ultralytics
or any Roboflow client library, so it is cheap to import.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np


@dataclass(frozen=True)
class SoccerPitchConfiguration:
    """FIFA-style pitch in centimeters with 32 named keypoints.

    The order of :attr:`vertices` matches the keypoint ordering produced by
    the Roboflow ``football-field-detection-f07vi`` model.
    """

    width: int = 6800
    """Pitch width along the goal line (cm)."""

    length: int = 10500
    """Pitch length along the touchline (cm)."""

    penalty_box_width: int = 4032
    penalty_box_length: int = 1650
    goal_box_width: int = 1832
    goal_box_length: int = 550
    centre_circle_radius: int = 915
    penalty_spot_distance: int = 1100

    @property
    def vertices(self) -> np.ndarray:
        """Return the 32 pitch keypoints in pitch coordinates (cm), shape (32, 2)."""
        w = self.width
        L = self.length
        pbw = self.penalty_box_width
        pbl = self.penalty_box_length
        gbw = self.goal_box_width
        gbl = self.goal_box_length
        ccr = self.centre_circle_radius
        psd = self.penalty_spot_distance
        pts = [
            (0, 0),
            (0, (w - pbw) / 2),
            (0, (w - gbw) / 2),
            (0, (w + gbw) / 2),
            (0, (w + pbw) / 2),
            (0, w),
            (gbl, (w - gbw) / 2),
            (gbl, (w + gbw) / 2),
            (psd, w / 2),
            (pbl, (w - pbw) / 2),
            (pbl, (w - gbw) / 2),
            (pbl, (w + gbw) / 2),
            (pbl, (w + pbw) / 2),
            (L / 2, 0),
            (L / 2, w / 2 - ccr),
            (L / 2, w / 2 + ccr),
            (L / 2, w),
            (L - pbl, (w - pbw) / 2),
            (L - pbl, (w - gbw) / 2),
            (L - pbl, (w + gbw) / 2),
            (L - pbl, (w + pbw) / 2),
            (L - psd, w / 2),
            (L - gbl, (w - gbw) / 2),
            (L - gbl, (w + gbw) / 2),
            (L, 0),
            (L, (w - pbw) / 2),
            (L, (w - gbw) / 2),
            (L, (w + gbw) / 2),
            (L, (w + pbw) / 2),
            (L, w),
            (L / 2 - ccr, w / 2),
            (L / 2 + ccr, w / 2),
        ]
        return np.asarray(pts, dtype=np.float32)

    @property
    def corner_indices(self) -> tuple[int, int, int, int]:
        """Indices (TL, BL, BR, TR) of the four corner-flag keypoints.

        Useful to back-project a pitch rectangle into image space for masking.
        """
        return (0, 5, 29, 24)

    @property
    def boundary_polygon(self) -> np.ndarray:
        """Closed pitch outline in pitch coordinates (cm), shape (4, 2).

        Order is TL, BL, BR, TR which traces the touchlines once.
        """
        verts = self.vertices
        idx = self.corner_indices
        return np.asarray([verts[i] for i in idx], dtype=np.float32)


def compute_homography(
    image_xy: np.ndarray,
    confidence: np.ndarray,
    pitch_xy: np.ndarray,
    min_keypoint_conf: float,
    min_visible_points: int = 4,
) -> Optional[np.ndarray]:
    """Solve the image -> pitch homography from visible keypoints.

    Args:
        image_xy: ``(K, 2)`` keypoint pixel coordinates from the model.
        confidence: ``(K,)`` per-keypoint visibility / confidence.
        pitch_xy: ``(K, 2)`` ground-truth pitch coordinates (cm) for those
            keypoints in the same order.
        min_keypoint_conf: keypoints below this score are dropped.
        min_visible_points: at least this many keypoints must remain after
            filtering or ``None`` is returned.

    Returns:
        A ``(3, 3)`` homography matrix mapping image pixels to pitch
        coordinates (cm), or ``None`` if too few keypoints are visible or
        ``cv2.findHomography`` fails to solve.
    """
    if image_xy.shape[0] != confidence.shape[0] or image_xy.shape[0] != pitch_xy.shape[0]:
        raise ValueError("image_xy, confidence, and pitch_xy must have the same length")

    mask = confidence >= float(min_keypoint_conf)
    if int(mask.sum()) < min_visible_points:
        return None

    src = image_xy[mask].astype(np.float32).reshape(-1, 1, 2)
    dst = pitch_xy[mask].astype(np.float32).reshape(-1, 1, 2)
    H, _inliers = cv2.findHomography(src, dst, method=cv2.RANSAC, ransacReprojThreshold=8.0)
    if H is None:
        return None
    return H.astype(np.float64)


def project_points(H: np.ndarray, xy: np.ndarray) -> np.ndarray:
    """Apply a 3x3 homography to an ``(N, 2)`` array of points.

    Returns a new ``(N, 2)`` array.
    """
    if xy.size == 0:
        return xy.astype(np.float32).reshape(-1, 2)
    pts = xy.astype(np.float32).reshape(-1, 1, 2)
    out = cv2.perspectiveTransform(pts, H.astype(np.float64))
    return out.reshape(-1, 2).astype(np.float32)


def back_project_polygon(
    H_image_to_pitch: np.ndarray,
    pitch_polygon: np.ndarray,
) -> Optional[np.ndarray]:
    """Project a polygon defined in pitch coordinates back into image space.

    Inverts ``H_image_to_pitch`` to get the pitch->image transform and then
    runs :func:`project_points`. Returns ``None`` if the matrix is singular.
    """
    try:
        H_inv = np.linalg.inv(H_image_to_pitch.astype(np.float64))
    except np.linalg.LinAlgError:
        return None
    return project_points(H_inv, pitch_polygon)
