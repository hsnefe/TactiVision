"""Map image points to a simplified top-down pitch diagram."""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from VisionEngine.config.settings import Settings


class FieldMapper:
    """
    Holds a homography from image quad to pitch plane and maps foot/body points
    for a mini-map overlay.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._H: Optional[np.ndarray] = None  # 3x3, image -> pitch

    def reset(self) -> None:
        self._H = None

    @property
    def is_configured(self) -> bool:
        return self._H is not None

    def configure_from_settings(self) -> None:
        """Build a 4-corner homography from ``Settings.pitch_corners_image``.

        Uses the pitch dimensions from
        :class:`~VisionEngine.field.pitch_geometry.SoccerPitchConfiguration`
        as the destination rectangle. Leaves ``self._H`` unchanged when
        keypoint-driven calibration has already been applied via
        :meth:`set_homography` and the legacy corner setting is absent.
        """
        corners = self._settings.pitch_corners_image
        if corners is None or len(corners) != 4:
            return

        from VisionEngine.field.pitch_geometry import SoccerPitchConfiguration

        cfg = SoccerPitchConfiguration()
        src = np.asarray(corners, dtype=np.float32)
        dst = np.asarray(
            [
                (0.0, 0.0),
                (float(cfg.length), 0.0),
                (float(cfg.length), float(cfg.width)),
                (0.0, float(cfg.width)),
            ],
            dtype=np.float32,
        )
        H = cv2.getPerspectiveTransform(src, dst)
        self._H = H.astype(np.float64)

    def set_homography(self, H: Optional[np.ndarray]) -> None:
        """Inject a precomputed image -> pitch homography (e.g. from keypoints)."""
        if H is None:
            self._H = None
            return
        self._H = np.asarray(H, dtype=np.float64).reshape(3, 3)

    def image_to_field(self, xy: tuple[float, float]) -> Optional[tuple[float, float]]:
        """Map one image point to pitch coordinates (cm), if configured."""
        if self._H is None:
            return None
        pt = np.asarray([[[float(xy[0]), float(xy[1])]]], dtype=np.float32)
        out = cv2.perspectiveTransform(pt, self._H)
        return float(out[0, 0, 0]), float(out[0, 0, 1])

    @property
    def homography(self) -> Optional[np.ndarray]:
        return self._H
