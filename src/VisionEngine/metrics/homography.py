"""Map image points to a simplified top-down pitch diagram."""

from __future__ import annotations

from typing import Optional

import numpy as np

from VisionEngine.config.settings import Settings


class FieldMapper:
    """
    Holds a homography from image quad to pitch plane and maps foot/body points
    for a mini-map overlay.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._H: Optional[np.ndarray] = None  # 3x3

    def reset(self) -> None:
        self._H = None

    def configure_from_settings(self) -> None:
        """Build homography from ``Settings.pitch_corners_image`` if set."""
        # TODO: cv2.getPerspectiveTransform(src_quad, dst_rect)
        corners = self._settings.pitch_corners_image
        if corners is None or len(corners) != 4:
            self._H = None
            return

    def image_to_field(self, xy: tuple[float, float]) -> Optional[tuple[float, float]]:
        """Map one image point to normalized field coordinates, if configured."""
        if self._H is None:
            return None
        # TODO: apply perspective transform
        return None
