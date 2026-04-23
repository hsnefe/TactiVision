"""Estimate camera pan proxy from field line orientations."""

from __future__ import annotations

import numpy as np

from VisionEngine.config.settings import Settings


class CameraPanEstimator:
    """
    Uses line segments (e.g. Canny + Hough) on field / edge regions to estimate
    a scalar pan change vs horizontal or vs the previous frame.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._last_pan_rad: float = 0.0

    def reset(self) -> None:
        self._last_pan_rad = 0.0

    def update(self, frame: np.ndarray) -> float:
        """
        Return an estimated pan angle in radians (stub returns 0.0).

        Positive/negative convention to be defined when implemented.
        """
        if not self._settings.enable_camera_pan:
            return 0.0
        return 0.0
