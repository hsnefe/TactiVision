"""Draw boxes, IDs, team colors, and HUD text — no business logic."""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from tactivision.analytics.possession import PossessionState
from tactivision.tracking.tracker import TrackedFrame


class AnnotationRenderer:
    """Compose overlays for the output video."""

    def __init__(self) -> None:
        pass

    def render(
        self,
        frame: np.ndarray,
        tracked: TrackedFrame,
        track_to_team: dict[int, int],
        possession: PossessionState,
        hud_text: Optional[str] = None,
    ) -> np.ndarray:
        """
        Return a BGR frame with annotations.

        Stub: optionally draws ``hud_text`` only; full boxes when data exists.
        """
        out = frame.copy()
        if hud_text:
            cv2.putText(
                out,
                hud_text,
                (16, 32),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
        # TODO: draw xyxy rectangles, track_ids, team colors, mini-field inset
        return out
