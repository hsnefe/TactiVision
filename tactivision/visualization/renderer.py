"""Draw boxes, IDs, team colors, and HUD text — no business logic."""

from __future__ import annotations

from typing import Optional

import cv2

from tactivision.analytics.possession import PossessionState
from tactivision.tracking.schema import FrameTracks
from tactivision.visualization.track_draw import draw_frame_tracks


class AnnotationRenderer:
    """Compose overlays for the output video."""

    def __init__(self) -> None:
        pass

    def render(
        self,
        frame: np.ndarray,
        tracks: FrameTracks,
        track_to_team: dict[int, int],
        possession: PossessionState,
        hud_text: Optional[str] = None,
    ) -> np.ndarray:
        """
        Return a BGR frame with track overlays and optional HUD text.

        Parameters
        ----------
        tracks :
            Per-frame structured tracks (boxes drawn via :func:`draw_frame_tracks`).
        """
        out = draw_frame_tracks(frame, tracks, thickness=2, show_role=True)
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
        if possession.team_id is not None:
            pos_txt = f"possession: team {possession.team_id}"
            cv2.putText(
                out,
                pos_txt,
                (16, 64),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 0),
                2,
                cv2.LINE_AA,
            )
        return out
