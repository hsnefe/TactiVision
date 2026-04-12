"""Draw tracked bounding boxes and IDs on a BGR frame."""

from __future__ import annotations

import cv2
import numpy as np

from tactivision.tracking.schema import FrameTracks, ObjectRole

# BGR colors per role (high contrast on green pitch).
ROLE_COLORS_BGR: dict[ObjectRole, tuple[int, int, int]] = {
    ObjectRole.PLAYER: (0, 255, 0),
    ObjectRole.REFEREE: (0, 165, 255),
    ObjectRole.BALL: (0, 0, 255),
    ObjectRole.OTHER: (200, 200, 200),
}


def draw_frame_tracks(
    frame: np.ndarray,
    frame_tracks: FrameTracks,
    *,
    thickness: int = 2,
    show_role: bool = True,
) -> np.ndarray:
    """
    Draw each instance's box, track id, and optional role label.

    Parameters
    ----------
    frame :
        HxWx3 BGR image (copied before drawing).
    frame_tracks :
        Structured tracks for one frame.
    thickness :
        Rectangle line thickness in pixels.
    show_role :
        If True, label format is ``role:id``; otherwise ``id`` only (or ``?`` if no id).
    """
    out = frame.copy()
    for inst in frame_tracks.instances:
        color = ROLE_COLORS_BGR.get(inst.role, ROLE_COLORS_BGR[ObjectRole.OTHER])
        x1, y1, x2, y2 = [int(round(v)) for v in inst.xyxy]
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness, lineType=cv2.LINE_AA)

        if show_role:
            if inst.track_id >= 0:
                label = f"{inst.role.value}:{inst.track_id}"
            else:
                label = f"{inst.role.value}:?"
        else:
            label = str(inst.track_id) if inst.track_id >= 0 else "?"

        (tw, th), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        ty = max(0, y1 - 4)
        cv2.rectangle(
            out,
            (x1, ty - th - 4),
            (x1 + tw + 2, ty + bl),
            color,
            -1,
        )
        cv2.putText(
            out,
            label,
            (x1 + 1, ty),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0) if sum(color) > 400 else (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return out


__all__ = ["draw_frame_tracks", "ROLE_COLORS_BGR"]
