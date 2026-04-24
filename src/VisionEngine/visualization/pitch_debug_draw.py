"""Debug overlay for grass-pitch filtering (contour + red dropped / green kept boxes)."""

from __future__ import annotations

import cv2
import numpy as np

from VisionEngine.schemas.schema import FrameTracks


def draw_pitch_filter_debug(
    frame: np.ndarray,
    *,
    contour: np.ndarray | None,
    pre_tracks: FrameTracks | None,
    filtered_tracks: FrameTracks,
    pitch_mask: np.ndarray | None = None,
) -> np.ndarray:
    """
    Draw pitch contour (cyan), original boxes (red), kept boxes (green).

    ``pre_tracks`` should be the tracker output before pitch filtering; when
    ``None``, only contour and filtered (green) boxes are drawn.
    """
    out = frame.copy()

    if pitch_mask is not None and pitch_mask.shape[:2] == out.shape[:2]:
        overlay = out.copy()
        green = np.zeros_like(out)
        green[:, :, 1] = pitch_mask
        cv2.addWeighted(overlay, 1.0, green, 0.12, 0, dst=out)

    if contour is not None and len(contour) >= 3:
        cv2.polylines(
            out,
            [np.round(contour).astype(np.int32)],
            isClosed=True,
            color=(255, 255, 0),
            thickness=2,
            lineType=cv2.LINE_AA,
        )

    kept_ids = {
        (round(inst.xyxy[0], 2), round(inst.xyxy[1], 2), round(inst.xyxy[2], 2), round(inst.xyxy[3], 2), inst.track_id)
        for inst in filtered_tracks.instances
    }

    if pre_tracks is not None:
        for inst in pre_tracks.instances:
            key = (
                round(inst.xyxy[0], 2),
                round(inst.xyxy[1], 2),
                round(inst.xyxy[2], 2),
                round(inst.xyxy[3], 2),
                inst.track_id,
            )
            x1, y1, x2, y2 = [int(round(v)) for v in inst.xyxy]
            color = (0, 0, 255) if key not in kept_ids else (0, 220, 0)
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 1, lineType=cv2.LINE_AA)
    else:
        for inst in filtered_tracks.instances:
            x1, y1, x2, y2 = [int(round(v)) for v in inst.xyxy]
            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 220, 0), 2, lineType=cv2.LINE_AA)

    cv2.putText(
        out,
        "pitch filter: red=removed green=kept",
        (16, int(out.shape[0] * 0.92)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return out
