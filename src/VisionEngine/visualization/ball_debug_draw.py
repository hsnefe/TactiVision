"""Debug overlays: raw ball (yellow), tracked ball emphasis (red), estimate (blue)."""

from __future__ import annotations

import cv2
import numpy as np

from VisionEngine.schemas.ball_types import RawBallDetection
from VisionEngine.schemas.schema import FrameTracks, ObjectRole

# BGR
YELLOW = (0, 255, 255)
RED = (0, 0, 255)
BLUE = (255, 0, 0)


def draw_raw_ball_detections(
    frame: np.ndarray,
    raw_balls: tuple[RawBallDetection, ...],
    *,
    thickness: int = 2,
) -> np.ndarray:
    """Yellow boxes + confidence text for low-threshold ``predict`` ball pass."""
    out = frame
    for rb in raw_balls:
        x1, y1, x2, y2 = [int(round(v)) for v in rb.xyxy]
        cv2.rectangle(out, (x1, y1), (x2, y2), YELLOW, thickness, cv2.LINE_AA)
        label = f"{rb.confidence:.2f}"
        cv2.putText(
            out,
            label,
            (x1, max(12, y1 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            YELLOW,
            1,
            cv2.LINE_AA,
        )
    return out


def draw_tracked_ball_highlight(
    frame: np.ndarray,
    tracks: FrameTracks,
    *,
    thickness: int = 3,
) -> np.ndarray:
    """
    Emphasize tracked ball boxes in red (may duplicate inner edge of generic draw).
    """
    out = frame
    for inst in tracks.instances:
        if inst.role is not ObjectRole.BALL:
            continue
        x1, y1, x2, y2 = [int(round(v)) for v in inst.xyxy]
        cv2.rectangle(out, (x1, y1), (x2, y2), RED, thickness, cv2.LINE_AA)
        tid = inst.track_id if inst.track_id >= 0 else "?"
        label = f"trk:{tid}"
        cv2.putText(
            out,
            label,
            (x1, y2 + 14),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            RED,
            1,
            cv2.LINE_AA,
        )
    return out


def draw_estimated_ball(
    frame: np.ndarray,
    center_xy: tuple[float, float] | None,
    *,
    radius: int = 10,
) -> np.ndarray:
    """Blue hollow circle for temporally extrapolated ball position."""
    if center_xy is None:
        return frame
    out = frame
    cx, cy = int(round(center_xy[0])), int(round(center_xy[1]))
    cv2.circle(out, (cx, cy), radius, BLUE, 2, cv2.LINE_AA)
    cv2.putText(
        out,
        "est",
        (cx + 12, cy + 4),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        BLUE,
        1,
        cv2.LINE_AA,
    )
    return out


def draw_ball_debug_legend(frame: np.ndarray, x: int = 8, y_start: int = 96) -> np.ndarray:
    """Small legend for debug colors."""
    out = frame
    lines = [
        ("raw predict", YELLOW),
        ("tracked", RED),
        ("estimated", BLUE),
    ]
    y = y_start
    for text, col in lines:
        cv2.putText(out, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)
        y += 18
    return out
