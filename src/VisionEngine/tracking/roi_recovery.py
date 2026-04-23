"""ROI-based aggressive detection helpers for recovering missed player tracks."""

from __future__ import annotations

import math


def bbox_iou(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> float:
    """Intersection-over-union for axis-aligned boxes."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = ix2 - ix1
    ih = iy2 - iy1
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    aa = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    ba = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = aa + ba - inter
    if denom <= 0:
        return 0.0
    return float(inter / denom)


def expand_xyxy(
    xyxy: tuple[float, float, float, float],
    margin_ratio: float,
    frame_w: int,
    frame_h: int,
) -> tuple[int, int, int, int]:
    """Expand bbox by margin_ratio * per-axis size; clip to frame bounds."""
    x1, y1, x2, y2 = xyxy
    bw = max(1e-3, x2 - x1)
    bh = max(1e-3, y2 - y1)
    mx = bw * margin_ratio
    my = bh * margin_ratio
    nx1 = int(math.floor(x1 - mx))
    ny1 = int(math.floor(y1 - my))
    nx2 = int(math.ceil(x2 + mx))
    ny2 = int(math.ceil(y2 + my))
    nx1 = max(0, min(nx1, frame_w - 1))
    ny1 = max(0, min(ny1, frame_h - 1))
    nx2 = max(nx1 + 1, min(nx2, frame_w))
    ny2 = max(ny1 + 1, min(ny2, frame_h))
    return nx1, ny1, nx2, ny2


def is_near_frame_border(
    xyxy: tuple[float, float, float, float],
    frame_w: int,
    frame_h: int,
    margin_px: int,
) -> bool:
    x1, y1, x2, y2 = xyxy
    m = float(max(0, margin_px))
    return (
        x1 <= m
        or y1 <= m
        or x2 >= float(frame_w - 1) - m
        or y2 >= float(frame_h - 1) - m
    )


def pick_best_by_iou(
    boxes: list[tuple[float, float, float, float]],
    reference: tuple[float, float, float, float],
) -> int | None:
    """Return index of box with highest IoU to ``reference``, or None if empty."""
    if not boxes:
        return None
    best_i = 0
    best_v = bbox_iou(boxes[0], reference)
    for i in range(1, len(boxes)):
        v = bbox_iou(boxes[i], reference)
        if v > best_v:
            best_v = v
            best_i = i
    return best_i


def center_distance(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> float:
    """Euclidean distance between bbox centers in pixels."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    acx = 0.5 * (ax1 + ax2)
    acy = 0.5 * (ay1 + ay2)
    bcx = 0.5 * (bx1 + bx2)
    bcy = 0.5 * (by1 + by2)
    return float(math.hypot(acx - bcx, acy - bcy))
