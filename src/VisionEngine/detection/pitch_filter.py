"""Green-pitch segmentation and geometric filtering for YOLO-style detections.

Uses HSV grass masking, morphology, and largest contour as the playing field.
Designed for real-time use: optional downscale before ``inRange`` / ``findContours``.
"""

from __future__ import annotations

import cv2
import numpy as np

from VisionEngine.schemas.ball_types import RawBallDetection
from VisionEngine.schemas.schema import FrameTracks, TrackedInstance


def extract_pitch_mask(
    frame: np.ndarray,
    *,
    hsv_lower: tuple[int, int, int] = (35, 40, 40),
    hsv_upper: tuple[int, int, int] = (85, 255, 255),
    morph_kernel_size: int = 5,
    process_long_side: int = 640,
) -> tuple[np.ndarray, np.ndarray | None]:
    """
    Segment the pitch via green HSV range, morphology, and largest contour.

    Parameters
    ----------
    frame :
        BGR image (H, W, 3).
    hsv_lower, hsv_upper :
        OpenCV HSV bounds for grass (H 0–179).
    morph_kernel_size :
        Odd kernel size for open+close noise reduction (clamped to odd >= 3).
    process_long_side :
        If ``max(H, W)`` exceeds this, the frame is resized (keeping aspect ratio)
        for mask extraction; the returned mask and contour match **full** frame size.

    Returns
    -------
    mask :
        uint8 single channel, same (H, W) as ``frame``; 255 inside largest contour.
    contour :
        Largest contour in full-frame pixel coordinates, shape ``(N, 1, 2)``
        float32, or ``None`` if no valid region was found.
    """
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("extract_pitch_mask expects a BGR image with shape (H, W, 3).")

    h, w = int(frame.shape[0]), int(frame.shape[1])
    full_mask = np.zeros((h, w), dtype=np.uint8)

    long_side = max(h, w)
    scale = 1.0
    work = frame
    if long_side > int(process_long_side) and process_long_side > 0:
        scale = float(process_long_side) / float(long_side)
        nw = max(1, int(round(w * scale)))
        nh = max(1, int(round(h * scale)))
        work = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)

    hsv = cv2.cvtColor(work, cv2.COLOR_BGR2HSV)
    lower = np.array(hsv_lower, dtype=np.uint8)
    upper = np.array(hsv_upper, dtype=np.uint8)
    mask_small = cv2.inRange(hsv, lower, upper)

    k = max(3, int(morph_kernel_size) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    mask_small = cv2.morphologyEx(mask_small, cv2.MORPH_OPEN, kernel, iterations=1)
    mask_small = cv2.morphologyEx(mask_small, cv2.MORPH_CLOSE, kernel, iterations=1)

    contours, _ = cv2.findContours(
        mask_small, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return full_mask, None

    best = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(best)
    if area < 1.0:
        return full_mask, None

    if scale != 1.0:
        inv = 1.0 / scale
        contour_full = (best.astype(np.float32) * inv).astype(np.float32)
    else:
        contour_full = best.astype(np.float32)

    contour_full = np.ascontiguousarray(contour_full.reshape(-1, 1, 2))
    cv2.drawContours(full_mask, [np.round(contour_full).astype(np.int32)], 0, 255, thickness=-1)
    return full_mask, contour_full


def filter_detections_by_pitch(
    detections: np.ndarray,
    contour: np.ndarray | None,
    *,
    min_area: float,
) -> np.ndarray:
    """
    Keep rows whose bbox center lies inside ``contour`` (``cv2.pointPolygonTest``)
    and whose box area is at least ``min_area`` pixels.

    Parameters
    ----------
    detections :
        Array of shape ``(N, 6)`` with columns
        ``x1, y1, x2, y2, class_id, confidence``.
    contour :
        Pitch contour in pixel space. If ``None``, no polygon test is applied
        (all rows pass the geometry step; callers typically skip filtering entirely).
    min_area :
        Minimum ``(x2-x1)*(y2-y1)`` in pixel space to drop tiny boxes.

    Returns
    -------
    filtered :
        Subset of ``detections`` with shape ``(M, 6)``, same dtype as input.
    """
    if detections.size == 0:
        return detections.reshape(0, 6).astype(np.float64, copy=False)

    d = np.asarray(detections, dtype=np.float64)
    if d.ndim != 2 or d.shape[1] != 6:
        raise ValueError("detections must have shape (N, 6) with x1,y1,x2,y2,class_id,conf.")

    if contour is None:
        return detections

    c = np.ascontiguousarray(contour.astype(np.float32))
    keep_rows: list[int] = []
    for i in range(d.shape[0]):
        x1, y1, x2, y2 = d[i, 0], d[i, 1], d[i, 2], d[i, 3]
        area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        if area < float(min_area):
            continue
        cx = 0.5 * (x1 + x2)
        cy = 0.5 * (y1 + y2)
        if cv2.pointPolygonTest(c, (float(cx), float(cy)), measureDist=False) >= 0:
            keep_rows.append(i)

    return detections[np.array(keep_rows, dtype=np.intp)]


def filter_frame_tracks_by_pitch(
    tracks: FrameTracks,
    contour: np.ndarray | None,
    *,
    min_area: float,
    pass_through_if_no_contour: bool = True,
) -> FrameTracks:
    """Apply :func:`filter_detections_by_pitch` logic to :class:`FrameTracks` instances."""
    if contour is None and pass_through_if_no_contour:
        return tracks

    if contour is None:
        return FrameTracks(
            frame_index=tracks.frame_index,
            timestamp_sec=tracks.timestamp_sec,
            instances=(),
        )

    kept: list[TrackedInstance] = []
    for inst in tracks.instances:
        x1, y1, x2, y2 = inst.xyxy
        area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        if area < float(min_area):
            continue
        cx = 0.5 * (x1 + x2)
        cy = 0.5 * (y1 + y2)
        c = np.ascontiguousarray(contour.astype(np.float32))
        if cv2.pointPolygonTest(c, (float(cx), float(cy)), measureDist=False) < 0:
            continue
        kept.append(inst)

    return FrameTracks(
        frame_index=tracks.frame_index,
        timestamp_sec=tracks.timestamp_sec,
        instances=tuple(kept),
    )


def filter_raw_ball_detections_by_pitch(
    raw: tuple[RawBallDetection, ...],
    contour: np.ndarray | None,
    *,
    min_area: float,
    pass_through_if_no_contour: bool = True,
) -> tuple[RawBallDetection, ...]:
    """Filter low-threshold ball ``predict`` boxes the same way as track boxes."""
    if not raw:
        return raw
    if contour is None and pass_through_if_no_contour:
        return raw
    if contour is None:
        return ()

    c = np.ascontiguousarray(contour.astype(np.float32))
    out: list[RawBallDetection] = []
    for det in raw:
        x1, y1, x2, y2 = det.xyxy
        area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        if area < float(min_area):
            continue
        cx = 0.5 * (x1 + x2)
        cy = 0.5 * (y1 + y2)
        if cv2.pointPolygonTest(c, (float(cx), float(cy)), measureDist=False) >= 0:
            out.append(det)
    return tuple(out)
