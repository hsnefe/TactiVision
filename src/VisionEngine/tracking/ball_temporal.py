"""Short-gap position hold / linear extrapolation when tracked ball drops out."""

from __future__ import annotations

from typing import Optional

from VisionEngine.schemas.schema import TrackedInstance


class BallTemporalBridge:
    """
    When the tracked ball is missing for a few frames, extrapolate from the last
    known center and per-frame velocity so downstream logic can still use a soft
    prior (and debug overlay shows a blue estimate).
    """

    def __init__(self, max_gap_frames: int) -> None:
        self._max_gap = max(1, max_gap_frames)
        self._last_cx: Optional[float] = None
        self._last_cy: Optional[float] = None
        self._vx: float = 0.0
        self._vy: float = 0.0
        self._last_seen_frame: int = -1

    def reset(self) -> None:
        self._last_cx = self._last_cy = None
        self._vx = self._vy = 0.0
        self._last_seen_frame = -1

    def update(
        self,
        frame_index: int,
        tracked_balls: tuple[TrackedInstance, ...],
    ) -> tuple[Optional[tuple[float, float]], bool]:
        """
        Returns ``(center_xy, is_estimated)``.

        When at least one tracked ball exists, updates state and returns
        ``(None, False)`` (no estimate overlay — real detection handles display).

        When none exist but gap ≤ max_gap, returns extrapolated center and
        ``is_estimated=True``.
        """
        if tracked_balls:
            b = max(tracked_balls, key=lambda x: x.confidence)
            cx = 0.5 * (b.xyxy[0] + b.xyxy[2])
            cy = 0.5 * (b.xyxy[1] + b.xyxy[3])

            if self._last_seen_frame >= 0 and self._last_cx is not None and self._last_cy is not None:
                df = frame_index - self._last_seen_frame
                if df > 0:
                    self._vx = (cx - self._last_cx) / df
                    self._vy = (cy - self._last_cy) / df

            self._last_cx, self._last_cy = cx, cy
            self._last_seen_frame = frame_index
            return None, False

        if self._last_cx is None or self._last_seen_frame < 0:
            return None, False

        gap = frame_index - self._last_seen_frame
        if gap <= 0 or gap > self._max_gap:
            return None, False

        ex = self._last_cx + self._vx * gap
        ey = self._last_cy + self._vy * gap
        return (ex, ey), True
