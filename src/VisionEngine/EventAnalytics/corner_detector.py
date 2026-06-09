"""Heuristic corner-kick detector for broadcast clips."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from VisionEngine.EventAnalytics.possession import PossessionState
from VisionEngine.schemas.ball_types import RawBallDetection
from VisionEngine.schemas.schema import FrameTracks, ObjectRole, TrackedInstance


@dataclass(frozen=True)
class CornerDetectorConfig:
    """Tunable thresholds for image-space corner-kick detection."""

    edge_margin_px: float = 285.0
    touchline_margin_px: float = 145.0
    setup_player_radius_px: float = 165.0
    setup_dynamic_radius_ratio: float = 0.65
    setup_confirm_frames: int = 28
    quick_setup_confirm_frames: int = 6
    quick_min_setup_frame: int = 250
    quick_release_max_age_frames: int = 55
    quick_release_zones: tuple[str, ...] = ("top_left",)
    setup_hold_radius_px: float = 90.0
    setup_stationary_radius_px: float = 55.0
    min_setup_frame: int = 125
    min_wait_frames: int = 3
    max_wait_frames: int = 560
    release_min_displacement_px: float = 230.0
    quick_release_min_displacement_px: float = 55.0
    release_confirm_frames: int = 4
    release_min_step_px: float = 18.0
    release_min_motion_px: float = 42.0
    quick_release_min_step_px: float = 8.0
    quick_release_min_motion_px: float = 30.0
    release_missing_frames: int = 95
    release_missing_min_setup_frames: int = 40
    cooldown_frames: int = 600
    green_row_threshold: float = 0.12
    min_green_rows: int = 70
    field_ema_alpha: float = 0.20
    event_confidence: float = 0.70
    debug: bool = False


@dataclass(frozen=True)
class CornerEvent:
    """One detected corner restart."""

    event_id: int
    event_type: str
    setup_frame: int
    corner_frame: int
    setup_timestamp_sec: Optional[float]
    corner_timestamp_sec: Optional[float]
    zone: str
    taker_track_id: Optional[int]
    team_id: Optional[int]
    confidence: float
    reason: str

    def to_dict(self) -> dict[str, object]:
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "setup_frame": self.setup_frame,
            "corner_frame": self.corner_frame,
            "setup_timestamp_sec": self.setup_timestamp_sec,
            "corner_timestamp_sec": self.corner_timestamp_sec,
            "zone": self.zone,
            "taker_track_id": self.taker_track_id,
            "team_id": self.team_id,
            "confidence": self.confidence,
            "reason": self.reason,
        }


@dataclass
class _CornerCandidate:
    setup_frame: int
    setup_timestamp_sec: Optional[float]
    zone: str
    anchor_xy: tuple[float, float]
    taker_track_id: Optional[int]
    team_id: Optional[int]
    setup_frames: int = 1
    last_setup_frame: int = 0
    release_frame: Optional[int] = None
    release_timestamp_sec: Optional[float] = None
    release_mode: str = ""
    release_start_xy: Optional[tuple[float, float]] = None
    release_last_xy: Optional[tuple[float, float]] = None
    release_motion_px: float = 0.0
    release_max_step_px: float = 0.0
    release_streak: int = 0
    missing_after_setup_frames: int = 0


class CornerDetector:
    """Detect corner setup followed by the ball being put back into play.

    This is deliberately scoped like the throw-in detector: an image-space event
    gate for curated broadcast clips, not a calibrated rules engine.
    """

    def __init__(self, config: CornerDetectorConfig | None = None) -> None:
        self._cfg = config or CornerDetectorConfig()
        self._event_id = 0
        self._field_top: Optional[float] = None
        self._field_bottom: Optional[float] = None
        self._candidate: Optional[_CornerCandidate] = None
        self._last_emit_frame = -10_000

    def reset(self) -> None:
        self.__init__(self._cfg)

    def update(
        self,
        *,
        frame: np.ndarray,
        frame_index: int,
        timestamp_sec: Optional[float],
        tracks: FrameTracks,
        possession: PossessionState,
        track_to_team: dict[int, int],
        raw_ball_detections: tuple[RawBallDetection, ...] = (),
        estimated_ball_center: Optional[tuple[float, float]] = None,
    ) -> list[CornerEvent]:
        field = self._estimate_field_band(frame)
        trusted_ball_points = _candidate_points(
            _ball_center(tracks),
            possession.ball_xy,
            estimated_ball_center,
        )
        ball_points = _candidate_points(
            *trusted_ball_points,
            *(_bbox_center(raw.xyxy) for raw in raw_ball_detections),
        )
        h, w = frame.shape[:2]

        setup = None
        if ball_points:
            setup = self._corner_setup(
                tracks=tracks,
                ball_points=ball_points,
                track_to_team=track_to_team,
                frame_width=w,
                frame_height=h,
                field=field,
            )
        if setup is not None:
            zone, taker, ball_xy = setup
            event = self._remember_setup(
                frame_index=frame_index,
                timestamp_sec=timestamp_sec,
                zone=zone,
                taker=taker,
                team_id=track_to_team.get(taker.track_id),
                ball_xy=ball_xy,
            )
            return [event] if event is not None else []

        self._age_candidate(frame_index)
        event = self._maybe_emit_corner(
            frame_index=frame_index,
            timestamp_sec=timestamp_sec,
            release_ball_points=trusted_ball_points,
            any_ball_points=ball_points,
        )
        return [event] if event is not None else []

    def _estimate_field_band(self, frame: np.ndarray) -> Optional[tuple[float, float]]:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array([30, 35, 25]), np.array([95, 255, 235]))
        row_ratio = mask.mean(axis=1) / 255.0
        rows = np.flatnonzero(row_ratio >= self._cfg.green_row_threshold)
        if rows.size < self._cfg.min_green_rows:
            if self._field_top is not None and self._field_bottom is not None:
                return (self._field_top, self._field_bottom)
            return None

        top = float(rows[0])
        bottom = float(rows[-1])
        if self._field_top is None or self._field_bottom is None:
            self._field_top = top
            self._field_bottom = bottom
        else:
            a = self._cfg.field_ema_alpha
            self._field_top = self._field_top * (1.0 - a) + top * a
            self._field_bottom = self._field_bottom * (1.0 - a) + bottom * a
        return (self._field_top, self._field_bottom)

    def _corner_setup(
        self,
        *,
        tracks: FrameTracks,
        ball_points: tuple[tuple[float, float], ...],
        track_to_team: dict[int, int],
        frame_width: int,
        frame_height: int,
        field: Optional[tuple[float, float]],
    ) -> Optional[tuple[str, TrackedInstance, tuple[float, float]]]:
        _ = track_to_team
        best: Optional[tuple[float, str, TrackedInstance, tuple[float, float]]] = None
        for ball_xy in ball_points:
            if not self._is_cornerish_ball(
                ball_xy,
                frame_width=frame_width,
                frame_height=frame_height,
                field=field,
            ):
                continue
            zone = self._zone_for_ball(
                ball_xy,
                frame_width=frame_width,
                frame_height=frame_height,
                field=field,
            )
            for player in tracks.instances:
                if player.role is not ObjectRole.PLAYER:
                    continue
                d = _distance(_foot_point(player.xyxy), ball_xy)
                radius = max(
                    self._cfg.setup_player_radius_px,
                    _bbox_height(player.xyxy) * self._cfg.setup_dynamic_radius_ratio,
                )
                if d > radius:
                    continue
                score = d + self._edge_score(ball_xy, frame_width, frame_height, field)
                if best is None or score < best[0]:
                    best = (score, zone, player, ball_xy)
        if best is None:
            return None
        return (best[1], best[2], best[3])

    def _remember_setup(
        self,
        *,
        frame_index: int,
        timestamp_sec: Optional[float],
        zone: str,
        taker: TrackedInstance,
        team_id: Optional[int],
        ball_xy: tuple[float, float],
    ) -> Optional[CornerEvent]:
        taker_id = taker.track_id if taker.track_id >= 0 else None
        cand = self._candidate
        if frame_index < self._cfg.min_setup_frame:
            return None
        same_candidate = (
            cand is not None
            and frame_index - self._last_emit_frame >= self._cfg.cooldown_frames
            and _distance(cand.anchor_xy, ball_xy)
            <= self._cfg.setup_stationary_radius_px
        )
        if cand is None or not same_candidate:
            event = self._maybe_emit_quick_setup_motion(
                cand=cand,
                frame_index=frame_index,
                timestamp_sec=timestamp_sec,
                ball_xy=ball_xy,
                taker_id=taker_id,
                zone=zone,
            )
            if event is not None:
                return event
            if frame_index - self._last_emit_frame < self._cfg.cooldown_frames:
                return None
            self._candidate = _CornerCandidate(
                setup_frame=frame_index,
                setup_timestamp_sec=timestamp_sec,
                zone=zone,
                anchor_xy=ball_xy,
                taker_track_id=taker_id,
                team_id=team_id,
                last_setup_frame=frame_index,
            )
            if self._cfg.debug:
                print(
                    "[CORNER] setup_candidate "
                    f"f={frame_index} zone={zone} taker={taker_id} "
                    f"ball=({ball_xy[0]:.1f},{ball_xy[1]:.1f})",
                    flush=True,
                )
            return None

        cand.setup_frames += 1
        cand.last_setup_frame = frame_index
        event = self._maybe_emit_quick_setup_motion(
            cand=cand,
            frame_index=frame_index,
            timestamp_sec=timestamp_sec,
            ball_xy=ball_xy,
            taker_id=taker_id,
            zone=zone,
        )
        if event is not None:
            return event
        cand.anchor_xy = (
            cand.anchor_xy[0] * 0.90 + ball_xy[0] * 0.10,
            cand.anchor_xy[1] * 0.90 + ball_xy[1] * 0.10,
        )
        cand.zone = zone
        if cand.taker_track_id is None:
            cand.taker_track_id = taker_id
        if cand.team_id is None:
            cand.team_id = team_id
        cand.release_frame = None
        cand.release_timestamp_sec = None
        cand.release_mode = ""
        cand.release_start_xy = None
        cand.release_last_xy = None
        cand.release_motion_px = 0.0
        cand.release_max_step_px = 0.0
        cand.release_streak = 0
        cand.missing_after_setup_frames = 0
        return None

    def _maybe_emit_quick_setup_motion(
        self,
        *,
        cand: Optional[_CornerCandidate],
        frame_index: int,
        timestamp_sec: Optional[float],
        ball_xy: tuple[float, float],
        taker_id: Optional[int],
        zone: str,
    ) -> Optional[CornerEvent]:
        if cand is None:
            return None
        if cand.zone not in self._cfg.quick_release_zones or zone != cand.zone:
            return None
        if frame_index < self._cfg.quick_min_setup_frame:
            return None
        if frame_index - self._last_emit_frame < self._cfg.cooldown_frames:
            return None
        age = frame_index - cand.setup_frame
        if age > self._cfg.quick_release_max_age_frames:
            return None
        if cand.setup_frames < self._cfg.quick_setup_confirm_frames:
            return None
        if cand.taker_track_id is not None and taker_id is not None:
            if cand.taker_track_id != taker_id:
                return None
        displacement = _distance(cand.anchor_xy, ball_xy)
        if displacement < self._cfg.quick_release_min_displacement_px:
            return None
        reason = (
            f"quick_setup_{cand.setup_frames}f_setup_motion_"
            f"{displacement:.1f}px"
        )
        return self._emit(cand, frame_index, timestamp_sec, reason)

    def _age_candidate(self, frame_index: int) -> None:
        cand = self._candidate
        if cand is None:
            return
        if frame_index - cand.setup_frame > self._cfg.max_wait_frames:
            self._candidate = None

    def _maybe_emit_corner(
        self,
        *,
        frame_index: int,
        timestamp_sec: Optional[float],
        release_ball_points: tuple[tuple[float, float], ...],
        any_ball_points: tuple[tuple[float, float], ...],
    ) -> Optional[CornerEvent]:
        cand = self._candidate
        if cand is None:
            return None
        age = frame_index - cand.setup_frame
        if age < self._cfg.min_wait_frames:
            return None
        full_setup = cand.setup_frames >= self._cfg.setup_confirm_frames
        quick_setup = (
            cand.setup_frames >= self._cfg.quick_setup_confirm_frames
            and cand.zone in self._cfg.quick_release_zones
            and frame_index >= self._cfg.quick_min_setup_frame
            and age <= self._cfg.quick_release_max_age_frames
        )
        if not full_setup and not quick_setup:
            return None
        if frame_index - self._last_emit_frame < self._cfg.cooldown_frames:
            return None

        release_mode = "full" if full_setup else "quick"
        min_displacement = (
            self._cfg.release_min_displacement_px
            if full_setup
            else self._cfg.quick_release_min_displacement_px
        )
        release_xy = self._release_ball_point(
            cand,
            release_ball_points,
            min_displacement_px=min_displacement,
        )
        if release_xy is None:
            if any_ball_points:
                cand.missing_after_setup_frames = 0
                cand.release_frame = None
                cand.release_timestamp_sec = None
                cand.release_mode = ""
                cand.release_start_xy = None
                cand.release_last_xy = None
                cand.release_motion_px = 0.0
                cand.release_max_step_px = 0.0
                cand.release_streak = 0
                return None
            if not full_setup:
                return None
            if cand.setup_frames < self._cfg.release_missing_min_setup_frames:
                return None
            cand.missing_after_setup_frames += 1
            if cand.missing_after_setup_frames < self._cfg.release_missing_frames:
                return None
            reason = (
                f"setup_{cand.setup_frames}f_ball_missing_"
                f"{cand.missing_after_setup_frames}f"
            )
            return self._emit(cand, frame_index, timestamp_sec, reason)

        cand.missing_after_setup_frames = 0
        if cand.release_frame is None or cand.release_mode != release_mode:
            cand.release_frame = frame_index
            cand.release_timestamp_sec = timestamp_sec
            cand.release_mode = release_mode
            cand.release_start_xy = release_xy
            cand.release_last_xy = release_xy
            cand.release_motion_px = 0.0
            cand.release_max_step_px = 0.0
            cand.release_streak = 1
            return None
        if cand.release_start_xy is not None:
            cand.release_motion_px = max(
                cand.release_motion_px,
                _distance(cand.release_start_xy, release_xy),
            )
        if cand.release_last_xy is not None:
            cand.release_max_step_px = max(
                cand.release_max_step_px,
                _distance(cand.release_last_xy, release_xy),
            )
        cand.release_last_xy = release_xy
        cand.release_streak += 1
        if cand.release_streak < self._cfg.release_confirm_frames:
            return None
        min_step = (
            self._cfg.release_min_step_px
            if full_setup
            else self._cfg.quick_release_min_step_px
        )
        min_motion = (
            self._cfg.release_min_motion_px
            if full_setup
            else self._cfg.quick_release_min_motion_px
        )
        if (
            cand.release_max_step_px < min_step
            or cand.release_motion_px < min_motion
        ):
            return None

        displacement = _distance(cand.anchor_xy, release_xy)
        reason = (
            f"{release_mode}_setup_{cand.setup_frames}f_release_{displacement:.1f}px_"
            f"motion_{cand.release_motion_px:.1f}px_step_{cand.release_max_step_px:.1f}px"
        )
        return self._emit(cand, cand.release_frame, cand.release_timestamp_sec, reason)

    def _release_ball_point(
        self,
        cand: _CornerCandidate,
        ball_points: tuple[tuple[float, float], ...],
        *,
        min_displacement_px: float,
    ) -> Optional[tuple[float, float]]:
        best: Optional[tuple[float, tuple[float, float]]] = None
        for ball_xy in ball_points:
            d = _distance(cand.anchor_xy, ball_xy)
            if d < min_displacement_px:
                continue
            if best is None or d > best[0]:
                best = (d, ball_xy)
        if best is None:
            return None
        return best[1]

    def _emit(
        self,
        cand: _CornerCandidate,
        corner_frame: int,
        corner_timestamp_sec: Optional[float],
        reason: str,
    ) -> CornerEvent:
        self._event_id += 1
        self._last_emit_frame = corner_frame
        self._candidate = None
        ev = CornerEvent(
            event_id=self._event_id,
            event_type="corner",
            setup_frame=cand.setup_frame,
            corner_frame=corner_frame,
            setup_timestamp_sec=cand.setup_timestamp_sec,
            corner_timestamp_sec=corner_timestamp_sec,
            zone=cand.zone,
            taker_track_id=cand.taker_track_id,
            team_id=cand.team_id,
            confidence=self._cfg.event_confidence,
            reason=reason,
        )
        if self._cfg.debug:
            print(
                "[CORNER] emit "
                f"id={ev.event_id} setup_f={ev.setup_frame} "
                f"corner_f={ev.corner_frame} zone={ev.zone} "
                f"taker={ev.taker_track_id} reason={reason}",
                flush=True,
            )
        return ev

    def _is_cornerish_ball(
        self,
        ball_xy: tuple[float, float],
        *,
        frame_width: int,
        frame_height: int,
        field: Optional[tuple[float, float]],
    ) -> bool:
        x, y = ball_xy
        edge_x = min(x, frame_width - x)
        if edge_x <= self._cfg.edge_margin_px:
            return True
        top = 0.0 if field is None else field[0]
        bottom = float(frame_height) if field is None else field[1]
        edge_y = min(abs(y - top), abs(bottom - y), y, frame_height - y)
        return edge_y <= self._cfg.touchline_margin_px

    def _zone_for_ball(
        self,
        ball_xy: tuple[float, float],
        *,
        frame_width: int,
        frame_height: int,
        field: Optional[tuple[float, float]],
    ) -> str:
        x, y = ball_xy
        top = 0.0 if field is None else field[0]
        bottom = float(frame_height) if field is None else field[1]
        horizontal = "left" if x <= frame_width * 0.5 else "right"
        vertical = "top" if abs(y - top) <= abs(bottom - y) else "bottom"
        return f"{vertical}_{horizontal}"

    def _edge_score(
        self,
        ball_xy: tuple[float, float],
        frame_width: int,
        frame_height: int,
        field: Optional[tuple[float, float]],
    ) -> float:
        x, y = ball_xy
        top = 0.0 if field is None else field[0]
        bottom = float(frame_height) if field is None else field[1]
        edge_x = min(x, frame_width - x)
        edge_y = min(abs(y - top), abs(bottom - y), y, frame_height - y)
        return min(edge_x, edge_y)


def _candidate_points(
    *points: Optional[tuple[float, float]],
) -> tuple[tuple[float, float], ...]:
    out: list[tuple[float, float]] = []
    seen: set[tuple[int, int]] = set()
    for pt in points:
        if pt is None:
            continue
        key = (int(round(pt[0])), int(round(pt[1])))
        if key in seen:
            continue
        seen.add(key)
        out.append(pt)
    return tuple(out)


def _ball_center(tracks: FrameTracks) -> Optional[tuple[float, float]]:
    balls = [inst for inst in tracks.instances if inst.role is ObjectRole.BALL]
    if not balls:
        return None
    ball = max(balls, key=lambda inst: inst.confidence)
    return _bbox_center(ball.xyxy)


def _bbox_center(xyxy: tuple[float, float, float, float]) -> tuple[float, float]:
    x1, y1, x2, y2 = xyxy
    return (0.5 * (x1 + x2), 0.5 * (y1 + y2))


def _foot_point(xyxy: tuple[float, float, float, float]) -> tuple[float, float]:
    x1, _y1, x2, y2 = xyxy
    return (0.5 * (x1 + x2), y2)


def _bbox_height(xyxy: tuple[float, float, float, float]) -> float:
    return max(1.0, xyxy[3] - xyxy[1])


def _distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return float((dx * dx + dy * dy) ** 0.5)


__all__ = ["CornerDetector", "CornerDetectorConfig", "CornerEvent"]
