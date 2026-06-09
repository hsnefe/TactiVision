"""Heuristic throw-in detector for broadcast clips."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from VisionEngine.EventAnalytics.possession import PossessionState
from VisionEngine.schemas.schema import FrameTracks, ObjectRole, TrackedInstance


@dataclass(frozen=True)
class ThrowInDetectorConfig:
    """Tunable thresholds for image-space throw-in detection."""

    boundary_margin_px: float = 55.0
    out_open_margin_px: float = 10.0
    player_boundary_margin_px: float = 95.0
    out_confirm_frames: int = 2
    min_wait_frames: int = 20
    max_wait_frames: int = 360
    cooldown_frames: int = 70
    min_inward_ball_px: float = 38.0
    max_release_inward_px: float = 330.0
    restart_confirm_frames: int = 3
    setup_min_age_frames: int = 55
    setup_confirm_frames: int = 3
    setup_min_hold_frames: int = 70
    visible_out_setup_min_hold_frames: int = 8
    side_override_setup_min_hold_frames: int = 50
    release_after_setup_gap_frames: int = 40
    side_override_min_age_frames: int = 20
    side_override_confirm_frames: int = 3
    setup_anchor_x_margin_px: float = 260.0
    release_min_displacement_px: float = 55.0
    tracked_release_min_displacement_px: float = 45.0
    tracked_release_player_boundary_margin_px: float = 160.0
    tracked_release_player_radius_px: float = 155.0
    tracked_release_ball_missing_frames: int = 6
    tracked_release_enabled: bool = False
    throw_pose_release_enabled: bool = True
    throw_pose_min_ball_missing_frames: int = 12
    throw_pose_max_ball_missing_frames: int = 50
    throw_pose_player_boundary_margin_px: float = 180.0
    throw_pose_player_radius_px: float = 110.0
    throw_pose_x_margin_ratio: float = 0.95
    throw_pose_y_above_ratio: float = 1.05
    throw_pose_y_below_ratio: float = 0.0
    visual_holder_suppression_player_radius_px: float = 120.0
    restart_possession_radius_px: float = 115.0
    ball_missing_open_frames: int = 10
    green_row_threshold: float = 0.12
    min_green_rows: int = 80
    field_ema_alpha: float = 0.20
    event_confidence: float = 0.72
    debug: bool = False


@dataclass(frozen=True)
class ThrowInEvent:
    """One detected throw-in restart."""

    event_id: int
    event_type: str
    out_frame: int
    throw_frame: int
    out_timestamp_sec: Optional[float]
    throw_timestamp_sec: Optional[float]
    side: str
    thrower_track_id: Optional[int]
    team_id: Optional[int]
    confidence: float
    reason: str

    def to_dict(self) -> dict[str, object]:
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "out_frame": self.out_frame,
            "throw_frame": self.throw_frame,
            "out_timestamp_sec": self.out_timestamp_sec,
            "throw_timestamp_sec": self.throw_timestamp_sec,
            "side": self.side,
            "thrower_track_id": self.thrower_track_id,
            "team_id": self.team_id,
            "confidence": self.confidence,
            "reason": self.reason,
        }


@dataclass
class _OutCandidate:
    frame_index: int
    timestamp_sec: Optional[float]
    side: str
    ball_xy: tuple[float, float]
    field_top: float
    field_bottom: float
    kind: str = "visible_out"
    setup_track_id: Optional[int] = None
    setup_team_id: Optional[int] = None
    setup_start_frame: Optional[int] = None
    setup_frame: Optional[int] = None
    setup_timestamp_sec: Optional[float] = None
    setup_ball_xy: Optional[tuple[float, float]] = None
    setup_frames: int = 0
    release_frame: Optional[int] = None
    release_timestamp_sec: Optional[float] = None
    release_thrower_track_id: Optional[int] = None
    release_team_id: Optional[int] = None
    inward_streak: int = 0
    side_override_side: Optional[str] = None
    side_override_track_id: Optional[int] = None
    side_override_start_frame: Optional[int] = None
    side_override_frames: int = 0
    side_override_ball_xy: Optional[tuple[float, float]] = None


class ThrowInDetector:
    """Detect OUT -> touchline restart sequences using image-space cues.

    This is intentionally separate from pass detection. It is meant as a
    conservative event gate for throw-in clips, not as a pitch-calibrated rules
    engine.
    """

    def __init__(self, config: ThrowInDetectorConfig | None = None) -> None:
        self._cfg = config or ThrowInDetectorConfig()
        self._event_id = 0
        self._field_top: Optional[float] = None
        self._field_bottom: Optional[float] = None
        self._out_candidate: Optional[_OutCandidate] = None
        self._near_boundary_streak = 0
        self._near_boundary_side: Optional[str] = None
        self._ball_missing_streak = 0
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
        estimated_ball_center: Optional[tuple[float, float]] = None,
    ) -> list[ThrowInEvent]:
        field = self._estimate_field_band(frame)
        tracked_ball_xy = _ball_center(tracks) or possession.ball_xy or estimated_ball_center
        visual_ball_candidates = _bright_ball_candidates(frame)
        ball_xy = tracked_ball_xy
        if field is None or ball_xy is None:
            self._ball_missing_streak += 1
            self._age_candidate(frame_index)
            return []
        had_ball_missing_streak = self._ball_missing_streak
        self._ball_missing_streak = 0

        field_top, field_bottom = field
        side = self._boundary_side(
            ball_xy,
            field_top,
            field_bottom,
            margin_px=self._cfg.boundary_margin_px,
        )
        if side is None:
            self._near_boundary_streak = 0
            self._near_boundary_side = None
        elif side == self._near_boundary_side:
            self._near_boundary_streak += 1
        else:
            self._near_boundary_side = side
            self._near_boundary_streak = 1

        open_side = self._open_candidate_side(
            ball_xy,
            field_top,
            field_bottom,
            had_ball_missing_streak=had_ball_missing_streak,
        )
        if self._can_open_out_candidate(frame_index, open_side):
            self._out_candidate = _OutCandidate(
                frame_index=frame_index,
                timestamp_sec=timestamp_sec,
                side=str(open_side),
                ball_xy=ball_xy,
                field_top=field_top,
                field_bottom=field_bottom,
                kind=(
                    "lost_ball_touchline_reentry"
                    if had_ball_missing_streak >= self._cfg.ball_missing_open_frames
                    else "visible_out"
                ),
            )
            if self._cfg.debug:
                print(
                    "[THROWIN] out_candidate "
                    f"f={frame_index} side={open_side} kind={self._out_candidate.kind} "
                f"ball=({ball_xy[0]:.1f},{ball_xy[1]:.1f})",
                    flush=True,
                )

        throw_pose_release = None
        if (
            self._cfg.throw_pose_release_enabled
            and self._out_candidate is None
            and tracked_ball_xy is not None
            and self._cfg.throw_pose_min_ball_missing_frames
            <= had_ball_missing_streak
            <= self._cfg.throw_pose_max_ball_missing_frames
        ):
            throw_pose_release = self._touchline_throw_pose_release_player(
                tracks=tracks,
                ball_xy=tracked_ball_xy,
                field_top=field_top,
                field_bottom=field_bottom,
            )
        if (
            throw_pose_release is not None
            and self._can_open_out_candidate(
                frame_index,
                throw_pose_release[0],
                require_boundary_streak=False,
            )
        ):
            release_side, release_player = throw_pose_release
            self._out_candidate = _OutCandidate(
                frame_index=frame_index,
                timestamp_sec=timestamp_sec,
                side=release_side,
                ball_xy=tracked_ball_xy,
                field_top=field_top,
                field_bottom=field_bottom,
                kind="touchline_throw_pose_release",
            )
            self._remember_holder_setup(
                self._out_candidate,
                frame_index=frame_index,
                timestamp_sec=timestamp_sec,
                ball_xy=tracked_ball_xy,
                holder=release_player,
                track_to_team=track_to_team,
            )
            self._out_candidate.setup_frames = max(
                self._out_candidate.setup_frames,
                self._cfg.setup_confirm_frames,
            )
            if self._out_candidate.setup_start_frame is not None:
                self._out_candidate.setup_start_frame = (
                    frame_index - self._out_candidate.setup_frames + 1
                )
            if self._cfg.debug:
                print(
                    "[THROWIN] throw_pose_candidate "
                    f"f={frame_index} side={release_side} "
                    f"holder={release_player.track_id}",
                    flush=True,
                )

        suppress_visual_holder = (
            tracked_ball_xy is not None
            and self._tracked_ball_near_player(
                tracks=tracks,
                ball_xy=tracked_ball_xy,
                radius_px=self._cfg.visual_holder_suppression_player_radius_px,
            )
        )
        holder_points = (
            (tracked_ball_xy,)
            if suppress_visual_holder
            else _candidate_points(tracked_ball_xy, visual_ball_candidates)
        )
        holder = self._touchline_holder(
            tracks=tracks,
            ball_candidates=holder_points,
            field_top=field_top,
            field_bottom=field_bottom,
            track_to_team=track_to_team,
        )
        if (
            self._out_candidate is None
            and holder is not None
            and had_ball_missing_streak >= self._cfg.ball_missing_open_frames
            and self._can_open_out_candidate(
                frame_index,
                holder[0],
                require_boundary_streak=False,
            )
        ):
            holder_side, holder_player, holder_ball_xy = holder
            self._out_candidate = _OutCandidate(
                frame_index=frame_index,
                timestamp_sec=timestamp_sec,
                side=holder_side,
                ball_xy=holder_ball_xy,
                field_top=field_top,
                field_bottom=field_bottom,
                kind="touchline_holder_setup",
            )
            self._remember_holder_setup(
                self._out_candidate,
                frame_index=frame_index,
                timestamp_sec=timestamp_sec,
                ball_xy=holder_ball_xy,
                holder=holder_player,
                track_to_team=track_to_team,
            )
            if self._cfg.debug:
                print(
                    "[THROWIN] holder_candidate "
                    f"f={frame_index} side={holder_side} holder={holder_player.track_id}",
                    flush=True,
                )

        tracked_release = None
        if (
            self._cfg.tracked_release_enabled
            and self._out_candidate is None
            and tracked_ball_xy is not None
        ):
            tracked_release = self._touchline_release_player(
                tracks=tracks,
                ball_xy=tracked_ball_xy,
                field_top=field_top,
                field_bottom=field_bottom,
            )
        if (
            tracked_release is not None
            and had_ball_missing_streak >= self._cfg.tracked_release_ball_missing_frames
            and self._can_open_out_candidate(
                frame_index,
                tracked_release[0],
                require_boundary_streak=False,
            )
        ):
            release_side, release_player = tracked_release
            self._out_candidate = _OutCandidate(
                frame_index=frame_index,
                timestamp_sec=timestamp_sec,
                side=release_side,
                ball_xy=tracked_ball_xy,
                field_top=field_top,
                field_bottom=field_bottom,
                kind="touchline_tracked_release",
            )
            self._remember_holder_setup(
                self._out_candidate,
                frame_index=frame_index,
                timestamp_sec=timestamp_sec,
                ball_xy=tracked_ball_xy,
                holder=release_player,
                track_to_team=track_to_team,
            )
            if self._cfg.debug:
                print(
                    "[THROWIN] tracked_release_candidate "
                    f"f={frame_index} side={release_side} holder={release_player.track_id}",
                    flush=True,
                )

        event = self._maybe_emit_throwin(
            frame_index=frame_index,
            timestamp_sec=timestamp_sec,
            ball_xy=ball_xy,
            ball_candidates=_candidate_points(tracked_ball_xy, visual_ball_candidates),
            possession=possession,
            tracks=tracks,
            track_to_team=track_to_team,
            field_top=field_top,
            field_bottom=field_bottom,
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

    def _boundary_side(
        self,
        ball_xy: tuple[float, float],
        field_top: float,
        field_bottom: float,
        *,
        margin_px: float,
    ) -> Optional[str]:
        y = ball_xy[1]
        if y <= field_top + margin_px:
            return "top"
        if y >= field_bottom - margin_px:
            return "bottom"
        return None

    def _open_candidate_side(
        self,
        ball_xy: tuple[float, float],
        field_top: float,
        field_bottom: float,
        *,
        had_ball_missing_streak: int,
    ) -> Optional[str]:
        tight = self._boundary_side(
            ball_xy,
            field_top,
            field_bottom,
            margin_px=self._cfg.out_open_margin_px,
        )
        if tight is not None:
            return tight
        if had_ball_missing_streak >= self._cfg.ball_missing_open_frames:
            return self._boundary_side(
                ball_xy,
                field_top,
                field_bottom,
                margin_px=self._cfg.boundary_margin_px,
            )
        return None

    def _can_open_out_candidate(
        self,
        frame_index: int,
        side: Optional[str],
        *,
        require_boundary_streak: bool = True,
    ) -> bool:
        if side is None:
            return False
        if frame_index - self._last_emit_frame < self._cfg.cooldown_frames:
            return False
        if (
            require_boundary_streak
            and self._near_boundary_streak < self._cfg.out_confirm_frames
        ):
            return False
        if self._out_candidate is not None:
            age = frame_index - self._out_candidate.frame_index
            if age <= self._cfg.max_wait_frames:
                return False
        return True

    def _age_candidate(self, frame_index: int) -> None:
        if self._out_candidate is None:
            return
        if frame_index - self._out_candidate.frame_index > self._cfg.max_wait_frames:
            self._out_candidate = None

    def _maybe_emit_throwin(
        self,
        *,
        frame_index: int,
        timestamp_sec: Optional[float],
        ball_xy: tuple[float, float],
        ball_candidates: tuple[tuple[float, float], ...],
        possession: PossessionState,
        tracks: FrameTracks,
        track_to_team: dict[int, int],
        field_top: float,
        field_bottom: float,
    ) -> Optional[ThrowInEvent]:
        cand = self._out_candidate
        if cand is None:
            return None
        age = frame_index - cand.frame_index
        min_wait = (
            1
            if cand.kind in {"touchline_tracked_release", "touchline_throw_pose_release"}
            else self._cfg.min_wait_frames
        )
        if age < min_wait:
            return None
        if age > self._cfg.max_wait_frames:
            self._out_candidate = None
            return None

        if self._maybe_override_candidate_side(
            cand,
            frame_index=frame_index,
            timestamp_sec=timestamp_sec,
            ball_candidates=ball_candidates,
            tracks=tracks,
            track_to_team=track_to_team,
            field_top=field_top,
            field_bottom=field_bottom,
            age=age,
        ):
            return None

        holder = None
        if age >= self._cfg.setup_min_age_frames:
            holder = self._touchline_holder(
                tracks=tracks,
                ball_candidates=ball_candidates,
                field_top=field_top,
                field_bottom=field_bottom,
                track_to_team=track_to_team,
                side_filter=cand.side,
                anchor_x=cand.ball_xy[0],
            )
        if holder is not None:
            self._remember_holder_setup(
                cand,
                frame_index=frame_index,
                timestamp_sec=timestamp_sec,
                ball_xy=holder[2],
                holder=holder[1],
                track_to_team=track_to_team,
            )
            cand.inward_streak = 0
            cand.release_frame = None
            cand.release_timestamp_sec = None
            cand.release_thrower_track_id = None
            cand.release_team_id = None
            return None

        if cand.setup_frames < self._cfg.setup_confirm_frames:
            return None
        if cand.setup_start_frame is not None:
            setup_age = frame_index - cand.setup_start_frame
            min_setup_hold = (
                self._cfg.side_override_setup_min_hold_frames
                if "_side_override" in cand.kind
                else self._cfg.visible_out_setup_min_hold_frames
                if cand.kind == "visible_out"
                else 0
                if cand.kind
                in {"touchline_tracked_release", "touchline_throw_pose_release"}
                else self._cfg.setup_min_hold_frames
            )
            if setup_age < min_setup_hold:
                return None
        if cand.kind == "visible_out" and cand.setup_frame is not None:
            release_gap = frame_index - cand.setup_frame
            if release_gap < self._cfg.release_after_setup_gap_frames:
                cand.inward_streak = 0
                cand.release_frame = None
                cand.release_timestamp_sec = None
                cand.release_thrower_track_id = None
                cand.release_team_id = None
                return None

        release = self._release_ball_candidate(cand, ball_xy, ball_candidates)
        if release is None:
            cand.inward_streak = 0
            cand.release_frame = None
            cand.release_timestamp_sec = None
            cand.release_thrower_track_id = None
            cand.release_team_id = None
            return None
        release_ball_xy, inward, _release_displacement = release

        thrower = self._restart_player(
            possession=possession,
            tracks=tracks,
            ball_xy=release_ball_xy,
            side=cand.side,
            field_top=field_top,
            field_bottom=field_bottom,
        )
        if thrower is None and cand.setup_track_id is None:
            cand.inward_streak = 0
            return None

        if cand.release_frame is None:
            cand.release_frame = frame_index
            cand.release_timestamp_sec = timestamp_sec
            cand.release_thrower_track_id = cand.setup_track_id
            if cand.release_thrower_track_id is None and thrower is not None:
                cand.release_thrower_track_id = (
                    thrower.track_id if thrower.track_id >= 0 else None
                )
            cand.release_team_id = cand.setup_team_id
            if cand.release_team_id is None and thrower is not None:
                cand.release_team_id = track_to_team.get(thrower.track_id)
            cand.inward_streak = 1
            return None

        cand.inward_streak += 1
        if cand.inward_streak < self._cfg.restart_confirm_frames:
            return None

        self._event_id += 1
        self._last_emit_frame = frame_index
        self._out_candidate = None
        ev = ThrowInEvent(
            event_id=self._event_id,
            event_type="throw_in",
            out_frame=cand.frame_index,
            throw_frame=cand.release_frame,
            out_timestamp_sec=cand.timestamp_sec,
            throw_timestamp_sec=cand.release_timestamp_sec,
            side=cand.side,
            thrower_track_id=cand.release_thrower_track_id,
            team_id=cand.release_team_id,
            confidence=self._cfg.event_confidence,
            reason=(
                f"{cand.kind}_restart_inward_{inward:.1f}px_"
                f"setup_{cand.setup_frames}f_confirmed_{cand.inward_streak}f"
            ),
        )
        if self._cfg.debug:
            print(
                "[THROWIN] emit "
                f"id={ev.event_id} out_f={ev.out_frame} throw_f={ev.throw_frame} "
                f"side={ev.side} thrower={ev.thrower_track_id}",
                flush=True,
            )
        return ev

    def _maybe_override_candidate_side(
        self,
        cand: _OutCandidate,
        *,
        frame_index: int,
        timestamp_sec: Optional[float],
        ball_candidates: tuple[tuple[float, float], ...],
        tracks: FrameTracks,
        track_to_team: dict[int, int],
        field_top: float,
        field_bottom: float,
        age: int,
    ) -> bool:
        """Move a stale out candidate to a stronger opposite touchline setup."""
        if cand.kind != "visible_out":
            return False
        if cand.side != "top":
            return False
        if age < self._cfg.side_override_min_age_frames:
            return False
        if cand.setup_frames >= self._cfg.setup_confirm_frames:
            self._reset_side_override(cand)
            return False

        same_holder = self._touchline_holder(
            tracks=tracks,
            ball_candidates=ball_candidates,
            field_top=field_top,
            field_bottom=field_bottom,
            track_to_team=track_to_team,
            side_filter=cand.side,
            anchor_x=cand.ball_xy[0],
        )
        if same_holder is not None:
            self._reset_side_override(cand)
            return False

        opposite_side = "bottom" if cand.side == "top" else "top"
        opposite_holder = self._touchline_holder(
            tracks=tracks,
            ball_candidates=ball_candidates,
            field_top=field_top,
            field_bottom=field_bottom,
            track_to_team=track_to_team,
            side_filter=opposite_side,
            anchor_x=cand.ball_xy[0],
        )
        if opposite_holder is None:
            self._reset_side_override(cand)
            return False

        holder_side, holder_player, holder_ball_xy = opposite_holder
        holder_track_id = holder_player.track_id if holder_player.track_id >= 0 else None
        if (
            cand.side_override_side == holder_side
            and cand.side_override_track_id == holder_track_id
        ):
            cand.side_override_frames += 1
        else:
            cand.side_override_side = holder_side
            cand.side_override_track_id = holder_track_id
            cand.side_override_start_frame = frame_index
            cand.side_override_frames = 1
        cand.side_override_ball_xy = holder_ball_xy

        if cand.side_override_frames < self._cfg.side_override_confirm_frames:
            return False

        cand.side = holder_side
        cand.kind = f"{cand.kind}_side_override"
        cand.ball_xy = holder_ball_xy
        cand.field_top = field_top
        cand.field_bottom = field_bottom
        cand.setup_track_id = None
        cand.setup_team_id = None
        cand.setup_start_frame = None
        cand.setup_frame = None
        cand.setup_timestamp_sec = None
        cand.setup_ball_xy = None
        cand.setup_frames = 0
        cand.release_frame = None
        cand.release_timestamp_sec = None
        cand.release_thrower_track_id = None
        cand.release_team_id = None
        cand.inward_streak = 0
        self._reset_side_override(cand)
        self._remember_holder_setup(
            cand,
            frame_index=frame_index,
            timestamp_sec=timestamp_sec,
            ball_xy=holder_ball_xy,
            holder=holder_player,
            track_to_team=track_to_team,
        )
        if self._cfg.debug:
            print(
                "[THROWIN] side_override "
                f"f={frame_index} side={holder_side} holder={holder_player.track_id}",
                flush=True,
            )
        return True

    @staticmethod
    def _reset_side_override(cand: _OutCandidate) -> None:
        cand.side_override_side = None
        cand.side_override_track_id = None
        cand.side_override_start_frame = None
        cand.side_override_frames = 0
        cand.side_override_ball_xy = None

    def _release_ball_candidate(
        self,
        cand: _OutCandidate,
        ball_xy: tuple[float, float],
        ball_candidates: tuple[tuple[float, float], ...],
    ) -> Optional[tuple[tuple[float, float], float, float]]:
        release_anchor = cand.setup_ball_xy or cand.ball_xy
        seen: set[tuple[int, int]] = set()
        ordered = (ball_xy,) + ball_candidates
        for candidate in ordered:
            key = (int(round(candidate[0])), int(round(candidate[1])))
            if key in seen:
                continue
            seen.add(key)
            release_displacement = _distance(candidate, release_anchor)
            min_displacement = (
                self._cfg.tracked_release_min_displacement_px
                if cand.kind
                in {"touchline_tracked_release", "touchline_throw_pose_release"}
                else self._cfg.release_min_displacement_px
            )
            if release_displacement < min_displacement:
                continue
            inward = self._inward_distance(cand.side, release_anchor, candidate)
            if inward < self._cfg.min_inward_ball_px:
                continue
            if inward > self._cfg.max_release_inward_px:
                continue
            return (candidate, inward, release_displacement)
        return None

    def _touchline_release_player(
        self,
        *,
        tracks: FrameTracks,
        ball_xy: tuple[float, float],
        field_top: float,
        field_bottom: float,
    ) -> Optional[tuple[str, TrackedInstance]]:
        best: Optional[tuple[float, str, TrackedInstance]] = None
        for player in tracks.instances:
            if player.role is not ObjectRole.PLAYER:
                continue
            side = self._player_side(
                player,
                field_top,
                field_bottom,
                margin_px=self._cfg.tracked_release_player_boundary_margin_px,
            )
            if side is None:
                continue
            px, py = _bbox_center(player.xyxy)
            center_dist = _distance((px, py), ball_xy)
            if center_dist > self._cfg.tracked_release_player_radius_px:
                continue
            if best is None or center_dist < best[0]:
                best = (center_dist, side, player)
        if best is None:
            return None
        return (best[1], best[2])

    def _touchline_throw_pose_release_player(
        self,
        *,
        tracks: FrameTracks,
        ball_xy: tuple[float, float],
        field_top: float,
        field_bottom: float,
    ) -> Optional[tuple[str, TrackedInstance]]:
        best: Optional[tuple[float, str, TrackedInstance]] = None
        for player in tracks.instances:
            if player.role is not ObjectRole.PLAYER:
                continue
            side = self._player_side(
                player,
                field_top,
                field_bottom,
                margin_px=self._cfg.throw_pose_player_boundary_margin_px,
            )
            if side is None:
                continue
            x1, y1, x2, y2 = player.xyxy
            width = max(1.0, x2 - x1)
            height = max(1.0, y2 - y1)
            bx, by = ball_xy
            x_margin = max(18.0, width * self._cfg.throw_pose_x_margin_ratio)
            if not (x1 - x_margin <= bx <= x2 + x_margin):
                continue
            upper_y = y1 - height * self._cfg.throw_pose_y_above_ratio
            lower_y = y1 + height * self._cfg.throw_pose_y_below_ratio
            if not (upper_y <= by <= lower_y):
                continue
            center_dist = _distance(_bbox_center(player.xyxy), ball_xy)
            if center_dist > self._cfg.throw_pose_player_radius_px:
                continue
            if best is None or center_dist < best[0]:
                best = (center_dist, side, player)
        if best is None:
            return None
        return (best[1], best[2])

    def _tracked_ball_near_player(
        self,
        *,
        tracks: FrameTracks,
        ball_xy: tuple[float, float],
        radius_px: float,
    ) -> bool:
        radius2 = float(radius_px) ** 2
        for player in tracks.instances:
            if player.role is not ObjectRole.PLAYER:
                continue
            px, py = _bbox_center(player.xyxy)
            dx = px - ball_xy[0]
            dy = py - ball_xy[1]
            if dx * dx + dy * dy <= radius2:
                return True
        return False

    def _remember_holder_setup(
        self,
        cand: _OutCandidate,
        *,
        frame_index: int,
        timestamp_sec: Optional[float],
        ball_xy: tuple[float, float],
        holder: TrackedInstance,
        track_to_team: dict[int, int],
    ) -> None:
        if cand.setup_track_id == holder.track_id:
            cand.setup_frames += 1
        else:
            cand.setup_track_id = holder.track_id if holder.track_id >= 0 else None
            cand.setup_team_id = track_to_team.get(holder.track_id)
            cand.setup_start_frame = frame_index
            cand.setup_frames = 1
        cand.setup_frame = frame_index
        cand.setup_timestamp_sec = timestamp_sec
        cand.setup_ball_xy = ball_xy

    def _touchline_holder(
        self,
        *,
        tracks: FrameTracks,
        ball_candidates: tuple[tuple[float, float], ...],
        field_top: float,
        field_bottom: float,
        track_to_team: dict[int, int],
        side_filter: Optional[str] = None,
        anchor_x: Optional[float] = None,
    ) -> Optional[tuple[str, TrackedInstance, tuple[float, float]]]:
        _ = track_to_team
        best: Optional[tuple[float, str, TrackedInstance, tuple[float, float]]] = None
        for player in tracks.instances:
            if player.role is not ObjectRole.PLAYER:
                continue
            side = self._player_side(player, field_top, field_bottom)
            if side is None or (side_filter is not None and side != side_filter):
                continue
            x1, y1, x2, y2 = player.xyxy
            width = max(1.0, x2 - x1)
            height = max(1.0, y2 - y1)
            for ball_xy in ball_candidates:
                bx, by = ball_xy
                if (
                    anchor_x is not None
                    and abs(bx - anchor_x) > self._cfg.setup_anchor_x_margin_px
                ):
                    continue
                x_margin = max(22.0, width * 0.95)
                upper_y = y1 - height * 0.95
                lower_y = y1 + height * 0.28
                if not (x1 - x_margin <= bx <= x2 + x_margin):
                    continue
                if not (upper_y <= by <= lower_y):
                    continue
                px, _py = _bbox_center(player.xyxy)
                d = _distance((px, y1), ball_xy)
                if best is None or d < best[0]:
                    best = (d, side, player, ball_xy)
        if best is None:
            return None
        return (best[1], best[2], best[3])

    def _player_side(
        self,
        player: TrackedInstance,
        field_top: float,
        field_bottom: float,
        *,
        margin_px: Optional[float] = None,
    ) -> Optional[str]:
        _x1, y1, _x2, y2 = player.xyxy
        margin = self._cfg.player_boundary_margin_px if margin_px is None else margin_px
        if y1 <= field_top + margin:
            return "top"
        if y2 >= field_bottom - margin:
            return "bottom"
        return None

    @staticmethod
    def _inward_distance(
        side: str, out_ball_xy: tuple[float, float], now_ball_xy: tuple[float, float]
    ) -> float:
        if side == "top":
            return now_ball_xy[1] - out_ball_xy[1]
        return out_ball_xy[1] - now_ball_xy[1]

    def _restart_player(
        self,
        *,
        possession: PossessionState,
        tracks: FrameTracks,
        ball_xy: tuple[float, float],
        side: str,
        field_top: float,
        field_bottom: float,
    ) -> Optional[TrackedInstance]:
        players = [inst for inst in tracks.instances if inst.role is ObjectRole.PLAYER]
        if possession.track_id is not None:
            for player in players:
                if player.track_id == possession.track_id:
                    return player

        best: Optional[TrackedInstance] = None
        best_d2 = float(self._cfg.restart_possession_radius_px) ** 2
        for player in players:
            px, py = _bbox_center(player.xyxy)
            if not self._player_near_side(py, side, field_top, field_bottom):
                continue
            dx = px - ball_xy[0]
            dy = py - ball_xy[1]
            d2 = dx * dx + dy * dy
            if d2 <= best_d2:
                best = player
                best_d2 = d2
        return best

    def _player_near_side(
        self, y: float, side: str, field_top: float, field_bottom: float
    ) -> bool:
        margin = self._cfg.player_boundary_margin_px
        if side == "top":
            return y <= field_top + margin
        return y >= field_bottom - margin


def _bbox_center(xyxy: tuple[float, float, float, float]) -> tuple[float, float]:
    x1, y1, x2, y2 = xyxy
    return (0.5 * (x1 + x2), 0.5 * (y1 + y2))


def _distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return float((dx * dx + dy * dy) ** 0.5)


def _candidate_points(
    tracked_ball_xy: Optional[tuple[float, float]],
    visual_ball_candidates: tuple[tuple[float, float], ...],
) -> tuple[tuple[float, float], ...]:
    pts: list[tuple[float, float]] = []
    if tracked_ball_xy is not None:
        pts.append(tracked_ball_xy)
    pts.extend(visual_ball_candidates)
    return tuple(pts)


def _bright_ball_candidates(frame: np.ndarray) -> tuple[tuple[float, float], ...]:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([0, 0, 145]), np.array([179, 95, 255]))
    mask = cv2.medianBlur(mask, 3)
    contours, _hierarchy = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    candidates: list[tuple[float, float, float]] = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 8.0 or area > 650.0:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        if w <= 1 or h <= 1:
            continue
        ratio = w / float(h)
        if ratio < 0.45 or ratio > 2.2:
            continue
        fill = area / float(w * h)
        if fill < 0.25:
            continue
        candidates.append((float(x + 0.5 * w), float(y + 0.5 * h), area))
    candidates.sort(key=lambda item: item[2], reverse=True)
    return tuple((x, y) for x, y, _area in candidates[:80])


def _ball_center(tracks: FrameTracks) -> Optional[tuple[float, float]]:
    balls = [inst for inst in tracks.instances if inst.role is ObjectRole.BALL]
    if not balls:
        return None
    ball = max(balls, key=lambda b: b.confidence)
    return _bbox_center(ball.xyxy)
