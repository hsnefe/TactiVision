"""Ball pass detection (controlled-possession FSM — release + confirmed receiver).

Separates transient ball/bbox overlaps from sustained control before opening pass candidates.

Uses :class:`~VisionEngine.schemas.schema.FrameTracks` for foot-point receiver proximity;
when ``tracks`` is omitted, receiver confirmation is skipped (no bbox-overlap fallback).
"""

from __future__ import annotations

import math
from collections import Counter, deque
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any, Optional

from VisionEngine.EventAnalytics.possession import PossessionState

if TYPE_CHECKING:
    from VisionEngine.schemas.schema import FrameTracks


@dataclass
class PassEvent:
    """A detected pass-like transfer of the ball between tracked players."""

    event_id: int
    event_type: str
    """``completed_pass`` | ``intercepted_pass`` | ``unknown_pass``."""

    start_frame: int
    end_frame: int

    start_time_sec: Optional[float]
    end_time_sec: Optional[float]

    from_player_id: Optional[int]

    to_player_id: Optional[int]

    from_team_id: Optional[int]
    to_team_id: Optional[int]

    duration_frames: int
    duration_sec: Optional[float]

    confidence: float

    ball_start_xy: Optional[tuple[float, float]]
    ball_end_xy: Optional[tuple[float, float]]

    raw_from_team_id: Optional[int] = None
    raw_to_team_id: Optional[int] = None
    team_stable: bool = False
    debug_reason: Optional[str] = None

    original_from_player_id: Optional[int] = None

    source: Optional[str] = None

    passer_detection_status: Optional[str] = None

    suppress_duplicate_pass_terminal: bool = False

    release_frame: Optional[int] = None
    receiver_confirm_start_frame: Optional[int] = None
    receiver_confirm_frames: Optional[int] = None
    ball_displacement_px: Optional[float] = None
    ignored_transient_contacts: Optional[int] = None
    pass_fsm_state: Optional[str] = None

    one_touch: bool = False
    one_touch_evidence: Optional[Any] = None
    """JSON-serializable evidence bag for tightened one-touch path."""

    post_receive_blocked: bool = False
    """Normal release was softened; true when passer used one-touch-eligible pathway."""

    source_reliability: Optional[str] = None
    """``reliable`` | ``unreliable`` | ``unknown`` — passer proximity near release."""

    source_reliability_reason: Optional[str] = None
    source_release_distance_px: Optional[float] = None
    source_release_radius_px: Optional[float] = None

    raw_from_player_id: Optional[int] = None
    raw_to_player_id: Optional[int] = None
    canonical_from_player_id: Optional[int] = None
    canonical_to_player_id: Optional[int] = None

    receiver_selection_reason: Optional[str] = None
    receiver_score: Optional[float] = None
    dribble_guard_status: Optional[str] = None
    id_alias_reason: Optional[str] = None
    receiver_contact_type: Optional[str] = None
    bbox_only_guard_status: Optional[str] = None
    receiver_bbox_intermediate_raw: Optional[int] = None
    """When retargeted from a bbox-only candidate: original ``to_player_id`` (intermediate track)."""

    touch_source_frames: Optional[int] = None
    touch_receiver_frames: Optional[int] = None
    touch_source_min_distance_px: Optional[float] = None
    touch_receiver_min_distance_px: Optional[float] = None
    avg_ball_speed_px_per_frame: Optional[float] = None

    source_recovery_method: Optional[str] = None
    """How a missing passer was recovered (optional JSON/debug metadata)."""

    invalid_intermediate_status: Optional[str] = None
    """Tracks bbox-only intermediate suppression context when relevant."""

    emit_path: Optional[str] = None
    """High-level pipeline identifier for debugging (e.g. ``primary_fsm``)."""

    source_decision_reason: Optional[str] = None
    receiver_decision_reason: Optional[str] = None

    source_contact_type: Optional[str] = None
    """Telemetry classifier for passer contact (distinct from reliability strings)."""

    receiver_contact_decision: Optional[str] = None
    """Telemetry classifier for receiver (``confirmed_receiver``, ``valid_touch``, …)."""

    source_visible_frames_near_release: Optional[int] = None
    receiver_visible_frames_near_endpoint: Optional[int] = None

    source_min_footpoint_distance_px: Optional[float] = None
    receiver_min_footpoint_distance_px: Optional[float] = None

    source_valid_touch_frames: Optional[int] = None
    receiver_valid_touch_frames: Optional[int] = None

    source_bbox_only_frames: Optional[int] = None
    receiver_bbox_only_frames: Optional[int] = None

    candidate_receivers_debug: Optional[list[dict[str, Any]]] = None
    """Top receiver scoring snapshot for the completion frame (debug only)."""

    bbox_guard_checked: bool = False
    bbox_guard_result: Optional[str] = None

    invalid_intermediate_checked: bool = False
    invalid_intermediate_result: Optional[str] = None

    source_recovery_checked: bool = False
    source_recovery_result: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        for key in ("ball_start_xy", "ball_end_xy"):
            v = data.get(key)
            if isinstance(v, tuple):
                data[key] = list(v)
        data.pop("suppress_duplicate_pass_terminal", None)
        return data


@dataclass
class PassDetectorConfig:
    """Config for :class:`PassDetector` (historic + newer FSM fields)."""

    min_possession_frames: int = 3
    """Legacy knob; mirrored from owner-stable defaults in ``Settings``."""

    min_pass_frames: int = 2
    cooldown_frames: int = 10
    same_player_grace_frames: int = 4
    require_same_team_for_completed: bool = True
    emit_interceptions: bool = True
    pass_debug: bool = False

    pass_team_stability_window: int = 10
    pass_team_min_stable_count: int = 6
    pass_unstable_team_as_unknown: bool = True

    pass_kickoff_bootstrap_guard_enabled: bool = True
    pass_kickoff_bootstrap_guard_max_frame: int = 70

    pass_owner_min_stable_frames: int = 5
    pass_owner_switch_min_frames: int = 5
    pass_transient_contact_max_frames: int = 4

    pass_release_min_frames_away: int = 3
    pass_release_min_distance_px: float = 25.0
    pass_release_min_ball_displacement_px: float = 20.0

    pass_receiver_confirm_frames: int = 5
    pass_receiver_max_gap_frames: int = 2
    pass_receiver_control_radius_px: float = 70.0

    pass_candidate_timeout_frames: int = 90
    """Discard in-flight candidates after this many frames without completion."""

    pass_post_receive_settle_frames: int = 12
    pass_post_receive_require_reconfirm: bool = True
    pass_allow_one_touch_release: bool = True
    pass_one_touch_window_frames: int = 8
    pass_one_touch_min_away_frames: int = 3
    pass_one_touch_min_outgoing_displacement_px: float = 30.0
    pass_one_touch_receiver_confirm_frames: int = 4
    pass_one_touch_control_radius_px: float = 75.0

    pass_source_reliability_enabled: bool = True
    pass_source_release_lookback_frames: int = 8
    pass_source_max_release_distance_px: float = 115.0
    pass_source_unknown_if_unreliable: bool = True

    pass_touch_fallback_enabled: bool = True
    pass_touch_history_frames: int = 80
    pass_touch_source_lookback_frames: int = 14
    pass_touch_receiver_lookahead_frames: int = 12

    pass_touch_contact_radius_px: float = 90.0
    pass_touch_strong_contact_radius_px: float = 65.0

    pass_touch_min_source_contact_frames: int = 2
    pass_touch_min_receiver_contact_frames: int = 3

    pass_touch_min_displacement_px: float = 25.0
    pass_touch_max_duration_frames: int = 50
    pass_touch_min_avg_speed_px_per_frame: float = 1.0

    pass_touch_duplicate_window_frames: int = 30
    pass_touch_confidence_cap: float = 0.78

    pass_touch_ignore_bbox_only_contact: bool = True
    pass_touch_use_footpoint_distance: bool = True

    pass_source_reliability_primary_fsm_grace_enabled: bool = True
    pass_source_reliability_lookback_frames: int = 16
    pass_source_reliability_lookahead_frames: int = 3
    pass_source_reliability_dynamic_radius_enabled: bool = True
    pass_source_reliability_min_radius_px: float = 80.0
    pass_source_reliability_bbox_height_ratio: float = 0.70
    pass_source_reliability_keep_primary_fsm_source: bool = True

    pass_touch_dynamic_radius_enabled: bool = True
    pass_touch_min_radius_px: float = 35.0
    pass_touch_max_radius_px: float = 95.0
    pass_touch_bbox_height_ratio: float = 0.28
    pass_touch_lower_body_fraction: float = 0.45
    pass_touch_require_lower_body_or_footpoint: bool = True
    pass_touch_bbox_overlap_only_penalty: float = 0.50

    pass_dribble_guard_enabled: bool = True
    pass_dribble_guard_window_frames: int = 12
    pass_dribble_guard_return_to_same_player_frames: int = 8
    pass_dribble_guard_max_receiver_contact_frames: int = 2
    pass_dribble_guard_min_receiver_distance_gain_px: float = 30.0
    pass_dribble_guard_skip_if_source_retains_control: bool = True

    pass_receiver_candidate_score_enabled: bool = True
    pass_receiver_bbox_only_max_score: float = 0.4
    pass_receiver_bbox_only_cannot_win: bool = True
    pass_receiver_min_score: float = 1.0

    pass_event_id_alias_enabled: bool = True
    pass_event_id_alias_max_gap_frames: int = 20
    pass_event_id_alias_max_center_distance_px: float = 65.0
    pass_event_id_alias_require_same_team: bool = True
    pass_event_id_alias_require_similar_bbox: bool = True
    pass_event_id_alias_bbox_height_ratio_tol: float = 0.35
    pass_event_id_alias_apply_to_output: bool = False
    pass_event_id_alias_apply_to_decision: bool = False

    pass_source_reliability_keep_visible_primary_source: bool = True
    pass_source_reliability_primary_keep_lookback_frames: int = 20
    pass_source_reliability_primary_keep_lookahead_frames: int = 5
    pass_source_reliability_primary_keep_min_visible_frames: int = 2
    pass_source_reliability_primary_keep_radius_px: float = 190.0
    pass_source_reliability_primary_keep_nearest_rank: int = 3

    pass_bbox_only_intermediate_guard_enabled: bool = True
    pass_bbox_only_max_contact_frames: int = 2
    pass_bbox_only_min_footpoint_distance_px: float = 120.0
    pass_bbox_only_chain_window_frames: int = 18
    pass_bbox_only_require_lower_body_touch: bool = True

    pass_valid_touch_gate_enabled: bool = True
    pass_valid_touch_history_frames: int = 80
    pass_valid_touch_max_gap_frames: int = 4
    pass_valid_touch_min_frames: int = 2
    pass_valid_touch_min_radius_px: float = 35.0
    pass_valid_touch_max_radius_px: float = 105.0
    pass_valid_touch_bbox_height_ratio: float = 0.32
    pass_valid_touch_lower_body_fraction: float = 0.45
    pass_valid_touch_bbox_only_footpoint_min_px: float = 125.0

    pass_source_recover_from_valid_touch_enabled: bool = True
    pass_source_recover_lookback_frames: int = 28
    pass_source_recover_max_distance_px: float = 170.0
    pass_source_recover_require_same_primary_context: bool = False

    pass_receiver_retarget_after_bbox_only_enabled: bool = True
    pass_receiver_retarget_lookahead_frames: int = 18
    pass_receiver_retarget_max_distance_px: float = 180.0

    pass_short_valid_touch_fallback_enabled: bool = False
    pass_short_valid_touch_max_duration_frames: int = 32
    pass_short_valid_touch_min_displacement_px: float = 22.0
    pass_short_valid_touch_duplicate_window_frames: int = 24
    pass_short_valid_touch_confidence_cap: float = 0.76

    pass_strict_bbox_only_classification_enabled: bool = True

    pass_none_source_recovery_enabled: bool = True
    pass_none_source_recovery_lookback_frames: int = 36
    pass_none_source_recovery_lookahead_frames: int = 4
    pass_none_source_recovery_min_visible_frames: int = 2
    pass_none_source_recovery_max_distance_px: float = 230.0
    pass_none_source_recovery_use_original_from: bool = True
    pass_none_source_recovery_use_last_valid_touch: bool = True
    pass_none_source_recovery_skip_kickoff_guard: bool = True

    pass_invalid_intermediate_memory_enabled: bool = True
    pass_invalid_intermediate_ttl_frames: int = 40
    pass_invalid_intermediate_min_footpoint_distance_px: float = 130.0
    pass_invalid_intermediate_max_valid_touch_frames: int = 1
    pass_invalid_intermediate_require_no_lower_body_touch: bool = True

    pass_chain_retarget_enabled: bool = True
    pass_chain_retarget_window_frames: int = 36
    pass_chain_retarget_require_valid_receiver_touch: bool = True
    pass_chain_retarget_max_receiver_distance_px: float = 220.0

    pass_short_valid_touch_fallback_recall_enabled: bool = False
    pass_short_valid_touch_fallback_max_duration_frames: int = 42
    pass_short_valid_touch_fallback_min_displacement_px: float = 18.0
    pass_short_valid_touch_fallback_duplicate_window_frames: int = 30
    pass_short_valid_touch_fallback_confidence_cap: float = 0.74

    pass_debug_throttle_duplicate_lines_frames: int = 30


@dataclass
class _FlightCandidate:
    start_frame: int
    start_time_sec: Optional[float]
    from_player_id: int
    from_team_id: Optional[int]
    ball_anchor_xy: Optional[tuple[float, float]]
    release_frame: int


@dataclass
class _PassFrameSnap:
    frame_index: int
    ball_xy: Optional[tuple[float, float]]
    feet: dict[int, tuple[float, float]]
    bbox_heights: dict[int, float]


@dataclass
class _TouchHistTrack:
    foot_xy: tuple[float, float]
    bbox: tuple[float, float, float, float]
    d_foot: float
    valid_touch: bool
    strong_touch: bool
    bbox_overlap_only: bool
    lower_body_valid: bool


@dataclass
class _TouchHistFrame:
    frame_index: int
    ball_xy: Optional[tuple[float, float]]
    tracks: dict[int, _TouchHistTrack]


@dataclass
class _LostTrackSig:
    track_id: int
    last_frame: int
    center_xy: tuple[float, float]
    bbox_h: float
    bbox_w: float
    team_id: Optional[int]


@dataclass
class _TouchEpisode:
    player_id: int
    team_id: Optional[int]
    start_frame: int
    end_frame: int
    contact_frames: int
    strong_contact_frames: int
    lower_body_touch_frames: int
    bbox_overlap_only_frames: int
    min_distance_to_ball: float
    first_ball_xy: Optional[tuple[float, float]]
    last_ball_xy: Optional[tuple[float, float]]


@dataclass
class _TouchOpenEpisode:
    player_id: int
    team_id_snap: Optional[int]
    start_frame: int
    end_frame: int
    misses: int
    contact_frames: int
    strong_contact_frames: int
    lower_body_touch_frames: int
    bbox_overlap_only_frames: int
    min_d: float
    first_ball_xy: Optional[tuple[float, float]]
    last_ball_xy: Optional[tuple[float, float]]


@dataclass
class _ValidTouchHistTrack:
    valid_touch: bool
    bbox_only: bool
    d_foot: float


@dataclass
class _ValidTouchHistFrame:
    frame_index: int
    ball_xy: Optional[tuple[float, float]]
    tracks: dict[int, _ValidTouchHistTrack]


@dataclass
class _ValidTouchEpisode:
    player_id: int
    team_id: Optional[int]
    start_frame: int
    end_frame: int
    contact_frames: int
    min_distance_to_ball: float
    first_ball_xy: Optional[tuple[float, float]]
    last_ball_xy: Optional[tuple[float, float]]


@dataclass
class _ValidTouchOpenEpisode:
    player_id: int
    team_id_snap: Optional[int]
    start_frame: int
    end_frame: int
    misses: int
    contact_frames: int
    min_d: float
    first_ball_xy: Optional[tuple[float, float]]
    last_ball_xy: Optional[tuple[float, float]]


@dataclass
class _PendingReturnPass:
    from_player_id: int
    to_player_id: int
    from_team_id: Optional[int]
    to_team_id: Optional[int]
    start_frame: int
    release_anchor_xy: tuple[float, float]
    expires_frame: int
    streak: int = 0
    first_candidate_frame: Optional[int] = None


@dataclass
class _FlightContactEpisode:
    player_id: int
    team_id: Optional[int]
    start_frame: int
    end_frame: int
    frames: int
    possession_frames: int
    candidate_frames: int
    valid_touch_frames: int
    lower_body_frames: int
    bbox_only_frames: int
    min_distance_to_ball: float
    first_ball_xy: Optional[tuple[float, float]]
    last_ball_xy: Optional[tuple[float, float]]


@dataclass
class _FlightContactOpenEpisode:
    player_id: int
    team_id_snap: Optional[int]
    start_frame: int
    end_frame: int
    misses: int
    frames: int
    possession_frames: int
    candidate_frames: int
    valid_touch_frames: int
    lower_body_frames: int
    bbox_only_frames: int
    min_d: float
    first_ball_xy: Optional[tuple[float, float]]
    last_ball_xy: Optional[tuple[float, float]]


def bbox_center(xyxy: tuple[float, float, float, float]) -> tuple[float, float]:
    x1, y1, x2, y2 = xyxy
    return (0.5 * (x1 + x2), 0.5 * (y1 + y2))


def _ball_inside_bbox(xyxy: tuple[float, float, float, float], bxy: tuple[float, float]) -> bool:
    x, y = bxy
    x1, y1, x2, y2 = xyxy
    return x1 <= x <= x2 and y1 <= y <= y2


def _ball_in_bbox_lower_half(
    xyxy: tuple[float, float, float, float], bxy: tuple[float, float]
) -> bool:
    if not _ball_inside_bbox(xyxy, bxy):
        return False
    x1, y1, x2, y2 = xyxy
    mid_y = y1 + 0.5 * (y2 - y1)
    return bxy[1] >= mid_y


def foot_xy(xyxy: tuple[float, float, float, float]) -> tuple[float, float]:
    x1, _, x2, y2 = xyxy
    return (0.5 * (x1 + x2), y2)


def _dist_px(a: tuple[float, float], b: tuple[float, float]) -> float:
    return float(math.hypot(a[0] - b[0], a[1] - b[1]))


def _point_segment_projection_t(
    p: tuple[float, float],
    a: tuple[float, float],
    b: tuple[float, float],
) -> float:
    vx = b[0] - a[0]
    vy = b[1] - a[1]
    denom = vx * vx + vy * vy
    if denom <= 1e-9:
        return 0.0
    return _clampf(((p[0] - a[0]) * vx + (p[1] - a[1]) * vy) / denom, 0.0, 1.0)


def _segment_intersects_bbox(
    a: tuple[float, float],
    b: tuple[float, float],
    xyxy: tuple[float, float, float, float],
    *,
    pad: float = 0.0,
) -> bool:
    x1, y1, x2, y2 = xyxy
    x1 -= pad
    y1 -= pad
    x2 += pad
    y2 += pad
    if (x1 <= a[0] <= x2 and y1 <= a[1] <= y2) or (
        x1 <= b[0] <= x2 and y1 <= b[1] <= y2
    ):
        return True

    def ccw(p1, p2, p3) -> bool:
        return (p3[1] - p1[1]) * (p2[0] - p1[0]) > (
            p2[1] - p1[1]
        ) * (p3[0] - p1[0])

    def seg_inter(p1, p2, p3, p4) -> bool:
        return ccw(p1, p3, p4) != ccw(p2, p3, p4) and ccw(p1, p2, p3) != ccw(
            p1, p2, p4
        )

    corners = ((x1, y1), (x2, y1), (x2, y2), (x1, y2))
    edges = zip(corners, corners[1:] + corners[:1])
    return any(seg_inter(a, b, e1, e2) for e1, e2 in edges)


def _bbox_height_xyxy(xyxy: tuple[float, float, float, float]) -> float:
    return max(1.0, float(xyxy[3] - xyxy[1]))


def _clampf(x: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, x)))


def _dynamic_touch_radius_px(cfg: PassDetectorConfig, bbox_h: float) -> float:
    if cfg.pass_touch_dynamic_radius_enabled:
        r = bbox_h * cfg.pass_touch_bbox_height_ratio
        return _clampf(
            r,
            cfg.pass_touch_min_radius_px,
            cfg.pass_touch_max_radius_px,
        )
    return float(cfg.pass_touch_contact_radius_px)


def _ball_in_lower_body_zone(
    xyxy: tuple[float, float, float, float],
    bxy: tuple[float, float],
    lower_frac: float,
) -> bool:
    x1, y1, x2, y2 = xyxy
    if not _ball_inside_bbox(xyxy, bxy):
        return False
    split_y = y1 + (1.0 - lower_frac) * (y2 - y1)
    return bxy[1] >= split_y


def classify_touch_features(
    cfg: PassDetectorConfig,
    bbox: tuple[float, float, float, float],
    ball_xy: tuple[float, float],
) -> tuple[float, bool, bool, bool]:
    """d_foot, lower_body_ok, valid_control_touch (foot or lower-zone), bbox_overlap_only."""

    foot = foot_xy(bbox)
    d_f = _dist_px(foot, ball_xy)
    bh = _bbox_height_xyxy(bbox)
    dyn_r = _dynamic_touch_radius_px(cfg, bh)

    overlap = _ball_inside_bbox(bbox, ball_xy)
    in_lower = _ball_in_lower_body_zone(bbox, ball_xy, cfg.pass_touch_lower_body_fraction)
    foot_like = d_f <= dyn_r
    near_below = in_lower and d_f <= dyn_r * 1.18
    lower_body_ok = foot_like or near_below

    bbox_only = overlap and not lower_body_ok
    if cfg.pass_touch_require_lower_body_or_footpoint:
        valid = lower_body_ok
    else:
        valid = (
            overlap
            if not cfg.pass_touch_ignore_bbox_only_contact
            else (foot_like or overlap)
        )
    dyn_strong_thr = dyn_r * 0.62
    strong_candidate = foot_like and d_f <= dyn_strong_thr
    strong = lower_body_ok and (
        strong_candidate or (
            cfg.pass_touch_strong_contact_radius_px > 0
            and d_f <= min(dyn_strong_thr, cfg.pass_touch_strong_contact_radius_px)
        )
    )
    return (d_f, lower_body_ok, valid, bbox_only)


def _dynamic_valid_touch_radius_px(cfg: PassDetectorConfig, bbox_h: float) -> float:
    r = bbox_h * cfg.pass_valid_touch_bbox_height_ratio
    return _clampf(
        r,
        cfg.pass_valid_touch_min_radius_px,
        cfg.pass_valid_touch_max_radius_px,
    )


def classify_valid_touch_gate(
    cfg: PassDetectorConfig,
    bbox: tuple[float, float, float, float],
    ball_xy: tuple[float, float],
) -> tuple[bool, bool, float]:
    """valid_touch flag, bbox_only (intersects bbox without valid touch), foot–ball distance."""

    foot = foot_xy(bbox)
    d_f = _dist_px(foot, ball_xy)
    bh = _bbox_height_xyxy(bbox)
    dyn_r = _dynamic_valid_touch_radius_px(cfg, bh)
    in_lower = _ball_in_lower_body_zone(
        bbox, ball_xy, cfg.pass_valid_touch_lower_body_fraction
    )
    near_foot = d_f <= dyn_r
    in_lower_close = in_lower and d_f <= dyn_r * 1.22
    valid_touch = near_foot or in_lower_close
    overlap = _ball_inside_bbox(bbox, ball_xy)
    bbox_only = overlap and not valid_touch
    return valid_touch, bbox_only, d_f


def _dynamic_source_radius_px(
    cfg: PassDetectorConfig, bbox_height: Optional[float]
) -> Optional[float]:
    if not cfg.pass_source_reliability_dynamic_radius_enabled or bbox_height is None:
        return None
    r = bbox_height * cfg.pass_source_reliability_bbox_height_ratio
    allowed = max(cfg.pass_source_reliability_min_radius_px, float(r))
    return float(allowed)


class PassDetector:
    """Controlled-possession pass FSM."""

    def __init__(self, config: PassDetectorConfig) -> None:
        self._config = config
        self._team_history: dict[int, deque[int]] = {}

        self._next_event_id: int = 1
        self._cooldown_until_frame: int = -10**9
        self._emitted_any_pass: bool = False

        # Foot cache each frame from tracks (best-effort)
        self._last_foot: dict[int, tuple[float, float]] = {}

        # Controlled owner smoothing
        self._co_owner: Optional[int] = None
        self._run_tid: Optional[int] = None
        self._same_run: int = 0
        self._alt_tid: Optional[int] = None
        self._alt_run: int = 0
        self._last_owner_switch: Optional[tuple[int, int, int]] = None

        # Release buildup
        self._away_run: int = 0
        self._away_anchor: Optional[tuple[float, float]] = None

        # In-flight
        self._flight: Optional[_FlightCandidate] = None
        self._recv_tid: Optional[int] = None
        self._recv_streak: int = 0
        self._recv_gap: int = 0
        self._recv_confirm_start_frame: Optional[int] = None

        self._intr_tid: Optional[int] = None
        self._intr_run: int = 0
        self._ignored_transient_episodes: int = 0
        self._flight_contact_open: dict[int, _FlightContactOpenEpisode] = {}
        self._flight_contact_finished: list[_FlightContactEpisode] = []

        self._reclaim_run: int = 0

        self._last_event_end_frame: Optional[int] = None
        self._last_receiver_id: Optional[int] = None
        self._last_event_from_player_id: Optional[int] = None
        self._last_event_pair: Optional[tuple[Optional[int], Optional[int]]] = None
        self._receiver_release_blocked_until: dict[int, int] = {}

        self._suppress_release_episode_logged: Optional[int] = None
        self._release_allow_logged_tid: Optional[int] = None

        self._one_touch_contact_xy: Optional[tuple[float, float]] = None
        self._one_touch_contact_frame: Optional[int] = None
        self._one_touch_first_away_frame: Optional[int] = None
        self._one_touch_away_run: int = 0
        self._one_touch_max_out: float = 0.0
        self._one_touch_recv_tid: Optional[int] = None
        self._one_touch_recv_streak: int = 0
        self._one_touch_recv_gap: int = 0
        self._one_touch_recv_confirm_start_frame: Optional[int] = None

        cap = (
            config.pass_candidate_timeout_frames
            + max(
                config.pass_source_release_lookback_frames,
                config.pass_source_reliability_lookback_frames,
            )
            + config.pass_source_reliability_lookahead_frames
            + 64
        )
        self._snap_hist: deque[_PassFrameSnap] = deque(maxlen=max(cap, 128))

        th = max(32, config.pass_touch_history_frames)
        self._touch_hist: deque[_TouchHistFrame] = deque(maxlen=th)
        self._touch_open: dict[int, _TouchOpenEpisode] = {}
        self._touch_finished: deque[_TouchEpisode] = deque(maxlen=96)
        self._pass_emission_log: deque[PassEvent] = deque(maxlen=512)
        self._pending_weak_chain_event: Optional[PassEvent] = None
        self._pending_return_pass: Optional[_PendingReturnPass] = None

        self._last_bbox_heights: dict[int, float] = {}
        self._last_bbox_xyxy: dict[int, tuple[float, float, float, float]] = {}

        self._dbg_throttle: dict[str, int] = {}

        self._prev_present_track_ids: set[int] = set()
        self._lost_track_buffer: deque[_LostTrackSig] = deque(maxlen=64)

        self._track_alias_fwd: dict[int, int] = {}
        self._track_alias_reason: dict[str, str] = {}
        self._tracks_ever_seen: set[int] = set()
        self._suppressed_bbox_receiver_chain: deque[tuple[int, int]] = deque(maxlen=96)
        self._invalid_intermediate_until: dict[int, int] = {}

        v_hist = max(32, config.pass_valid_touch_history_frames)
        self._valid_touch_snap_hist: deque[_ValidTouchHistFrame] = deque(maxlen=v_hist)
        self._valid_touch_open: dict[int, _ValidTouchOpenEpisode] = {}
        self._valid_touch_finished: deque[_ValidTouchEpisode] = deque(maxlen=128)

        self._player_geom_previous_frame: dict[
            int, tuple[int, tuple[float, float, float, float], Optional[int]]
        ] = {}

    def reset(self) -> None:
        self.__init__(self._config)

    def _record_teams_for_frame(self, track_to_team: dict[int, int]) -> None:
        w = max(1, self._config.pass_team_stability_window)
        for tid, team_id in track_to_team.items():
            dq = self._team_history.setdefault(tid, deque(maxlen=w))
            dq.append(team_id)

    def _team_label_stable_for_track(self, tid: int, team_id: int | None) -> bool:
        if team_id is None:
            return False
        dq = self._team_history.get(tid)
        need = max(1, self._config.pass_team_min_stable_count)
        if dq is None or len(dq) < need:
            return False
        hist = list(dq)
        if sum(1 for x in hist if x == team_id) < need:
            return False
        mode_tid, mode_cnt = Counter(hist).most_common(1)[0]
        return mode_tid == team_id and mode_cnt >= need

    def _team_history_mode_for_track(self, tid: int) -> Optional[int]:
        dq = self._team_history.get(tid)
        if not dq:
            return None
        return Counter(dq).most_common(1)[0][0]

    def _dbg(self, message: str) -> None:
        if self._config.pass_debug:
            print(message, flush=True)

    @staticmethod
    def _resolver_team(
        possession: PossessionState, track_id: int, track_to_team: dict[int, int]
    ) -> Optional[int]:
        if possession.track_id == track_id and possession.team_id is not None:
            return possession.team_id
        return track_to_team.get(track_id)

    def _update_foot_cache(self, tracks: Optional["FrameTracks"]) -> None:
        self._last_foot.clear()
        self._last_bbox_heights.clear()
        self._last_bbox_xyxy.clear()
        if tracks is None:
            return
        from VisionEngine.schemas.schema import ObjectRole

        for inst in tracks.instances:
            if inst.role is not ObjectRole.PLAYER:
                continue
            if inst.track_id < 0:
                continue
            xyxy = tuple(float(x) for x in inst.xyxy)
            tid = inst.track_id
            self._last_foot[tid] = foot_xy(xyxy)
            self._last_bbox_xyxy[tid] = xyxy
            self._last_bbox_heights[tid] = _bbox_height_xyxy(xyxy)

    def _foot_of(self, tid: int, possession: PossessionState) -> Optional[tuple[float, float]]:
        if tid in self._last_foot:
            return self._last_foot[tid]
        return None

    def _nearest_receiver_tid_legacy(
        self,
        *,
        tracks: Optional["FrameTracks"],
        ball_xy: tuple[float, float],
        exclude: int,
        radius_px: Optional[float] = None,
    ) -> tuple[Optional[int], float]:
        cfg = self._config
        if tracks is None:
            return None, float("-inf")

        radius = (
            radius_px
            if radius_px is not None
            else cfg.pass_receiver_control_radius_px
        )
        from VisionEngine.schemas.schema import ObjectRole

        best_tid: Optional[int] = None
        best_d = float("inf")
        for inst in tracks.instances:
            if inst.role is not ObjectRole.PLAYER:
                continue
            if inst.track_id < 0 or inst.track_id == exclude:
                continue
            fxy = foot_xy(inst.xyxy)
            d = _dist_px(fxy, ball_xy)
            if d <= radius and d < best_d:
                best_d = d
                best_tid = inst.track_id
        if best_tid is None:
            return None, float("-inf")
        return best_tid, -best_d

    def _score_receiver_candidate_one_frame(
        self,
        *,
        track_id: int,
        bbox: tuple[float, float, float, float],
        ball_xy: tuple[float, float],
        exclude: int,
        release_anchor: Optional[tuple[float, float]],
        ft: tuple[float, float],
        d_foot: float,
        lower_body_valid: bool,
        bbox_overlap_only: bool,
        team_id: Optional[int],
        track_to_team: dict[int, int],
        team_stable: bool,
    ) -> float:
        cfg = self._config
        if track_id == exclude:
            return float("-inf")
        score = 0.0
        if lower_body_valid:
            score += 2.6
        else:
            score -= 0.8
        if bbox_overlap_only:
            score -= cfg.pass_touch_bbox_overlap_only_penalty
            score = min(score, cfg.pass_receiver_bbox_only_max_score)
        score += max(0.0, 2.2 - d_foot / 45.0)
        if release_anchor is not None:
            v1 = (ball_xy[0] - release_anchor[0], ball_xy[1] - release_anchor[1])
            v2 = (ft[0] - release_anchor[0], ft[1] - release_anchor[1])
            n1 = math.hypot(v1[0], v1[1])
            n2 = math.hypot(v2[0], v2[1])
            if n1 > 1e-3 and n2 > 1e-3:
                cos = (v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2)
                score += max(0.0, cos) * 1.1
        if team_stable and team_id is not None:
            score += 0.25
        return score

    def _pick_receiver_for_pass(
        self,
        *,
        tracks: Optional["FrameTracks"],
        ball_xy: Optional[tuple[float, float]],
        exclude: int,
        frame_index: int,
        release_anchor: Optional[tuple[float, float]],
        radius_px: Optional[float],
        track_to_team: dict[int, int],
    ) -> tuple[Optional[int], str, float]:
        cfg = self._config
        if tracks is None or ball_xy is None:
            return None, "missing_tracks_ball", float("-inf")

        legacy_tid, legacy_margin = self._nearest_receiver_tid_legacy(
            tracks=tracks,
            ball_xy=ball_xy,
            exclude=exclude,
            radius_px=radius_px,
        )

        if not cfg.pass_receiver_candidate_score_enabled:
            if legacy_tid is None:
                return None, "legacy_none", legacy_margin
            return legacy_tid, "nearest_foot", 1.0

        from VisionEngine.schemas.schema import ObjectRole

        search_r = radius_px if radius_px is not None else cfg.pass_receiver_control_radius_px
        search_r = float(search_r) * 1.45 + 40.0

        raw_scored: list[tuple[int, float, bool, bool]] = []

        for inst in tracks.instances:
            if inst.role is not ObjectRole.PLAYER:
                continue
            tid = inst.track_id
            if tid < 0 or tid == exclude:
                continue
            if (
                cfg.pass_invalid_intermediate_memory_enabled
                and self._invalid_intermediate_active(tid, frame_index)
            ):
                continue
            bbox = tuple(float(x) for x in inst.xyxy)
            ft = foot_xy(bbox)
            d_near = _dist_px(ft, ball_xy)
            if d_near > search_r:
                continue
            d_foot, lower_ok, valid_ctrl, bbox_only_flag = classify_touch_features(
                cfg, bbox, ball_xy
            )
            qual_touch = lower_ok or valid_ctrl
            tm = track_to_team.get(tid)
            stab = tm is None or self._team_label_stable_for_track(tid, tm)
            sc = self._score_receiver_candidate_one_frame(
                track_id=tid,
                bbox=bbox,
                ball_xy=tuple(ball_xy),
                exclude=exclude,
                release_anchor=release_anchor,
                ft=ft,
                d_foot=d_foot,
                lower_body_valid=lower_ok,
                bbox_overlap_only=bbox_only_flag,
                team_id=tm,
                track_to_team=track_to_team,
                team_stable=stab,
            )
            raw_scored.append((tid, sc, qual_touch, bbox_only_flag))

        if not raw_scored:
            if legacy_tid is not None:
                return legacy_tid, "nearest_foot_fallback_pool_empty", legacy_margin
            return None, "no_candidates", float("-inf")

        if cfg.pass_receiver_bbox_only_cannot_win:
            qualified = [x for x in raw_scored if x[2]]
            pool_ids = qualified if qualified else raw_scored
        else:
            any_lb = any(x[2] for x in raw_scored)
            pool_ids = raw_scored if not any_lb else [x for x in raw_scored if x[2]]

        tid_best, score_best, _, _ = max(pool_ids, key=lambda z: z[1])
        if score_best < cfg.pass_receiver_min_score:
            if legacy_tid is not None:
                return (
                    legacy_tid,
                    "below_min_fallback_nearest",
                    float(max(legacy_margin, cfg.pass_receiver_min_score * 0.6)),
                )
            return None, "below_min_score", score_best

        reason = "footpoint_score" if tid_best != legacy_tid else "nearest_aligned_score"
        if cfg.pass_debug and legacy_tid is not None and legacy_tid != tid_best:
            leg = next((x for x in raw_scored if x[0] == legacy_tid), None)
            win = next((x for x in raw_scored if x[0] == tid_best), None)
            if (
                cfg.pass_receiver_bbox_only_cannot_win
                and leg is not None
                and win is not None
                and leg[3]
                and not leg[2]
                and win[2]
            ):
                self._dbg_gate(
                    f"recv_bboxbeat:{legacy_tid}->{tid_best}",
                    frame_index,
                    "[PASS-RECEIVER] choose "
                    f"{tid_best} over {legacy_tid} reason=valid_touch_beats_bbox_only",
                )
            else:
                self._dbg_gate(
                    f"recv_cmp:{legacy_tid}->{tid_best}",
                    frame_index,
                    "[PASS-RECEIVER] choose "
                    f"{tid_best} over {legacy_tid} reason={reason} score={score_best:.2f}",
                )
        return tid_best, reason, score_best

    def _nearest_receiver_tid(
        self,
        *,
        tracks: Optional["FrameTracks"],
        ball_xy: tuple[float, float],
        exclude: int,
        radius_px: Optional[float] = None,
        frame_index: int = 0,
        release_anchor: Optional[tuple[float, float]] = None,
        track_to_team: Optional[dict[int, int]] = None,
    ) -> Optional[int]:
        tt = track_to_team if track_to_team is not None else {}
        tid, _, _ = self._pick_receiver_for_pass(
            tracks=tracks,
            ball_xy=ball_xy,
            exclude=exclude,
            frame_index=frame_index,
            release_anchor=release_anchor,
            radius_px=radius_px,
            track_to_team=tt,
        )
        if tid is None:
            tid, _ = self._nearest_receiver_tid_legacy(
                tracks=tracks,
                ball_xy=ball_xy,
                exclude=exclude,
                radius_px=radius_px,
            )
        return tid

    def _clear_one_touch_accumulator(self) -> None:
        self._one_touch_contact_xy = None
        self._one_touch_contact_frame = None
        self._one_touch_first_away_frame = None
        self._one_touch_away_run = 0
        self._one_touch_max_out = 0.0
        self._one_touch_recv_tid = None
        self._one_touch_recv_streak = 0
        self._one_touch_recv_gap = 0
        self._one_touch_recv_confirm_start_frame = None

    def _post_receive_blocked(self, owner_id: Optional[int], frame_index: int) -> bool:
        cfg = self._config
        if owner_id is None or not cfg.pass_post_receive_require_reconfirm:
            return False
        until = self._receiver_release_blocked_until.get(owner_id)
        return until is not None and frame_index < until

    def _expire_receive_blocks(self, frame_index: int) -> None:
        cfg = self._config
        for tid in list(self._receiver_release_blocked_until.keys()):
            until = self._receiver_release_blocked_until[tid]
            if frame_index < until:
                continue
            if (
                cfg.pass_debug
                and tid == self._last_receiver_id
                and tid == self._co_owner
                and self._release_allow_logged_tid != tid
            ):
                self._dbg(
                    "[PASS-FSM] receiver release allowed "
                    f"owner={tid} current={frame_index}"
                )
                self._release_allow_logged_tid = tid
            del self._receiver_release_blocked_until[tid]

    def _record_pass_frame_snap(
        self, frame_index: int, ball_xy: Optional[tuple[float, float]]
    ) -> None:
        cfg = self._config
        if not cfg.pass_source_reliability_enabled:
            return
        self._snap_hist.append(
            _PassFrameSnap(
                frame_index=frame_index,
                ball_xy=tuple(ball_xy) if ball_xy is not None else None,
                feet=dict(self._last_foot),
                bbox_heights=dict(self._last_bbox_heights),
            )
        )

    def _dbg_gate(self, gate_key: str, frame_index: int, message: str) -> None:
        if not self._config.pass_debug:
            return
        th = max(1, self._config.pass_debug_throttle_duplicate_lines_frames)
        last = self._dbg_throttle.get(gate_key, -10**9)
        if frame_index - last < th:
            return
        self._dbg_throttle[gate_key] = frame_index
        self._dbg(message)

    def _prune_invalid_intermediate(self, frame_index: int) -> None:
        cfg = self._config
        if not cfg.pass_invalid_intermediate_memory_enabled:
            return
        dead = [
            tid
            for tid, exp in self._invalid_intermediate_until.items()
            if frame_index > exp
        ]
        for tid in dead:
            self._invalid_intermediate_until.pop(tid, None)

    def _invalid_intermediate_active(self, tid: Optional[int], frame_index: int) -> bool:
        cfg = self._config
        if tid is None or not cfg.pass_invalid_intermediate_memory_enabled:
            return False
        exp = self._invalid_intermediate_until.get(tid)
        if exp is None:
            return False
        if frame_index > exp:
            self._invalid_intermediate_until.pop(tid, None)
            return False
        return True

    def _mark_invalid_intermediate_track(
        self, tid: int, frame_index: int, reason: str
    ) -> None:
        cfg = self._config
        if not cfg.pass_invalid_intermediate_memory_enabled:
            return
        exp = frame_index + cfg.pass_invalid_intermediate_ttl_frames
        prev = self._invalid_intermediate_until.get(tid)
        self._invalid_intermediate_until[tid] = max(prev or 0, exp)
        self._dbg_gate(
            f"inv_mid:{tid}:{reason}",
            frame_index,
            "[PASS-INVALID-INTERMEDIATE] mark "
            f"track={tid} until={self._invalid_intermediate_until[tid]} "
            f"reason={reason}",
        )

    def _recover_passer_last_valid_touch_simple(
        self,
        *,
        before_frame: int,
        exclude_recv: Optional[int],
        frame_index: int,
        track_to_team: dict[int, int],
        lookback_frames: int,
    ) -> Optional[tuple[int, Optional[int]]]:
        cfg = self._config
        lb = max(0, before_frame - lookback_frames)
        best: Optional[_ValidTouchEpisode] = None
        for ep in self._valid_touch_finished:
            if ep.player_id == exclude_recv:
                continue
            if ep.contact_frames < cfg.pass_valid_touch_min_frames:
                continue
            if ep.end_frame >= before_frame:
                continue
            if ep.end_frame < lb:
                continue
            if self._invalid_intermediate_active(ep.player_id, frame_index):
                continue
            if best is None or ep.end_frame > best.end_frame:
                best = ep
        for tid, eo in self._valid_touch_open.items():
            if tid == exclude_recv:
                continue
            if eo.contact_frames < cfg.pass_valid_touch_min_frames:
                continue
            if eo.end_frame >= before_frame:
                continue
            if eo.end_frame < lb:
                continue
            if self._invalid_intermediate_active(tid, frame_index):
                continue
            ep = _ValidTouchEpisode(
                player_id=tid,
                team_id=track_to_team.get(tid) or eo.team_id_snap,
                start_frame=eo.start_frame,
                end_frame=eo.end_frame,
                contact_frames=eo.contact_frames,
                min_distance_to_ball=eo.min_d,
                first_ball_xy=eo.first_ball_xy,
                last_ball_xy=eo.last_ball_xy,
            )
            if best is None or ep.end_frame > best.end_frame:
                best = ep
        if best is None:
            return None
        ft = best.team_id or track_to_team.get(best.player_id)
        return best.player_id, ft

    def _recover_passer_last_loose_touch_simple(
        self,
        *,
        before_frame: int,
        exclude_recv: Optional[int],
        frame_index: int,
        track_to_team: dict[int, int],
        lookback_frames: int,
    ) -> Optional[tuple[int, Optional[int]]]:
        cfg = self._config
        lb = max(0, before_frame - lookback_frames)
        max_d = min(
            cfg.pass_source_recover_max_distance_px,
            max(cfg.pass_source_reliability_min_radius_px, cfg.pass_touch_max_radius_px),
        )
        best_tid: Optional[int] = None
        best_frame = -1
        best_dist = float("inf")
        for snap in self._snap_hist:
            fi = snap.frame_index
            if fi < lb or fi >= before_frame or snap.ball_xy is None:
                continue
            for tid, fxy in snap.feet.items():
                if tid == exclude_recv:
                    continue
                if self._invalid_intermediate_active(tid, frame_index):
                    continue
                dd = _dist_px(fxy, snap.ball_xy)
                if dd > max_d:
                    continue
                if fi > best_frame or (fi == best_frame and dd < best_dist):
                    best_tid = tid
                    best_frame = fi
                    best_dist = dd
        if best_tid is None:
            return None
        return best_tid, track_to_team.get(best_tid)

    def _recover_recent_loose_source_excluding(
        self,
        *,
        before_frame: int,
        exclude_ids: set[int],
        frame_index: int,
        track_to_team: dict[int, int],
        lookback_frames: int,
    ) -> Optional[tuple[int, Optional[int], int, float, Optional[tuple[float, float]]]]:
        cfg = self._config
        lb = max(0, before_frame - lookback_frames)
        max_d = min(
            cfg.pass_source_recover_max_distance_px,
            max(cfg.pass_source_reliability_min_radius_px, cfg.pass_touch_max_radius_px),
        )
        best_tid: Optional[int] = None
        best_frame = -1
        best_dist = float("inf")
        best_ball: Optional[tuple[float, float]] = None
        for snap in self._snap_hist:
            fi = snap.frame_index
            if fi < lb or fi >= before_frame or snap.ball_xy is None:
                continue
            for tid, fxy in snap.feet.items():
                if tid in exclude_ids:
                    continue
                if self._invalid_intermediate_active(tid, frame_index):
                    continue
                dd = _dist_px(fxy, snap.ball_xy)
                if dd > max_d:
                    continue
                if fi > best_frame or (fi == best_frame and dd < best_dist):
                    best_tid = tid
                    best_frame = fi
                    best_dist = dd
                    best_ball = snap.ball_xy
        if best_tid is None:
            return None
        return best_tid, track_to_team.get(best_tid), best_frame, best_dist, best_ball

    def _snap_visible_ball_foot_metrics(
        self, tid: int, lo_f: int, hi_f: int
    ) -> tuple[int, Optional[float]]:
        """Count frames with foot + ball samples; minimum foot-ball distance."""

        visible_frames = 0
        min_d: Optional[float] = None
        for snap in self._snap_hist:
            fi = snap.frame_index
            if fi < lo_f or fi > hi_f:
                continue
            if snap.ball_xy is None:
                continue
            foot = snap.feet.get(tid)
            if foot is None:
                continue
            visible_frames += 1
            d = _dist_px(foot, snap.ball_xy)
            min_d = d if min_d is None else min(min_d, d)
        return visible_frames, min_d

    def _apply_none_source_recovery(
        self,
        event: PassEvent,
        frame_index: int,
        track_to_team: dict[int, int],
        cand: Optional[_FlightCandidate],
    ) -> PassEvent:
        cfg = self._config
        if not cfg.pass_none_source_recovery_enabled:
            return event
        if event.from_player_id is not None:
            return event
        if cfg.pass_none_source_recovery_skip_kickoff_guard:
            if event.source == "kickoff_bootstrap_unknown_passer":
                return event
            dr = event.debug_reason or ""
            if "first_owner_bootstrap_ambiguous" in dr:
                return event

        if event.source in ("touch_to_touch_fallback", "short_valid_touch_fallback"):
            return event

        dest = event.to_player_id
        lb0 = event.start_frame - cfg.pass_none_source_recovery_lookback_frames
        hi0 = event.start_frame + cfg.pass_none_source_recovery_lookahead_frames
        max_d = cfg.pass_none_source_recovery_max_distance_px
        min_vis = cfg.pass_none_source_recovery_min_visible_frames

        if (
            cfg.pass_none_source_recovery_use_original_from
            and event.original_from_player_id is not None
        ):
            orig = event.original_from_player_id
            if not self._invalid_intermediate_active(orig, frame_index):
                vis, md = self._snap_visible_ball_foot_metrics(orig, lb0, hi0)
                primary_ok = event.pass_fsm_state in (
                    "completed",
                    "completed_one_touch",
                )
                cond_vis = vis >= min_vis and md is not None and md <= max_d
                cond_primary = primary_ok and md is not None and md <= max_d
                if cond_vis or cond_primary:
                    nft = track_to_team.get(orig)
                    event.from_player_id = orig
                    event.from_team_id = nft
                    event.source_reliability = "recovered"
                    event.source_reliability_reason = "none_source_recovered"
                    event.source_recovery_method = "original_from"
                    event.confidence = min(event.confidence, 0.82)
                    event.event_type = self._classify(
                        nft,
                        event.to_team_id,
                        cfg.require_same_team_for_completed,
                        cfg.emit_interceptions,
                    )
                    ts = dest if dest is not None else "?"
                    self._dbg_gate(
                        f"none_rec_orig:{orig}->{dest}",
                        frame_index,
                        "[PASS-SOURCE-RECOVER] recovered "
                        f"None->{ts} as {orig}->{ts} method=original_from",
                    )
                    return event

        if cfg.pass_none_source_recovery_use_last_valid_touch:
            rc = self._recover_passer_last_valid_touch_simple(
                before_frame=event.start_frame,
                exclude_recv=dest,
                frame_index=frame_index,
                track_to_team=track_to_team,
                lookback_frames=cfg.pass_none_source_recovery_lookback_frames,
            )
            if rc is not None:
                tid, ft = rc
                event.from_player_id = tid
                event.from_team_id = ft or track_to_team.get(tid)
                event.source_reliability = "recovered"
                event.source_reliability_reason = "none_source_recovered"
                event.source_recovery_method = "last_valid_touch"
                event.confidence = min(event.confidence, 0.82)
                event.event_type = self._classify(
                    event.from_team_id,
                    event.to_team_id,
                    cfg.require_same_team_for_completed,
                    cfg.emit_interceptions,
                )
                fs = tid
                ts = dest if dest is not None else "?"
                self._dbg_gate(
                    f"none_rec_vt:{tid}->{dest}",
                    frame_index,
                    "[PASS-SOURCE-RECOVER] recovered "
                    f"None->{ts} as {fs}->{ts} method=last_valid_touch",
                )
                return event

            rc = self._recover_passer_last_loose_touch_simple(
                before_frame=event.start_frame,
                exclude_recv=dest,
                frame_index=frame_index,
                track_to_team=track_to_team,
                lookback_frames=cfg.pass_none_source_recovery_lookback_frames,
            )
            if rc is not None:
                tid, ft = rc
                event.from_player_id = tid
                event.from_team_id = ft or track_to_team.get(tid)
                event.source_reliability = "recovered"
                event.source_reliability_reason = "none_source_recovered_loose_touch"
                event.source_recovery_method = "last_loose_touch"
                event.confidence = min(event.confidence, 0.80)
                event.event_type = self._classify(
                    event.from_team_id,
                    event.to_team_id,
                    cfg.require_same_team_for_completed,
                    cfg.emit_interceptions,
                )
                ts = dest if dest is not None else "?"
                self._dbg_gate(
                    f"none_rec_loose:{tid}->{dest}",
                    frame_index,
                    "[PASS-SOURCE-RECOVER] recovered "
                    f"None->{ts} as {tid}->{ts} method=last_loose_touch",
                )

        return event

    def _primary_fsm_touch_for_source_grace(self, ev: PassEvent) -> bool:
        if ev.pass_fsm_state == "touch_to_touch_fallback":
            return False
        if ev.source == "touch_to_touch_fallback":
            return False
        return ev.pass_fsm_state in ("completed", "completed_one_touch")

    def _measure_pass_source_at_anchor(
        self, from_tid: int, anchor_frame: int
    ) -> tuple[str, Optional[str], Optional[float], Optional[float]]:
        cfg = self._config
        if not cfg.pass_source_reliability_enabled:
            return "reliable", None, None, None

        lb = anchor_frame - cfg.pass_source_reliability_lookback_frames
        if lb < 0:
            lb = 0
        hi = anchor_frame + cfg.pass_source_reliability_lookahead_frames

        min_d: Optional[float] = None
        saw_pair = False
        dyn_at_anchor: Optional[float] = None

        for snap in self._snap_hist:
            fi = snap.frame_index
            if fi < lb or fi > hi:
                continue
            if fi == anchor_frame:
                bh_a = snap.bbox_heights.get(from_tid)
                dyn_at_anchor = _dynamic_source_radius_px(cfg, bh_a)

            fpt = snap.feet.get(from_tid)
            if fpt is None or snap.ball_xy is None:
                continue
            saw_pair = True
            dd = _dist_px(fpt, snap.ball_xy)
            if min_d is None or dd < min_d:
                min_d = dd

        if dyn_at_anchor is None:
            for snap in reversed(self._snap_hist):
                if snap.frame_index > anchor_frame:
                    continue
                bh_a = snap.bbox_heights.get(from_tid)
                if bh_a is not None:
                    dyn_at_anchor = _dynamic_source_radius_px(cfg, bh_a)
                    break

        threshold = cfg.pass_source_max_release_distance_px
        if dyn_at_anchor is not None:
            threshold = float(dyn_at_anchor)

        if not saw_pair:
            return "unknown", "source_track_missing_near_release", None, dyn_at_anchor

        assert min_d is not None
        if min_d <= threshold:
            return "reliable", None, min_d, dyn_at_anchor
        return "unreliable", "source_far_from_release", min_d, dyn_at_anchor

    def _measure_pass_source_reliability(
        self, from_tid: int, cand: _FlightCandidate
    ) -> tuple[str, Optional[str], Optional[float], Optional[float]]:
        return self._measure_pass_source_at_anchor(from_tid, cand.release_frame)

    def _primary_keep_candidate_valid(
        self, snap: _PassFrameSnap, otid: int, ball_xy: tuple[float, float]
    ) -> bool:
        cfg = self._config
        fxy = snap.feet.get(otid)
        if fxy is None:
            return False
        dd = _dist_px(fxy, ball_xy)
        bh = snap.bbox_heights.get(otid)
        dyn = (
            _dynamic_touch_radius_px(cfg, bh)
            if bh is not None
            else float(cfg.pass_touch_contact_radius_px)
        )
        thr = max(cfg.pass_source_reliability_primary_keep_radius_px, float(dyn) * 1.12)
        return dd <= thr

    def _primary_keep_nearest_rank(
        self, snap: _PassFrameSnap, tid: int, ball_xy: tuple[float, float]
    ) -> Optional[int]:
        if tid not in snap.feet:
            return None
        pool: list[tuple[int, float]] = []
        for otid in snap.feet:
            if not self._primary_keep_candidate_valid(snap, otid, ball_xy):
                continue
            dd = _dist_px(snap.feet[otid], ball_xy)
            pool.append((otid, dd))
        pool.sort(key=lambda z: z[1])
        order = [t for t, _ in pool]
        if tid not in order:
            return None
        return order.index(tid) + 1

    def _visible_primary_source_keep(
        self, from_tid: int, anchor_frame: int
    ) -> tuple[bool, float, int, Optional[int]]:
        cfg = self._config
        lb = max(0, anchor_frame - cfg.pass_source_reliability_primary_keep_lookback_frames)
        hi = anchor_frame + cfg.pass_source_reliability_primary_keep_lookahead_frames
        min_d = float("inf")
        visible_frames = 0
        best_rank: Optional[int] = None
        rk_anchor: Optional[int] = None

        for snap in self._snap_hist:
            fi = snap.frame_index
            if fi < lb or fi > hi:
                continue
            if snap.ball_xy is None:
                continue
            foot = snap.feet.get(from_tid)
            if foot is None:
                continue
            visible_frames += 1
            dd = _dist_px(foot, snap.ball_xy)
            min_d = min(min_d, dd)
            rk = self._primary_keep_nearest_rank(snap, from_tid, snap.ball_xy)
            if rk is not None:
                best_rank = rk if best_rank is None else min(best_rank, rk)

        for snap in self._snap_hist:
            if snap.frame_index != anchor_frame or snap.ball_xy is None:
                continue
            if from_tid not in snap.feet:
                break
            rk_anchor = self._primary_keep_nearest_rank(
                snap, from_tid, snap.ball_xy
            )
            break

        md_out = min_d if min_d < float("inf") else -1.0
        if visible_frames < cfg.pass_source_reliability_primary_keep_min_visible_frames:
            return False, md_out, visible_frames, best_rank

        if md_out <= cfg.pass_source_reliability_primary_keep_radius_px:
            return True, md_out, visible_frames, best_rank

        nmax = cfg.pass_source_reliability_primary_keep_nearest_rank
        if best_rank is not None and best_rank <= nmax:
            return True, md_out, visible_frames, best_rank

        if rk_anchor == 1 and visible_frames >= max(
            cfg.pass_source_reliability_primary_keep_min_visible_frames, 1
        ):
            return True, md_out, visible_frames, best_rank

        return False, md_out, visible_frames, best_rank

    def _same_team_retarget_after_source_recovery(
        self,
        *,
        event: PassEvent,
        cand: _FlightCandidate,
        receiver_id: int,
        frame_index: int,
        tracks: Optional["FrameTracks"],
        ball_xy: Optional[tuple[float, float]],
        track_to_team: dict[int, int],
    ) -> Optional[int]:
        if tracks is None or ball_xy is None:
            return None
        if event.source_recovery_method != "recent_valid_touch_source_replace":
            return None
        if event.event_type != "intercepted_pass":
            return None
        if event.from_player_id is None or event.from_team_id is None:
            return None
        if event.to_team_id is None or event.from_team_id == event.to_team_id:
            return None

        cfg = self._config
        from VisionEngine.schemas.schema import ObjectRole

        search_r = float(cfg.pass_receiver_control_radius_px) * 1.45 + 40.0
        best_tid: Optional[int] = None
        best_score = float("-inf")
        best_d = float("inf")
        receiver_d: Optional[float] = None

        for inst in tracks.instances:
            if inst.role is not ObjectRole.PLAYER:
                continue
            tid = inst.track_id
            if tid == receiver_id:
                bbox = tuple(float(x) for x in inst.xyxy)
                receiver_d = _dist_px(foot_xy(bbox), ball_xy)
                continue
            if tid < 0 or tid in (event.from_player_id, receiver_id):
                continue
            if track_to_team.get(tid) != event.from_team_id:
                continue
            if (
                cfg.pass_invalid_intermediate_memory_enabled
                and self._invalid_intermediate_active(tid, frame_index)
            ):
                continue
            bbox = tuple(float(x) for x in inst.xyxy)
            ft = foot_xy(bbox)
            d_near = _dist_px(ft, ball_xy)
            if d_near > search_r:
                continue
            d_foot, lower_ok, valid_ctrl, bbox_only_flag = classify_touch_features(
                cfg, bbox, ball_xy
            )
            if bbox_only_flag and not (lower_ok or valid_ctrl):
                continue
            tm = track_to_team.get(tid)
            stab = tm is None or self._team_label_stable_for_track(tid, tm)
            sc = self._score_receiver_candidate_one_frame(
                track_id=tid,
                bbox=bbox,
                ball_xy=ball_xy,
                exclude=event.from_player_id,
                release_anchor=cand.ball_anchor_xy,
                ft=ft,
                d_foot=d_foot,
                lower_body_valid=lower_ok,
                bbox_overlap_only=bbox_only_flag,
                team_id=tm,
                track_to_team=track_to_team,
                team_stable=stab,
            )
            if sc > best_score:
                best_tid = tid
                best_score = sc
                best_d = d_near

        if best_tid is None or best_score < 0.35:
            return None
        tight_receiver_d = max(40.0, cfg.pass_receiver_control_radius_px * 0.65)
        if (
            receiver_d is not None
            and receiver_d <= tight_receiver_d
            and best_d + 15.0 >= receiver_d
        ):
            return None

        old_receiver = receiver_id
        event.to_player_id = best_tid
        event.to_team_id = event.from_team_id
        event.raw_to_team_id = event.from_team_id
        event.receiver_bbox_intermediate_raw = old_receiver
        event.receiver_selection_reason = "same_team_retarget_after_source_recover"
        event.receiver_score = round(best_score, 6)
        event.receiver_contact_type = self._receiver_contact_label_for_emit(
            best_tid, cand, frame_index
        )
        event.bbox_only_guard_status = "source_recovery_same_team_retargeted"
        event.bbox_guard_checked = True
        event.bbox_guard_result = "retargeted"
        event.event_type = self._classify(
            event.from_team_id,
            event.to_team_id,
            cfg.require_same_team_for_completed,
            cfg.emit_interceptions,
        )
        event.confidence = min(event.confidence, 0.82)
        reason = f"source_recovered_same_team_receiver:{old_receiver}->{best_tid}"
        event.debug_reason = (
            reason if event.debug_reason is None else f"{event.debug_reason};{reason}"
        )
        self._dbg_gate(
            f"source_recover_recv_rt:{event.from_player_id}:{old_receiver}->{best_tid}",
            frame_index,
            "[PASS-RECEIVER-RETARGET] "
            f"{event.from_player_id}->{old_receiver} retargeted_to={best_tid} "
            f"reason=source_recovered_same_team score={best_score:.2f} d={best_d:.1f}",
        )
        return best_tid

    def _short_interception_duel_should_suppress(self, event: PassEvent) -> bool:
        cfg = self._config
        if event.event_type != "intercepted_pass":
            return False
        if event.from_team_id is None or event.to_team_id is None:
            return False
        if event.from_team_id == event.to_team_id:
            return False
        if event.start_frame <= cfg.pass_kickoff_bootstrap_guard_max_frame:
            return False
        if event.duration_frames > 8:
            return False
        disp = event.ball_displacement_px
        if disp is None or disp > 42.0:
            return False
        if event.receiver_contact_type not in ("lower_body", "lower_body_sustained"):
            return False
        return True

    def _recent_source_valid_touch_for_keep(
        self, from_tid: int, release_frame: int
    ) -> Optional[_ValidTouchEpisode]:
        cfg = self._config
        lookback = max(
            cfg.pass_source_recover_lookback_frames,
            cfg.pass_touch_history_frames,
        )
        lb = max(0, release_frame - lookback)
        best: Optional[_ValidTouchEpisode] = None

        def maybe_keep(ep: _ValidTouchEpisode) -> None:
            nonlocal best
            if ep.player_id != from_tid:
                return
            if ep.contact_frames < cfg.pass_valid_touch_min_frames:
                return
            if ep.end_frame >= release_frame or ep.end_frame < lb:
                return
            if best is None or (ep.end_frame, ep.contact_frames) > (
                best.end_frame,
                best.contact_frames,
            ):
                best = ep

        for ep in self._valid_touch_finished:
            maybe_keep(ep)

        eo = self._valid_touch_open.get(from_tid)
        if eo is not None:
            maybe_keep(
                _ValidTouchEpisode(
                    player_id=from_tid,
                    team_id=eo.team_id_snap,
                    start_frame=eo.start_frame,
                    end_frame=eo.end_frame,
                    contact_frames=eo.contact_frames,
                    min_distance_to_ball=eo.min_d,
                    first_ball_xy=eo.first_ball_xy,
                    last_ball_xy=eo.last_ball_xy,
                )
            )

        return best

    def _apply_pass_source_reliability_touch(
        self, event: PassEvent, anchor_frame: int, from_tid: int
    ) -> PassEvent:
        cfg = self._config
        if not cfg.pass_source_reliability_enabled:
            return event
        tid = event.from_player_id if event.from_player_id is not None else from_tid
        if tid is None:
            return event

        rel, reason, min_d, rad_use = self._measure_pass_source_at_anchor(tid, anchor_frame)
        event.source_reliability = rel
        event.source_reliability_reason = reason
        event.source_release_distance_px = round(min_d, 4) if min_d is not None else None
        event.source_release_radius_px = round(rad_use, 4) if rad_use is not None else None

        dist_s = "?" if min_d is None else f"{min_d:.1f}"
        rad_s = "?" if rad_use is None else f"{rad_use:.1f}"
        if cfg.pass_debug and rel == "reliable":
            self._dbg(
                f"[PASS-SOURCE] reliable primary_fsm from={tid} "
                f"dist={dist_s} radius={rad_s}"
            )
        elif cfg.pass_debug and rel in ("unreliable", "unknown"):
            self._dbg(
                f"[PASS-SOURCE] {rel} from={tid} dist={dist_s} radius={rad_s}"
            )

        downgrade = cfg.pass_source_unknown_if_unreliable and rel in (
            "unreliable",
            "unknown",
        )
        grace_ok = False
        if (
            downgrade
            and cfg.pass_source_reliability_primary_fsm_grace_enabled
            and cfg.pass_source_reliability_keep_primary_fsm_source
            and self._primary_fsm_touch_for_source_grace(event)
            and reason == "source_far_from_release"
            and min_d is not None
            and min_d <= cfg.pass_source_max_release_distance_px
        ):
            grace_ok = True
            downgrade = False

        if grace_ok:
            event.source_reliability = "reliable"
            event.source_reliability_reason = "primary_fsm_recent_touch_legacy_radius_cap"
            if cfg.pass_debug:
                self._dbg(
                    "[PASS-SOURCE] keep source="
                    f"{tid} reason=primary_fsm_recent_touch dist={dist_s}"
                )

        if not downgrade:
            return event

        old_from = event.from_player_id
        old_team = event.from_team_id
        if event.original_from_player_id is None:
            event.original_from_player_id = old_from
        event.raw_from_team_id = old_team

        event.from_player_id = None
        event.from_team_id = None
        event.event_type = "unknown_pass"
        event.confidence = min(event.confidence, 0.70)
        dr = reason or rel or "source_unverified"
        if event.debug_reason:
            event.debug_reason = f"{event.debug_reason};{dr}"
        else:
            event.debug_reason = dr

        return event

    def _apply_pass_source_reliability(
        self,
        event: PassEvent,
        cand: _FlightCandidate,
        *,
        track_to_team: Optional[dict[int, int]] = None,
    ) -> PassEvent:
        cfg = self._config
        if not cfg.pass_source_reliability_enabled:
            return event

        tid = event.from_player_id
        if tid is None:
            return event

        rel, reason, min_d, rad_use = self._measure_pass_source_reliability(tid, cand)
        event.source_reliability = rel
        event.source_reliability_reason = reason
        event.source_release_distance_px = round(min_d, 4) if min_d is not None else None
        event.source_release_radius_px = round(rad_use, 4) if rad_use is not None else None

        dist_s = "?" if min_d is None else f"{min_d:.1f}"
        rad_s = "?" if rad_use is None else f"{rad_use:.1f}"
        if cfg.pass_debug and rel == "reliable":
            self._dbg(
                f"[PASS-SOURCE] reliable primary_fsm from={tid} "
                f"dist={dist_s} radius={rad_s}"
            )
        elif cfg.pass_debug and rel in ("unreliable", "unknown"):
            self._dbg(
                f"[PASS-SOURCE] {rel} from={tid} dist={dist_s} radius={rad_s}"
            )

        downgrade = cfg.pass_source_unknown_if_unreliable and rel in (
            "unreliable",
            "unknown",
        )
        grace_ok = False
        if (
            downgrade
            and cfg.pass_source_reliability_primary_fsm_grace_enabled
            and cfg.pass_source_reliability_keep_primary_fsm_source
            and self._primary_fsm_touch_for_source_grace(event)
            and reason == "source_far_from_release"
            and min_d is not None
            and min_d <= cfg.pass_source_max_release_distance_px
        ):
            grace_ok = True
            downgrade = False

        if grace_ok:
            event.source_reliability = "reliable"
            event.source_reliability_reason = "primary_fsm_recent_touch_legacy_radius_cap"
            if cfg.pass_debug:
                self._dbg(
                    "[PASS-SOURCE] keep source="
                    f"{tid} reason=primary_fsm_recent_touch dist={dist_s}"
                )

        if (
            downgrade
            and cfg.pass_source_reliability_keep_visible_primary_source
            and self._primary_fsm_touch_for_source_grace(event)
        ):
            vk, md_v, vf, br = self._visible_primary_source_keep(tid, cand.release_frame)
            if vk:
                downgrade = False
                event.source_reliability = "reliable"
                event.source_reliability_reason = "visible_primary_source_kept"
                if cfg.pass_debug:
                    brs = "?" if br is None else str(br)
                    self._dbg(
                        f"[PASS-SOURCE] keep visible primary source={tid} "
                        f"reason=visible_primary_source_kept dist={md_v:.1f} "
                        f"visible_frames={vf} best_rank={brs}"
                    )

        if not downgrade:
            return event

        if (
            track_to_team is not None
            and self._try_recover_passer_via_valid_touch(
                event, cand, tid, track_to_team
            )
        ):
            return event

        if (
            event.pass_fsm_state == "completed"
            and not event.one_touch
            and reason == "source_far_from_release"
        ):
            last_ev = self._pass_emission_log[-1] if self._pass_emission_log else None
            from_recovered_interception = (
                last_ev is not None
                and last_ev.to_player_id == tid
                and last_ev.event_type == "intercepted_pass"
                and last_ev.source_recovery_method == "recent_valid_touch_source_replace"
            )
            keep_ep = None
            if not from_recovered_interception:
                keep_ep = self._recent_source_valid_touch_for_keep(
                    tid, cand.release_frame
                )
            if keep_ep is not None:
                event.source_reliability = "reliable"
                event.source_reliability_reason = (
                    "primary_owner_recent_valid_touch_kept"
                )
                event.confidence = min(event.confidence, 0.88)
                if cfg.pass_debug:
                    self._dbg_gate(
                        f"source_keep_recent_vt:{tid}:{cand.release_frame}",
                        cand.release_frame,
                        "[PASS-SOURCE] keep source="
                        f"{tid} reason=primary_owner_recent_valid_touch "
                        f"touch_end={keep_ep.end_frame} "
                        f"contact_frames={keep_ep.contact_frames}",
                    )
                return event

        mark_reason = reason or rel or "source_unverified"
        if cfg.pass_debug:
            self._dbg(
                f"[PASS-SOURCE] mark unknown old={tid} reason={mark_reason}"
            )

        old_from = event.from_player_id
        old_team = event.from_team_id
        if event.original_from_player_id is None:
            event.original_from_player_id = old_from
        event.raw_from_team_id = old_team

        event.from_player_id = None
        event.from_team_id = None
        event.event_type = "unknown_pass"
        event.confidence = min(event.confidence, 0.70)
        dr = mark_reason
        if event.debug_reason:
            event.debug_reason = f"{event.debug_reason};{dr}"
        else:
            event.debug_reason = dr

        return event

    def _finalize_emitted_pass(self, event: PassEvent) -> PassEvent:
        cfg = self._config
        is_first = not self._emitted_any_pass
        self._emitted_any_pass = True

        ft, tt = event.from_team_id, event.to_team_id
        qualifies = (
            cfg.pass_kickoff_bootstrap_guard_enabled
            and is_first
            and event.event_type == "intercepted_pass"
            and event.from_player_id is not None
            and event.to_player_id is not None
            and event.start_frame <= cfg.pass_kickoff_bootstrap_guard_max_frame
            and ft is not None
            and tt is not None
            and ft != tt
        )
        if not qualifies:
            return event

        if self._recover_kickoff_loose_source(event):
            return event

        old_from_p = event.from_player_id
        old_from_team = ft
        old_to_team = tt

        event.original_from_player_id = old_from_p
        event.raw_from_team_id = old_from_team
        event.raw_to_team_id = old_to_team

        event.from_player_id = None
        event.from_team_id = None
        event.event_type = "unknown_pass"
        event.confidence = min(event.confidence, 0.70)
        event.source = "kickoff_bootstrap_unknown_passer"
        event.debug_reason = "first_owner_bootstrap_ambiguous"
        event.passer_detection_status = "missing_or_unreliable"

        receiver = event.to_player_id
        if cfg.pass_debug:
            self._dbg(
                "[PASS-KICKOFF-GUARD] first event passer marked unknown: "
                f"original_from={old_from_p} receiver={receiver} "
                "reason=first_owner_bootstrap_ambiguous"
            )
            print(format_pass_terminal_for_debug(event), flush=True)
            event.suppress_duplicate_pass_terminal = True
        else:
            event.suppress_duplicate_pass_terminal = False

        return event

    def _recover_kickoff_loose_source(self, event: PassEvent) -> bool:
        cfg = self._config
        to_team = event.to_team_id
        recv = event.to_player_id
        if to_team is None or recv is None:
            return False

        lb = max(0, event.start_frame - 24)
        hi = event.start_frame
        best_tid: Optional[int] = None
        best_frame = -1
        best_dist = float("inf")
        for snap in self._snap_hist:
            fi = snap.frame_index
            if fi < lb or fi > hi or snap.ball_xy is None:
                continue
            for tid, fxy in snap.feet.items():
                if tid == recv:
                    continue
                tm = self._team_history_mode_for_track(tid)
                if tm != to_team:
                    continue
                dd = _dist_px(fxy, snap.ball_xy)
                if dd > 55.0:
                    continue
                if dd < best_dist or (abs(dd - best_dist) < 1e-6 and fi > best_frame):
                    best_tid = tid
                    best_frame = fi
                    best_dist = dd

        if best_tid is None:
            return False
        old_from = event.from_player_id
        old_team = event.from_team_id
        event.original_from_player_id = old_from
        event.from_player_id = best_tid
        event.from_team_id = to_team
        event.raw_from_team_id = to_team
        event.raw_to_team_id = to_team
        event.event_type = self._classify(
            to_team,
            to_team,
            cfg.require_same_team_for_completed,
            cfg.emit_interceptions,
        )
        event.confidence = min(max(event.confidence, 0.78), 0.82)
        event.source = "kickoff_loose_source_recovered"
        event.debug_reason = (
            "kickoff_loose_source_recovered"
            f":old_from={old_from}:old_team={old_team}:frame={best_frame}:dist={best_dist:.1f}"
        )
        event.passer_detection_status = "recovered"
        event.source_recovery_method = "kickoff_loose_source"
        if cfg.pass_debug:
            self._dbg_gate(
                f"kickoff_loose_source:{best_tid}->{recv}",
                event.end_frame,
                "[PASS-KICKOFF-GUARD] recovered first passer "
                f"{best_tid}->{recv} frame={best_frame} dist={best_dist:.1f}",
            )
        return True

    def _compute_pass_confidence(
        self,
        *,
        baseline_evt_type: str,
        evt_type: str,
        duration_frames: int,
        raw_from: Optional[int],
        raw_to: Optional[int],
        from_stable: bool,
        to_stable: bool,
        ball_start: Optional[tuple[float, float]],
        ball_end: Optional[tuple[float, float]],
    ) -> float:
        from_ok_for_bonus = raw_from is None or from_stable
        to_ok_for_bonus = raw_to is None or to_stable
        has_team_signal = raw_from is not None or raw_to is not None

        stability_bonus = (
            0.15 if has_team_signal and from_ok_for_bonus and to_ok_for_bonus else 0.0
        )
        ball_bonus = 0.10 if ball_start is not None and ball_end is not None else 0.0
        duration_bonus = 0.10 if duration_frames >= 4 else 0.0
        same_team_bonus = (
            0.10
            if (
                baseline_evt_type == "completed_pass"
                and raw_from is not None
                and raw_to is not None
                and raw_from == raw_to
            )
            else 0.0
        )

        conf = min(
            0.95,
            0.55 + stability_bonus + ball_bonus + duration_bonus + same_team_bonus,
        )

        both_teams_id_stable_for_intercept = (
            raw_from is not None
            and raw_to is not None
            and from_stable
            and to_stable
        )
        if evt_type == "unknown_pass":
            conf = min(conf, 0.75)
        elif evt_type == "intercepted_pass":
            if not both_teams_id_stable_for_intercept:
                conf = min(conf, 0.80)
        if duration_frames <= 3:
            conf = min(conf, 0.75)
        return max(0.0, conf)

    def _classify(
        self,
        from_team: Optional[int],
        to_team: Optional[int],
        require_same_completed: bool,
        emit_intercept: bool,
    ) -> str:
        both_known = from_team is not None and to_team is not None
        if both_known:
            if from_team == to_team:
                return "completed_pass"
            return "intercepted_pass" if emit_intercept else "unknown_pass"

        if require_same_completed:
            return "unknown_pass"

        if from_team is not None and to_team is None:
            return "completed_pass"
        return "unknown_pass"

    def _update_controlled_owner(
        self, possession_tid: Optional[int], *, frame_index: int
    ) -> Optional[tuple[int, int]]:
        cfg = self._config
        tid = possession_tid

        if tid is None:
            self._run_tid = None
            self._same_run = 0
            self._alt_tid = None
            self._alt_run = 0
            return None

        if tid == self._run_tid:
            self._same_run += 1
        else:
            self._run_tid = tid
            self._same_run = 1

        if self._co_owner is None:
            if self._same_run >= cfg.pass_owner_min_stable_frames:
                self._co_owner = tid
                self._dbg(
                    f"[PASS-FSM] stable_owner={tid} frames={self._same_run} "
                    f"@ frame={frame_index}"
                )
            return None

        if tid == self._co_owner:
            self._alt_tid = None
            self._alt_run = 0
            return None

        if tid != self._alt_tid:
            self._alt_tid = tid
            self._alt_run = 1
        else:
            self._alt_run += 1

        transient = cfg.pass_transient_contact_max_frames
        need_switch = max(
            cfg.pass_owner_switch_min_frames, cfg.pass_owner_min_stable_frames
        )

        if self._alt_run <= transient:
            return None

        if self._alt_run >= need_switch:
            prev = self._co_owner
            self._co_owner = tid
            self._dbg(
                "[PASS-FSM] controlled_owner_switch "
                f"{prev}->{tid} frames={self._alt_run} @ frame={frame_index}"
            )
            self._alt_tid = None
            self._alt_run = 0
            if prev is not None:
                self._last_owner_switch = (prev, tid, frame_index)
                return (prev, tid)
        return None

    def _try_emit_owner_switch_handoff(
        self,
        *,
        prev_owner: int,
        new_owner: int,
        frame_index: int,
        timestamp_sec: Optional[float],
        possession: PossessionState,
        track_to_team: dict[int, int],
        tracks: Optional["FrameTracks"],
    ) -> Optional[PassEvent]:
        cfg = self._config
        if prev_owner != self._last_receiver_id:
            return None
        last_ev = self._pass_emission_log[-1] if self._pass_emission_log else None
        if (
            last_ev is None
            or last_ev.to_player_id != prev_owner
            or not self._event_receiver_is_weak_intermediate_candidate(last_ev)
        ):
            return None
        if new_owner == self._last_event_from_player_id:
            return None
        if self._last_event_end_frame is None:
            return None
        handoff_window = max(24, cfg.pass_post_receive_settle_frames * 2)
        if frame_index - self._last_event_end_frame > handoff_window:
            return None
        if possession.ball_xy is None:
            return None

        anchor = self._away_anchor
        if anchor is None:
            anchor = tuple(possession.ball_xy)
        start_frame = max(0, frame_index - max(1, self._away_run) + 1)
        cand = _FlightCandidate(
            start_frame=start_frame,
            start_time_sec=timestamp_sec,
            from_player_id=prev_owner,
            from_team_id=track_to_team.get(prev_owner),
            ball_anchor_xy=tuple(anchor),
            release_frame=start_frame,
        )
        flight_len = max(1, frame_index - start_frame + 1)
        self._dbg_gate(
            f"owner_switch_handoff:{prev_owner}->{new_owner}:{frame_index}",
            frame_index,
            "[PASS-FSM] owner_switch_handoff "
            f"{prev_owner}->{new_owner} frames={flight_len}",
        )
        ev = self._emit_completed(
            cand=cand,
            receiver_id=new_owner,
            flight_len=flight_len,
            frame_index=frame_index,
            timestamp_sec=timestamp_sec,
            possession=possession,
            track_to_team=track_to_team,
            tracks=tracks,
            ball_xy_for_guard=(
                tuple(possession.ball_xy) if possession.ball_xy is not None else None
            ),
            receiver_selection_reason="owner_switch_handoff",
            receiver_score=None,
        )
        self._away_run = 0
        self._away_anchor = None
        if ev is not None:
            self._last_owner_switch = None
        return ev

    def _make_direct_fallback_event(
        self,
        *,
        from_player_id: int,
        to_player_id: int,
        from_team_id: Optional[int],
        to_team_id: Optional[int],
        start_frame: int,
        end_frame: int,
        timestamp_sec: Optional[float],
        ball_start_xy: Optional[tuple[float, float]],
        ball_end_xy: Optional[tuple[float, float]],
        confidence: float,
        pass_fsm_state: str,
        receiver_selection_reason: str,
        receiver_contact_type: str,
        debug_reason: str,
        possession: PossessionState,
        track_to_team: dict[int, int],
        tracks: Optional["FrameTracks"],
        emit_path_base: str,
        receiver_confirm_start_frame: Optional[int] = None,
        receiver_confirm_frames: Optional[int] = None,
    ) -> PassEvent:
        cfg = self._config
        evt_type = self._classify(
            from_team_id,
            to_team_id,
            cfg.require_same_team_for_completed,
            cfg.emit_interceptions,
        )
        team_stable = (
            (from_team_id is None or self._team_label_stable_for_track(from_player_id, from_team_id))
            and (to_team_id is None or self._team_label_stable_for_track(to_player_id, to_team_id))
        )
        disp = (
            _dist_px(ball_start_xy, ball_end_xy)
            if ball_start_xy is not None and ball_end_xy is not None
            else None
        )
        event = PassEvent(
            event_id=self._next_event_id,
            event_type=evt_type,
            start_frame=start_frame,
            end_frame=end_frame,
            start_time_sec=None,
            end_time_sec=timestamp_sec,
            from_player_id=from_player_id,
            to_player_id=to_player_id,
            from_team_id=from_team_id,
            to_team_id=to_team_id,
            duration_frames=max(1, end_frame - start_frame + 1),
            duration_sec=None,
            confidence=confidence,
            ball_start_xy=ball_start_xy,
            ball_end_xy=ball_end_xy,
            raw_from_team_id=from_team_id,
            raw_to_team_id=to_team_id,
            team_stable=team_stable,
            debug_reason=debug_reason,
            release_frame=start_frame,
            receiver_confirm_start_frame=(
                receiver_confirm_start_frame
                if receiver_confirm_start_frame is not None
                else end_frame
            ),
            receiver_confirm_frames=receiver_confirm_frames,
            ball_displacement_px=disp,
            pass_fsm_state=pass_fsm_state,
            source=emit_path_base,
            source_reliability="recovered",
            source_reliability_reason=debug_reason,
            receiver_selection_reason=receiver_selection_reason,
            dribble_guard_status="passed_guard",
            receiver_contact_type=receiver_contact_type,
            bbox_only_guard_status="direct_fallback",
            emit_path=emit_path_base,
            receiver_contact_decision=receiver_selection_reason,
        )
        self._next_event_id += 1
        self._apply_pass_event_aliases(event, end_frame, track_to_team=track_to_team)
        bx = ball_end_xy
        if bx is None and possession.ball_xy is not None:
            bx = (float(possession.ball_xy[0]), float(possession.ball_xy[1]))
        self._attach_pass_decision_telemetry(
            event,
            tracks=tracks,
            ball_xy=bx,
            track_to_team=track_to_team,
            frame_index=end_frame,
            emit_path_base=emit_path_base,
            flight_start_frame=start_frame,
            flight_end_frame=end_frame,
            release_anchor_xy=ball_start_xy,
        )
        self._remember_pass_emission(event)
        return event

    def _try_emit_owner_switch_proximity_fallback(
        self,
        *,
        prev_owner: int,
        new_owner: int,
        frame_index: int,
        timestamp_sec: Optional[float],
        possession: PossessionState,
        track_to_team: dict[int, int],
        tracks: Optional["FrameTracks"],
    ) -> Optional[PassEvent]:
        if tracks is None or possession.ball_xy is None or self._away_anchor is None:
            return None
        last_ev = self._pass_emission_log[-1] if self._pass_emission_log else None
        if (
            last_ev is None
            or last_ev.to_player_id != prev_owner
            or last_ev.from_player_id is None
            or prev_owner != self._last_receiver_id
        ):
            return None
        if last_ev.pass_fsm_state != "completed" or last_ev.one_touch:
            return None
        last_contact = (last_ev.receiver_contact_type or "").lower()
        if last_contact not in ("unknown", "bbox_overlap_skim") and "bbox" not in last_contact:
            return None
        if frame_index - last_ev.end_frame > 70:
            return None

        cfg = self._config
        from VisionEngine.schemas.schema import ObjectRole

        ball_xy = (float(possession.ball_xy[0]), float(possession.ball_xy[1]))
        new_owner_d: Optional[float] = None
        best_tid: Optional[int] = None
        best_d = float("inf")
        best_score = float("-inf")
        search_r = float(cfg.pass_receiver_control_radius_px) * 1.45 + 40.0

        for inst in tracks.instances:
            if inst.role is not ObjectRole.PLAYER or inst.track_id < 0:
                continue
            tid = inst.track_id
            bbox = tuple(float(x) for x in inst.xyxy)
            d = _dist_px(foot_xy(bbox), ball_xy)
            if tid == new_owner:
                new_owner_d = d
            if tid in (prev_owner, new_owner):
                continue
            if d > search_r:
                continue
            tm = track_to_team.get(tid)
            prev_team = track_to_team.get(prev_owner)
            if prev_team is not None and tm is not None and tm != prev_team:
                continue
            d_foot, lower_ok, valid_ctrl, bbox_only_flag = classify_touch_features(
                cfg, bbox, ball_xy
            )
            if bbox_only_flag and not (lower_ok or valid_ctrl):
                continue
            stab = tm is None or self._team_label_stable_for_track(tid, tm)
            sc = self._score_receiver_candidate_one_frame(
                track_id=tid,
                bbox=bbox,
                ball_xy=ball_xy,
                exclude=prev_owner,
                release_anchor=self._away_anchor,
                ft=foot_xy(bbox),
                d_foot=d_foot,
                lower_body_valid=lower_ok,
                bbox_overlap_only=bbox_only_flag,
                team_id=tm,
                track_to_team=track_to_team,
                team_stable=stab,
            )
            if d < best_d:
                best_tid = tid
                best_d = d
                best_score = sc

        if best_tid is None:
            return None
        if best_d > 60.0:
            return None
        if best_score < 0.35:
            return None
        if new_owner_d is not None and best_d + 20.0 >= new_owner_d:
            return None

        from_team = track_to_team.get(prev_owner)
        to_team = track_to_team.get(best_tid)
        if to_team is None:
            to_team = from_team
        start_frame = max(0, frame_index - max(1, self._away_run) + 1)
        event = self._make_direct_fallback_event(
            from_player_id=prev_owner,
            to_player_id=best_tid,
            from_team_id=from_team,
            to_team_id=to_team,
            start_frame=start_frame,
            end_frame=frame_index,
            timestamp_sec=timestamp_sec,
            ball_start_xy=tuple(self._away_anchor),
            ball_end_xy=ball_xy,
            confidence=0.76,
            pass_fsm_state="owner_switch_proximity_fallback",
            receiver_selection_reason="owner_switch_proximity_receiver",
            receiver_contact_type="proximity_receiver",
            debug_reason="owner_switch_proximity_receiver",
            possession=possession,
            track_to_team=track_to_team,
            tracks=tracks,
            emit_path_base="owner_switch_proximity_fallback",
            receiver_confirm_start_frame=frame_index,
            receiver_confirm_frames=1,
        )
        self._dbg_gate(
            f"owner_switch_prox:{prev_owner}->{best_tid}:{frame_index}",
            frame_index,
            "[PASS-FSM] owner_switch_proximity_fallback "
            f"{prev_owner}->{best_tid} over_owner={new_owner} "
            f"d={best_d:.1f} score={best_score:.2f}",
        )

        if (
            last_ev.from_player_id is not None
            and last_ev.from_player_id != best_tid
            and last_ev.from_player_id != prev_owner
        ):
            self._pending_return_pass = _PendingReturnPass(
                from_player_id=best_tid,
                to_player_id=last_ev.from_player_id,
                from_team_id=to_team,
                to_team_id=last_ev.from_team_id,
                start_frame=frame_index + 1,
                release_anchor_xy=ball_xy,
                expires_frame=frame_index + 70,
            )
        self._away_run = 0
        self._away_anchor = None
        return event

    def _try_emit_pending_return_pass(
        self,
        *,
        frame_index: int,
        timestamp_sec: Optional[float],
        possession: PossessionState,
        track_to_team: dict[int, int],
        tracks: Optional["FrameTracks"],
    ) -> Optional[PassEvent]:
        pending = self._pending_return_pass
        if pending is None:
            return None
        if frame_index > pending.expires_frame:
            self._pending_return_pass = None
            return None
        if tracks is None or possession.ball_xy is None:
            return None

        from VisionEngine.schemas.schema import ObjectRole

        ball_xy = (float(possession.ball_xy[0]), float(possession.ball_xy[1]))
        target_bbox: Optional[tuple[float, float, float, float]] = None
        for inst in tracks.instances:
            if (
                inst.role is ObjectRole.PLAYER
                and inst.track_id == pending.to_player_id
            ):
                target_bbox = tuple(float(x) for x in inst.xyxy)
                break
        if target_bbox is None:
            pending.streak = 0
            pending.first_candidate_frame = None
            return None

        target_foot = foot_xy(target_bbox)
        target_d = _dist_px(target_foot, ball_xy)
        disp = _dist_px(pending.release_anchor_xy, ball_xy)
        v1 = (
            ball_xy[0] - pending.release_anchor_xy[0],
            ball_xy[1] - pending.release_anchor_xy[1],
        )
        v2 = (
            target_foot[0] - pending.release_anchor_xy[0],
            target_foot[1] - pending.release_anchor_xy[1],
        )
        n1 = math.hypot(v1[0], v1[1])
        n2 = math.hypot(v2[0], v2[1])
        align = 0.0
        if n1 > 1e-3 and n2 > 1e-3:
            align = (v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2)

        if target_d <= 110.0 and disp >= 80.0 and align >= 0.75:
            pending.streak += 1
            if pending.first_candidate_frame is None:
                pending.first_candidate_frame = frame_index
        else:
            pending.streak = 0
            pending.first_candidate_frame = None
            return None

        if pending.streak < 3:
            return None

        event = self._make_direct_fallback_event(
            from_player_id=pending.from_player_id,
            to_player_id=pending.to_player_id,
            from_team_id=pending.from_team_id,
            to_team_id=pending.to_team_id,
            start_frame=pending.start_frame,
            end_frame=frame_index,
            timestamp_sec=timestamp_sec,
            ball_start_xy=pending.release_anchor_xy,
            ball_end_xy=ball_xy,
            confidence=0.74,
            pass_fsm_state="pending_return_fallback",
            receiver_selection_reason="give_and_go_return_target",
            receiver_contact_type="proximity_return",
            debug_reason="pending_return_pass",
            possession=possession,
            track_to_team=track_to_team,
            tracks=tracks,
            emit_path_base="pending_return_fallback",
            receiver_confirm_start_frame=pending.first_candidate_frame,
            receiver_confirm_frames=pending.streak,
        )
        self._dbg_gate(
            f"pending_return:{pending.from_player_id}->{pending.to_player_id}:{frame_index}",
            frame_index,
            "[PASS-FSM] pending_return_fallback "
            f"{pending.from_player_id}->{pending.to_player_id} "
            f"d={target_d:.1f} disp={disp:.1f} align={align:.2f}",
        )
        self._pending_return_pass = None
        return event

    def _clear_flight(self) -> None:
        self._flight = None
        self._recv_tid = None
        self._recv_streak = 0
        self._recv_gap = 0
        self._recv_confirm_start_frame = None
        self._flight_recv_pick_reason = ""
        self._flight_recv_pick_score = -1.0
        self._intr_tid = None
        self._intr_run = 0
        self._reclaim_run = 0
        self._reset_flight_contacts()

    def _reset_flight_contacts(self) -> None:
        self._flight_contact_open.clear()
        self._flight_contact_finished.clear()

    def _flight_contact_open_to_episode(
        self, o: _FlightContactOpenEpisode
    ) -> _FlightContactEpisode:
        return _FlightContactEpisode(
            player_id=o.player_id,
            team_id=o.team_id_snap,
            start_frame=o.start_frame,
            end_frame=o.end_frame,
            frames=o.frames,
            possession_frames=o.possession_frames,
            candidate_frames=o.candidate_frames,
            valid_touch_frames=o.valid_touch_frames,
            lower_body_frames=o.lower_body_frames,
            bbox_only_frames=o.bbox_only_frames,
            min_distance_to_ball=float(o.min_d),
            first_ball_xy=o.first_ball_xy,
            last_ball_xy=o.last_ball_xy,
        )

    def _finish_flight_contact_open(self, tid: int) -> None:
        o = self._flight_contact_open.pop(tid, None)
        if o is None or o.frames <= 0:
            return
        self._flight_contact_finished.append(self._flight_contact_open_to_episode(o))

    def _flight_contact_snapshot(self) -> list[_FlightContactEpisode]:
        out = list(self._flight_contact_finished)
        out.extend(
            self._flight_contact_open_to_episode(o)
            for o in self._flight_contact_open.values()
            if o.frames > 0
        )
        out.sort(key=lambda e: (e.start_frame, e.end_frame, e.player_id))
        return out

    def _record_flight_contact_frame(
        self,
        *,
        cand: _FlightCandidate,
        poss_tid: Optional[int],
        cand_recv: Optional[int],
        frame_index: int,
        ball_xy: Optional[tuple[float, float]],
        track_to_team: dict[int, int],
    ) -> None:
        if ball_xy is None:
            return

        touches: dict[int, dict[str, bool]] = {}

        def add(tid: Optional[int], kind: str) -> None:
            if tid is None or tid == cand.from_player_id:
                return
            rec = touches.setdefault(
                tid,
                {"possession": False, "candidate": False},
            )
            rec[kind] = True

        if poss_tid is not None and poss_tid != self._co_owner:
            add(poss_tid, "possession")
        add(cand_recv, "candidate")

        max_gap = max(2, self._config.pass_receiver_max_gap_frames)
        for tid in list(self._flight_contact_open.keys()):
            if tid in touches:
                continue
            o = self._flight_contact_open[tid]
            o.misses += 1
            if o.misses > max_gap:
                self._finish_flight_contact_open(tid)

        for tid, flags in touches.items():
            bbox = self._last_bbox_xyxy.get(tid)
            foot = self._last_foot.get(tid)
            d_foot = float("inf")
            valid_touch = False
            lower_body = False
            bbox_only = False
            if bbox is not None:
                d_foot, lower_body, valid_touch, bbox_only = classify_touch_features(
                    self._config, bbox, ball_xy
                )
            elif foot is not None:
                d_foot = _dist_px(foot, ball_xy)

            o = self._flight_contact_open.get(tid)
            if o is None:
                o = _FlightContactOpenEpisode(
                    player_id=tid,
                    team_id_snap=track_to_team.get(tid),
                    start_frame=frame_index,
                    end_frame=frame_index,
                    misses=0,
                    frames=0,
                    possession_frames=0,
                    candidate_frames=0,
                    valid_touch_frames=0,
                    lower_body_frames=0,
                    bbox_only_frames=0,
                    min_d=d_foot,
                    first_ball_xy=tuple(ball_xy),
                    last_ball_xy=tuple(ball_xy),
                )
                self._flight_contact_open[tid] = o

            o.end_frame = frame_index
            o.misses = 0
            o.frames += 1
            o.possession_frames += 1 if flags.get("possession") else 0
            o.candidate_frames += 1 if flags.get("candidate") else 0
            o.valid_touch_frames += 1 if valid_touch else 0
            o.lower_body_frames += 1 if lower_body else 0
            o.bbox_only_frames += 1 if bbox_only else 0
            o.min_d = min(o.min_d, d_foot)
            o.last_ball_xy = tuple(ball_xy)
            if o.team_id_snap is None:
                o.team_id_snap = track_to_team.get(tid)

    def _dribble_overlap_guard_fsm(
        self,
        cand: _FlightCandidate,
        receiver_id: int,
        frame_index: int,
        ball_xy: Optional[tuple[float, float]],
    ) -> tuple[bool, str]:
        cfg = self._config
        if not cfg.pass_dribble_guard_enabled:
            return False, ""

        aid = cand.from_player_id
        bid = receiver_id
        wf = max(5, cfg.pass_dribble_guard_window_frames)
        lo = frame_index - wf

        src_lb = 0
        recv_lb = 0
        recv_bo = 0
        recv_first_lb_frame: Optional[int] = None

        for snap in self._touch_hist:
            fi = snap.frame_index
            if fi < lo or fi > frame_index:
                continue
            ta = snap.tracks.get(aid)
            tb = snap.tracks.get(bid)
            if ta is not None and ta.lower_body_valid:
                src_lb += 1
            if tb is None:
                continue
            if tb.lower_body_valid:
                recv_lb += 1
                if recv_first_lb_frame is None:
                    recv_first_lb_frame = fi
            elif tb.bbox_overlap_only:
                recv_bo += 1

        if recv_lb >= cfg.pass_touch_min_receiver_contact_frames:
            return False, ""

        if cfg.pass_dribble_guard_skip_if_source_retains_control:
            brief_recv = recv_lb <= cfg.pass_dribble_guard_max_receiver_contact_frames
            bboxish = recv_bo >= recv_lb + (3 if recv_lb == 0 else 1)
            if brief_recv and src_lb >= wf // 4 + 3 and bboxish:
                return True, "source_retains_control receiver_bbox_only"

        if recv_first_lb_frame is not None and recv_lb <= recv_bo:
            ret_w = cfg.pass_dribble_guard_return_to_same_player_frames
            if ret_w > 0:
                hi = recv_first_lb_frame + ret_w
                for snap in self._touch_hist:
                    fi = snap.frame_index
                    if fi <= recv_first_lb_frame or fi > hi:
                        continue
                    ta = snap.tracks.get(aid)
                    if ta is not None and ta.lower_body_valid:
                        return True, "ball_return_same_player dribble_overlap_not_pass"

        if (
            recv_lb <= cfg.pass_dribble_guard_max_receiver_contact_frames
            and ball_xy is not None
            and cand.ball_anchor_xy is not None
            and recv_bo >= recv_lb + 5
            and recv_lb <= 2
            and (
                _dist_px(cand.ball_anchor_xy, ball_xy)
                < cfg.pass_dribble_guard_min_receiver_distance_gain_px
            )
        ):
            return True, "overlap_low_displacement dribble_overlap_not_pass"

        return False, ""

    def _source_retains_control_guard_fsm(
        self,
        cand: _FlightCandidate,
        receiver_id: int,
        frame_index: int,
        possession: PossessionState,
    ) -> tuple[bool, str]:
        cfg = self._config
        if not cfg.pass_dribble_guard_enabled:
            return False, ""
        if possession.track_id != cand.from_player_id:
            return False, ""

        recv_lb, recv_vt, _recv_bo = self._touch_hist_recv_metrics(
            receiver_id, cand.start_frame, frame_index
        )
        if recv_lb >= 1 or recv_vt >= cfg.pass_valid_touch_min_frames:
            return False, ""

        lo = max(
            cand.start_frame,
            frame_index - max(5, cfg.pass_dribble_guard_window_frames),
        )
        src_lb, src_vt, _src_bo = self._touch_hist_recv_metrics(
            cand.from_player_id, lo, frame_index
        )
        if src_lb < 1 and src_vt < cfg.pass_valid_touch_min_frames:
            return False, ""

        return True, "source_retains_control receiver_unconfirmed"

    def _touch_pair_dribble_skip(self, a: _TouchEpisode, b: _TouchEpisode) -> bool:
        cfg = self._config
        if not cfg.pass_dribble_guard_enabled:
            return False
        if b.lower_body_touch_frames >= cfg.pass_touch_min_receiver_contact_frames:
            return False
        if b.bbox_overlap_only_frames <= b.lower_body_touch_frames + 3:
            return False
        ratio_lb = float(b.lower_body_touch_frames) / float(max(1, b.contact_frames))
        carrier_ok = a.lower_body_touch_frames >= max(
            cfg.pass_receiver_confirm_frames, 8
        )
        shallow_recv = ratio_lb < 0.38 or b.contact_frames <= (
            cfg.pass_dribble_guard_max_receiver_contact_frames + cfg.pass_receiver_confirm_frames // 3
        )
        return carrier_ok and shallow_recv and cfg.pass_touch_require_lower_body_or_footpoint

    def _gather_present_track_ids(
        self, tracks: Optional["FrameTracks"]
    ) -> set[int]:
        """Player track ids observed on this frame (non-negative IDs only)."""

        if tracks is None:
            return set()
        from VisionEngine.schemas.schema import ObjectRole

        out: set[int] = set()
        for inst in tracks.instances:
            if inst.role is not ObjectRole.PLAYER:
                continue
            if inst.track_id >= 0:
                out.add(inst.track_id)
        return out

    def _consume_disappearing_tracks(
        self,
        frame_index: int,
        present_ids: set[int],
        track_to_team: dict[int, int],
    ) -> None:
        cfg = self._config
        if not cfg.pass_event_id_alias_enabled:
            return
        if not self._prev_present_track_ids:
            return
        disappeared = sorted(self._prev_present_track_ids - present_ids)
        for gone in disappeared:
            snap = self._player_geom_previous_frame.get(gone)
            if snap is None:
                continue
            fi, bbox, team_snap = snap
            center = bbox_center(bbox)
            bh = float(bbox[3] - bbox[1])
            bw = float(max(1e-6, bbox[2] - bbox[0]))
            team_id = track_to_team.get(gone)
            if team_id is None:
                team_id = team_snap

            self._lost_track_buffer.append(
                _LostTrackSig(
                    track_id=gone,
                    last_frame=max(fi, frame_index - 1),
                    center_xy=(float(center[0]), float(center[1])),
                    bbox_h=bh,
                    bbox_w=bw,
                    team_id=team_id,
                )
            )

    def _alias_track_if_first_seen(
        self,
        frame_index: int,
        *,
        tid: int,
        present_ids: set[int],
        track_to_team: dict[int, int],
        bbox: tuple[float, float, float, float],
    ) -> None:
        cfg = self._config
        if tid in self._tracks_ever_seen:
            return
        self._tracks_ever_seen.add(tid)
        if not cfg.pass_event_id_alias_enabled:
            return
        if not cfg.pass_event_id_alias_apply_to_decision:
            return

        center = bbox_center(bbox)
        bh = _bbox_height_xyxy(bbox)
        team_here = track_to_team.get(tid)

        match: Optional[_LostTrackSig] = None
        best_metric = float("inf")

        for cand in reversed(list(self._lost_track_buffer)):
            if cand.track_id in present_ids or cand.track_id == tid:
                continue
            gap = frame_index - cand.last_frame
            if gap < 1 or gap > cfg.pass_event_id_alias_max_gap_frames:
                continue
            if (
                cfg.pass_event_id_alias_require_same_team
                and team_here is not None
                and cand.team_id is not None
                and cand.team_id != team_here
            ):
                continue
            dc = _dist_px(center, cand.center_xy)
            if dc > cfg.pass_event_id_alias_max_center_distance_px:
                continue
            if cfg.pass_event_id_alias_require_similar_bbox and cand.bbox_h > 0:
                hh = cand.bbox_h
                dr = (
                    abs(bh - hh) / max(max(bh, hh), 1e-3)
                )
                if dr > cfg.pass_event_id_alias_bbox_height_ratio_tol:
                    continue
            metric = dc + float(gap) * 3.8
            if metric < best_metric:
                best_metric = metric
                match = cand

        if match is None or match.track_id == tid:
            return

        canon = match.track_id
        self._track_alias_fwd[tid] = canon
        self._track_alias_reason[str(tid)] = "recent_nearby_team_bbox_reappear"
        if cfg.pass_debug:
            self._dbg(
                f"[PASS-ALIAS] map display_id={tid} -> canon_id={canon} "
                f"g={frame_index - match.last_frame} d={best_metric:.1f}"
            )

    def _canonical_player_alias(self, tid: Optional[int]) -> tuple[Optional[int], Optional[str]]:
        if tid is None:
            return None, None
        dest = tid
        seen_ids: set[int] = set()
        reason_note: Optional[str] = None
        while dest in self._track_alias_fwd and dest not in seen_ids:
            seen_ids.add(dest)
            nxt = self._track_alias_fwd[dest]
            reason_note = self._track_alias_reason.get(str(dest), reason_note)
            dest = nxt
            if len(seen_ids) > 8:
                break
        return (dest if dest != tid else tid), (reason_note if dest != tid else None)

    def _ephemeral_canonical_alias(
        self,
        tid: int,
        frame_index: int,
        *,
        track_to_team: dict[int, int],
    ) -> tuple[int, Optional[str]]:
        """Resolve conservative ID alias at emit time without mutating forward maps."""

        cfg = self._config
        if not cfg.pass_event_id_alias_enabled:
            return tid, None
        bbox = self._last_bbox_xyxy.get(tid)
        if bbox is None:
            return tid, None
        present_ids = set(self._last_bbox_xyxy.keys())
        center = bbox_center(bbox)
        bh = _bbox_height_xyxy(bbox)
        team_here = track_to_team.get(tid)
        match: Optional[_LostTrackSig] = None
        best_metric = float("inf")

        for cand in reversed(list(self._lost_track_buffer)):
            if cand.track_id in present_ids or cand.track_id == tid:
                continue
            gap = frame_index - cand.last_frame
            if gap < 1 or gap > cfg.pass_event_id_alias_max_gap_frames:
                continue
            if (
                cfg.pass_event_id_alias_require_same_team
                and team_here is not None
                and cand.team_id is not None
                and cand.team_id != team_here
            ):
                continue
            dc = _dist_px(center, cand.center_xy)
            if dc > cfg.pass_event_id_alias_max_center_distance_px:
                continue
            if cfg.pass_event_id_alias_require_similar_bbox and cand.bbox_h > 0:
                hh = cand.bbox_h
                dr = abs(bh - hh) / max(max(bh, hh), 1e-3)
                if dr > cfg.pass_event_id_alias_bbox_height_ratio_tol:
                    continue
            metric = dc + float(gap) * 3.8
            if metric < best_metric:
                best_metric = metric
                match = cand

        if match is None or match.track_id == tid:
            return tid, None
        return match.track_id, "recent_nearby_team_bbox_reappear"

    def _apply_pass_event_aliases(
        self, ev: PassEvent, frame_index: int, *, track_to_team: dict[int, int]
    ) -> None:
        cfg = self._config

        rf0 = ev.from_player_id
        rt0 = ev.to_player_id

        ev.raw_from_player_id = rf0
        ev.raw_to_player_id = (
            ev.receiver_bbox_intermediate_raw
            if ev.receiver_bbox_intermediate_raw is not None
            else rt0
        )

        if not cfg.pass_event_id_alias_enabled:
            ev.canonical_from_player_id = None
            ev.canonical_to_player_id = None
            ev.id_alias_reason = None
            return

        cf: Optional[int] = rf0
        ct: Optional[int] = rt0
        rn_f: Optional[str] = None
        rn_t: Optional[str] = None

        if rf0 is not None:
            if cfg.pass_event_id_alias_apply_to_decision:
                cf, rn_f = self._canonical_player_alias(rf0)
            else:
                cf, rn_f = self._ephemeral_canonical_alias(
                    rf0, frame_index, track_to_team=track_to_team
                )
        if rt0 is not None:
            if cfg.pass_event_id_alias_apply_to_decision:
                ct, rn_t = self._canonical_player_alias(rt0)
            else:
                ct, rn_t = self._ephemeral_canonical_alias(
                    rt0, frame_index, track_to_team=track_to_team
                )

        apply_out = cfg.pass_event_id_alias_apply_to_output
        display_from: Optional[int] = (
            cf if (apply_out and cf is not None) else rf0
        )
        display_to: Optional[int] = ct if (apply_out and ct is not None) else rt0

        ev.from_player_id = display_from
        ev.to_player_id = display_to

        ev.canonical_from_player_id = (
            cf if (cf is not None and rf0 is not None and cf != rf0) else None
        )
        ev.canonical_to_player_id = (
            ct if (ct is not None and rt0 is not None and ct != rt0) else None
        )

        if ev.event_type != "unknown_pass" and display_from is not None:
            nft = track_to_team.get(display_from)
            if nft is not None:
                ev.from_team_id = nft
        if display_to is not None:
            ntt = track_to_team.get(display_to)
            if ntt is not None:
                ev.to_team_id = ntt

        bits: list[str] = []
        if rn_f and cf is not None and rf0 is not None and cf != rf0:
            bits.append(f"from:{rf0}->{cf}:{rn_f}")
        if rn_t and ct is not None and rt0 is not None and ct != rt0:
            bits.append(f"to:{rt0}->{ct}:{rn_t}")

        if bits:
            ev.id_alias_reason = ";".join(bits)
            if cfg.pass_debug:
                self._dbg_gate(
                    "id_alias_evt",
                    frame_index,
                    f"[PASS-ALIAS] applied ev={ev.event_id}: {ev.id_alias_reason}",
                )
                afo = "True" if apply_out else "False"
                afd = "True" if cfg.pass_event_id_alias_apply_to_decision else "False"
                if rn_f and rf0 is not None and cf is not None and cf != rf0:
                    self._dbg_gate(
                        f"pass_alias_detail:{ev.event_id}:from",
                        frame_index,
                        f"[PASS-ALIAS] raw={rf0} canonical={cf} "
                        f"apply_to_output={afo} apply_to_decision={afd}",
                    )
                if rn_t and rt0 is not None and ct is not None and ct != rt0:
                    self._dbg_gate(
                        f"pass_alias_detail:{ev.event_id}:to",
                        frame_index,
                        f"[PASS-ALIAS] raw={rt0} canonical={ct} "
                        f"apply_to_output={afo} apply_to_decision={afd}",
                    )
        else:
            ev.id_alias_reason = None

    def _snapshot_tracking_state_after_frame(
        self,
        frame_index: int,
        tracks: Optional["FrameTracks"],
        track_to_team: dict[int, int],
        present_ids: set[int],
    ) -> None:
        """Caches per-player geometry + updates ``_prev_present_track_ids``."""

        self._player_geom_previous_frame.clear()
        if tracks is not None:
            from VisionEngine.schemas.schema import ObjectRole

            for inst in tracks.instances:
                if inst.role is not ObjectRole.PLAYER:
                    continue
                tid = inst.track_id
                if tid < 0:
                    continue
                bbox = tuple(float(x) for x in inst.xyxy)
                team_snap = track_to_team.get(tid)
                self._player_geom_previous_frame[tid] = (
                    frame_index,
                    bbox,
                    team_snap,
                )

        self._prev_present_track_ids = set(present_ids)

    def _maybe_alias_first_seen_players(
        self,
        frame_index: int,
        tracks: Optional["FrameTracks"],
        present_ids: set[int],
        track_to_team: dict[int, int],
    ) -> None:
        if tracks is None:
            return
        from VisionEngine.schemas.schema import ObjectRole

        for inst in tracks.instances:
            if inst.role is not ObjectRole.PLAYER:
                continue
            tid = inst.track_id
            if tid < 0:
                continue
            bbox = tuple(float(x) for x in inst.xyxy)
            self._alias_track_if_first_seen(
                frame_index,
                tid=tid,
                present_ids=present_ids,
                track_to_team=track_to_team,
                bbox=bbox,
            )

    def _try_emit_one_touch(
        self,
        *,
        frame_index: int,
        timestamp_sec: Optional[float],
        possession: PossessionState,
        track_to_team: dict[int, int],
        tracks: Optional["FrameTracks"],
    ) -> Optional[PassEvent]:
        cfg = self._config
        if not cfg.pass_allow_one_touch_release:
            return None

        recv = self._last_receiver_id
        base_ef = self._last_event_end_frame
        owner = self._co_owner
        deadline = (
            None if base_ef is None else base_ef + cfg.pass_one_touch_window_frames
        )

        if deadline is not None and frame_index > deadline:
            if (
                cfg.pass_debug
                and recv is not None
                and owner == recv
                and self._one_touch_contact_frame is not None
            ):
                self._dbg(
                    "[PASS-FSM] one_touch_rejected owner="
                    f"{recv} reason=outside_one_touch_window"
                )
            self._clear_one_touch_accumulator()
            return None

        if (
            recv is None
            or base_ef is None
            or owner != recv
            or not self._post_receive_blocked(owner, frame_index)
        ):
            self._clear_one_touch_accumulator()
            return None

        ball_xy = possession.ball_xy
        if ball_xy is None or tracks is None:
            return None

        f_own = self._foot_of(owner, possession)
        if f_own is None:
            return None

        bubble = cfg.pass_one_touch_control_radius_px
        d_own = _dist_px(f_own, ball_xy)

        if self._one_touch_contact_frame is None:
            if d_own <= bubble:
                self._one_touch_contact_xy = tuple(ball_xy)
                self._one_touch_contact_frame = frame_index
                self._dbg(
                    f"[PASS-FSM] one_touch_candidate owner={owner} frame={frame_index}"
                )
            return None

        if self._one_touch_contact_xy is not None:
            self._one_touch_max_out = max(
                self._one_touch_max_out,
                _dist_px(self._one_touch_contact_xy, ball_xy),
            )

        if d_own > bubble:
            self._one_touch_away_run += 1
            if self._one_touch_first_away_frame is None:
                self._one_touch_first_away_frame = frame_index
        else:
            self._one_touch_away_run = 0

        cand_recv = self._nearest_receiver_tid(
            tracks=tracks,
            ball_xy=ball_xy,
            exclude=owner,
            radius_px=bubble,
            frame_index=frame_index,
            release_anchor=self._one_touch_contact_xy,
            track_to_team=track_to_team,
        )
        rx = cand_recv
        if rx is None:
            self._one_touch_recv_gap += 1
            if self._one_touch_recv_gap > cfg.pass_receiver_max_gap_frames:
                self._one_touch_recv_streak = 0
                self._one_touch_recv_tid = None
                self._one_touch_recv_confirm_start_frame = None
        elif self._one_touch_recv_tid is None:
            self._one_touch_recv_tid = rx
            self._one_touch_recv_streak = 1
            self._one_touch_recv_gap = 0
            self._one_touch_recv_confirm_start_frame = frame_index
        elif rx == self._one_touch_recv_tid:
            self._one_touch_recv_streak += 1
            self._one_touch_recv_gap = 0
            if cfg.pass_debug and (
                self._one_touch_recv_streak
                == cfg.pass_one_touch_receiver_confirm_frames
            ):
                self._dbg(
                    f"[PASS-FSM] one_touch_receiver_candidate={rx} frames="
                    f"{self._one_touch_recv_streak} confirmed"
                )
        else:
            self._one_touch_recv_tid = rx
            self._one_touch_recv_streak = 1
            self._one_touch_recv_gap = 0
            self._one_touch_recv_confirm_start_frame = frame_index

        recv_ok = (
            self._one_touch_recv_streak >= cfg.pass_one_touch_receiver_confirm_frames
            and self._one_touch_recv_tid is not None
            and self._one_touch_recv_tid != owner
        )
        motion_ok = (
            self._one_touch_away_run >= cfg.pass_one_touch_min_away_frames
            and self._one_touch_max_out
            >= cfg.pass_one_touch_min_outgoing_displacement_px
            and self._one_touch_contact_frame is not None
        )

        emit_ok = motion_ok and recv_ok
        if not emit_ok and frame_index == deadline and self._one_touch_contact_frame:
            reason = ""
            if not motion_ok and not recv_ok:
                reason = "insufficient_motion_and_receiver"
            elif not motion_ok:
                reason = "insufficient_outgoing_displacement"
            else:
                reason = "receiver_not_confirmed"
            if cfg.pass_debug:
                self._dbg(
                    f"[PASS-FSM] one_touch_rejected owner={owner} reason={reason}"
                )
            self._clear_one_touch_accumulator()
            return None

        if not emit_ok:
            return None

        start_f = self._one_touch_contact_frame
        flight_len = frame_index - start_f + 1
        if flight_len < cfg.min_pass_frames:
            return None

        cand = _FlightCandidate(
            start_frame=start_f,
            start_time_sec=timestamp_sec,
            from_player_id=owner,
            from_team_id=track_to_team.get(owner),
            ball_anchor_xy=self._one_touch_contact_xy,
            release_frame=self._one_touch_first_away_frame or start_f,
        )
        receiver_id = self._one_touch_recv_tid
        assert receiver_id is not None
        evidence: dict[str, Any] = {
            "contact_frame": start_f,
            "first_away_frame": self._one_touch_first_away_frame,
            "max_displacement_px": round(self._one_touch_max_out, 4),
            "away_run_peak": cfg.pass_one_touch_min_away_frames,
            "receiver_id": receiver_id,
        }

        self._dbg(
            f"[PASS-FSM] one_touch_confirmed {owner}->{receiver_id} frames={flight_len}"
        )
        ev_o = self._emit_completed(
            cand=cand,
            receiver_id=receiver_id,
            flight_len=flight_len,
            frame_index=frame_index,
            timestamp_sec=timestamp_sec,
            possession=possession,
            track_to_team=track_to_team,
            one_touch=True,
            one_touch_evidence=evidence,
            tracks=tracks,
            ball_xy_for_guard=tuple(ball_xy) if ball_xy is not None else None,
            receiver_selection_reason="one_touch_pick",
            receiver_score=None,
        )
        if ev_o is None:
            self._clear_one_touch_accumulator()
            return None
        return ev_o

    def _advance_transient_overlap(
        self, poss_tid: Optional[int], from_id: int, cand_recv: Optional[int], frame_index: int
    ) -> None:
        cfg = self._config
        if (
            poss_tid is not None
            and poss_tid not in (from_id, cand_recv)
            and poss_tid != self._co_owner
        ):
            if poss_tid == self._intr_tid:
                self._intr_run += 1
            else:
                self._intr_tid = poss_tid
                self._intr_run = 1
            if cfg.pass_debug and self._intr_run == 1:
                self._dbg(
                    "[PASS-FSM] transient_contact track="
                    f"{poss_tid} frame={frame_index}"
                )
            return

        if self._intr_tid is not None and self._intr_run > 0:
            if self._intr_run <= cfg.pass_transient_contact_max_frames:
                self._ignored_transient_episodes += 1
                self._dbg(
                    "[PASS-FSM] transient contact ignored "
                    f"track={self._intr_tid} frames={self._intr_run}"
                )
        self._intr_tid = None
        self._intr_run = 0

    def _flight_contact_duplicate_blocks(
        self,
        *,
        from_id: Optional[int],
        to_id: int,
        start_frame: int,
        end_frame: int,
        frame_index: int,
        window_frames: int = 24,
    ) -> bool:
        def overlap(s1: int, e1: int, s2: int, e2: int) -> bool:
            return not (e1 < s2 or s1 > e2)

        for ev in self._pass_emission_log:
            if overlap(start_frame, end_frame, ev.start_frame, ev.end_frame):
                return True
            if (
                ev.from_player_id == from_id
                and ev.to_player_id == to_id
                and abs(ev.end_frame - end_frame) <= window_frames
            ):
                return True
            if ev.to_player_id == to_id and abs(ev.end_frame - end_frame) <= 8:
                return True
        pending = self._pending_weak_chain_event
        if pending is not None and overlap(
            start_frame, end_frame, pending.start_frame, pending.end_frame
        ):
            return True
        del frame_index
        return False

    @staticmethod
    def _flight_contact_label(ep: _FlightContactEpisode) -> str:
        if ep.lower_body_frames > 0:
            return "flight_contact_lower_body"
        if ep.valid_touch_frames > 0:
            return "flight_contact_valid_touch"
        if ep.possession_frames > 0:
            return "flight_contact_possession"
        if ep.candidate_frames > 0:
            return "flight_contact_candidate"
        if ep.bbox_only_frames > 0:
            return "flight_contact_bbox_only"
        return "flight_contact"

    def _emit_flight_contact_transfer_event(
        self,
        *,
        from_id: int,
        to_id: int,
        start_frame: int,
        end_frame: int,
        release_frame: int,
        receiver_confirm_start_frame: int,
        receiver_confirm_frames: int,
        ball_start_xy: Optional[tuple[float, float]],
        ball_end_xy: Optional[tuple[float, float]],
        frame_index: int,
        timestamp_sec: Optional[float],
        possession: PossessionState,
        track_to_team: dict[int, int],
        tracks: Optional["FrameTracks"],
        source: str,
        debug_reason: str,
        receiver_contact_type: str,
        receiver_selection_reason: str,
        confidence_cap: float,
        check_duplicate: bool = True,
    ) -> Optional[PassEvent]:
        if check_duplicate and self._flight_contact_duplicate_blocks(
            from_id=from_id,
            to_id=to_id,
            start_frame=start_frame,
            end_frame=end_frame,
            frame_index=frame_index,
        ):
            return None

        cfg = self._config
        from_team = track_to_team.get(from_id)
        to_team = self._resolver_team(possession, to_id, track_to_team)
        evt_type = self._classify(
            from_team,
            to_team,
            cfg.require_same_team_for_completed,
            cfg.emit_interceptions,
        )
        duration_frames = max(1, end_frame - start_frame + 1)
        disp = None
        if ball_start_xy is not None and ball_end_xy is not None:
            disp = _dist_px(ball_start_xy, ball_end_xy)

        event = PassEvent(
            event_id=self._next_event_id,
            event_type=evt_type,
            start_frame=start_frame,
            end_frame=end_frame,
            start_time_sec=None,
            end_time_sec=timestamp_sec,
            from_player_id=from_id,
            to_player_id=to_id,
            from_team_id=from_team,
            to_team_id=to_team,
            duration_frames=duration_frames,
            duration_sec=None,
            confidence=confidence_cap,
            ball_start_xy=ball_start_xy,
            ball_end_xy=ball_end_xy,
            raw_from_team_id=from_team,
            raw_to_team_id=to_team,
            team_stable=False,
            debug_reason=debug_reason,
            source=source,
            release_frame=release_frame,
            receiver_confirm_start_frame=receiver_confirm_start_frame,
            receiver_confirm_frames=receiver_confirm_frames,
            ball_displacement_px=disp,
            pass_fsm_state=source,
            receiver_selection_reason=receiver_selection_reason,
            dribble_guard_status="passed_guard",
            receiver_contact_type=receiver_contact_type,
            bbox_only_guard_status="skipped_flight_contact_fallback",
            raw_from_player_id=from_id,
            raw_to_player_id=to_id,
            source_reliability="flight_contact",
            source_reliability_reason=source,
            receiver_contact_decision=receiver_contact_type,
        )
        self._next_event_id += 1
        event = self._finalize_emitted_pass(event)
        self._apply_pass_event_aliases(event, frame_index, track_to_team=track_to_team)
        bx = ball_end_xy
        if bx is None and possession.ball_xy is not None:
            bx = (float(possession.ball_xy[0]), float(possession.ball_xy[1]))
        self._attach_pass_decision_telemetry(
            event,
            tracks=tracks,
            ball_xy=bx,
            track_to_team=track_to_team,
            frame_index=frame_index,
            emit_path_base=source,
            flight_start_frame=start_frame,
            flight_end_frame=end_frame,
            release_anchor_xy=ball_start_xy,
        )
        self._remember_pass_emission(event)
        return event

    def _source_return_after_flight_contact(
        self,
        cand: _FlightCandidate,
        ep: _FlightContactEpisode,
        frame_index: int,
    ) -> Optional[tuple[int, int, int, float, Optional[tuple[float, float]], Optional[tuple[float, float]]]]:
        cfg = self._config
        lo = ep.end_frame + max(2, cfg.pass_receiver_max_gap_frames)
        hi = frame_index
        threshold = max(70.0, cfg.pass_touch_contact_radius_px)
        frames: list[int] = []
        min_d = float("inf")
        first_ball: Optional[tuple[float, float]] = None
        last_ball: Optional[tuple[float, float]] = None
        for snap in self._snap_hist:
            fi = snap.frame_index
            if fi < lo or fi > hi or snap.ball_xy is None:
                continue
            foot = snap.feet.get(cand.from_player_id)
            if foot is None:
                continue
            d = _dist_px(foot, snap.ball_xy)
            if d > threshold:
                continue
            frames.append(fi)
            min_d = min(min_d, d)
            if first_ball is None:
                first_ball = snap.ball_xy
            last_ball = snap.ball_xy
        if len(frames) < 2:
            return None
        return frames[0], frames[-1], len(frames), min_d, first_ball, last_ball

    def _flight_contact_mid_split_candidate(
        self,
        *,
        cand: _FlightCandidate,
        final_receiver_id: int,
        final_confirm_start: int,
    ) -> Optional[_FlightContactEpisode]:
        cfg = self._config
        candidates: list[_FlightContactEpisode] = []
        for ep in self._flight_contact_snapshot():
            if ep.player_id in (cand.from_player_id, final_receiver_id):
                continue
            if self._invalid_intermediate_active(ep.player_id, final_confirm_start):
                continue
            if ep.start_frame <= cand.release_frame + 1:
                continue
            if ep.end_frame >= final_confirm_start - max(4, cfg.pass_receiver_confirm_frames):
                continue
            md = ep.min_distance_to_ball
            strong_candidate = ep.candidate_frames >= 1 and md <= 65.0
            strong_possession = ep.possession_frames >= 1 and md <= 70.0
            strong_valid = ep.valid_touch_frames >= 1 and md <= 80.0
            if strong_candidate or strong_possession or strong_valid:
                candidates.append(ep)

        if not candidates:
            return None

        return min(
            candidates,
            key=lambda e: (
                e.min_distance_to_ball,
                -e.possession_frames,
                -e.candidate_frames,
                e.start_frame,
            ),
        )

    def _try_emit_flight_contact_split(
        self,
        *,
        cand: _FlightCandidate,
        final_receiver_id: int,
        frame_index: int,
        timestamp_sec: Optional[float],
        possession: PossessionState,
        track_to_team: dict[int, int],
        tracks: Optional["FrameTracks"],
    ) -> list[PassEvent]:
        final_start = self._recv_confirm_start_frame or frame_index
        mid = self._flight_contact_mid_split_candidate(
            cand=cand,
            final_receiver_id=final_receiver_id,
            final_confirm_start=final_start,
        )
        if mid is None:
            return []

        first_end = max(mid.end_frame, mid.start_frame)
        second_start = max(mid.end_frame, mid.start_frame)

        if self._flight_contact_duplicate_blocks(
            from_id=cand.from_player_id,
            to_id=mid.player_id,
            start_frame=cand.start_frame,
            end_frame=first_end,
            frame_index=frame_index,
        ) or self._flight_contact_duplicate_blocks(
            from_id=mid.player_id,
            to_id=final_receiver_id,
            start_frame=second_start,
            end_frame=frame_index,
            frame_index=frame_index,
        ):
            return []

        label = self._flight_contact_label(mid)
        ev1 = self._emit_flight_contact_transfer_event(
            from_id=cand.from_player_id,
            to_id=mid.player_id,
            start_frame=cand.start_frame,
            end_frame=first_end,
            release_frame=cand.release_frame,
            receiver_confirm_start_frame=mid.start_frame,
            receiver_confirm_frames=max(1, mid.frames),
            ball_start_xy=cand.ball_anchor_xy,
            ball_end_xy=mid.last_ball_xy,
            frame_index=frame_index,
            timestamp_sec=timestamp_sec,
            possession=possession,
            track_to_team=track_to_team,
            tracks=tracks,
            source="flight_contact_split_fallback",
            debug_reason="flight_contact_split_intermediate",
            receiver_contact_type=label,
            receiver_selection_reason="flight_contact_intermediate_split",
            confidence_cap=0.74,
            check_duplicate=False,
        )
        ev2 = self._emit_flight_contact_transfer_event(
            from_id=mid.player_id,
            to_id=final_receiver_id,
            start_frame=second_start,
            end_frame=frame_index,
            release_frame=second_start,
            receiver_confirm_start_frame=final_start,
            receiver_confirm_frames=self._recv_streak,
            ball_start_xy=mid.last_ball_xy,
            ball_end_xy=(
                tuple(possession.ball_xy) if possession.ball_xy is not None else None
            ),
            frame_index=frame_index,
            timestamp_sec=timestamp_sec,
            possession=possession,
            track_to_team=track_to_team,
            tracks=tracks,
            source="flight_contact_split_fallback",
            debug_reason="flight_contact_split_final_receiver",
            receiver_contact_type="confirmed_receiver",
            receiver_selection_reason="flight_contact_final_receiver_split",
            confidence_cap=0.74,
            check_duplicate=False,
        )
        if ev1 is None or ev2 is None:
            return []

        self._dbg_gate(
            f"flight_split:{cand.from_player_id}->{mid.player_id}->{final_receiver_id}",
            frame_index,
            "[PASS-FLIGHT-CONTACT] split "
            f"{cand.from_player_id}->{final_receiver_id} via {mid.player_id}",
        )
        self._apply_emit_shared_state(
            frame_index, mid.player_id, final_receiver_id, clear_fsm_flight=True
        )
        self._ignored_transient_episodes = 0
        return [ev1, ev2]

    def _try_emit_timeout_source_loose_split(
        self,
        *,
        cand: _FlightCandidate,
        final_receiver_id: int,
        frame_index: int,
        timestamp_sec: Optional[float],
        possession: PossessionState,
        track_to_team: dict[int, int],
        tracks: Optional["FrameTracks"],
    ) -> list[PassEvent]:
        cur = cand.from_player_id
        last_ev = self._pass_emission_log[-1] if self._pass_emission_log else None
        if (
            last_ev is None
            or last_ev.source != "flight_contact_timeout_fallback"
            or last_ev.to_player_id != cur
            or cur != self._last_receiver_id
        ):
            return []

        cfg = self._config
        lb = max(0, cand.release_frame - cfg.pass_source_recover_lookback_frames)
        hi = cand.release_frame + cfg.pass_source_reliability_lookahead_frames
        loose = self._recover_recent_loose_source_excluding(
            before_frame=cand.release_frame,
            exclude_ids={cur, final_receiver_id},
            frame_index=frame_index,
            track_to_team=track_to_team,
            lookback_frames=cfg.pass_none_source_recovery_lookback_frames,
        )
        if loose is None:
            return []

        loose_id, _loose_team, loose_frame, loose_dist, loose_ball = loose
        _cur_vis, cur_md = self._snap_visible_ball_foot_metrics(cur, lb, hi)
        if cur_md is not None and loose_dist + 20.0 >= cur_md:
            return []

        first_start = max(
            last_ev.end_frame + 1, last_ev.release_frame or last_ev.end_frame
        )
        first_end = max(first_start, loose_frame)
        if self._flight_contact_duplicate_blocks(
            from_id=cur,
            to_id=loose_id,
            start_frame=first_start,
            end_frame=first_end,
            frame_index=frame_index,
        ) or self._flight_contact_duplicate_blocks(
            from_id=loose_id,
            to_id=final_receiver_id,
            start_frame=cand.start_frame,
            end_frame=frame_index,
            frame_index=frame_index,
        ):
            return []

        ev1 = self._emit_flight_contact_transfer_event(
            from_id=cur,
            to_id=loose_id,
            start_frame=first_start,
            end_frame=first_end,
            release_frame=first_start,
            receiver_confirm_start_frame=loose_frame,
            receiver_confirm_frames=1,
            ball_start_xy=last_ev.ball_end_xy,
            ball_end_xy=loose_ball,
            frame_index=frame_index,
            timestamp_sec=timestamp_sec,
            possession=possession,
            track_to_team=track_to_team,
            tracks=tracks,
            source="flight_contact_source_split_fallback",
            debug_reason="timeout_receiver_to_loose_source",
            receiver_contact_type="flight_contact_loose_source",
            receiver_selection_reason="timeout_receiver_loose_source_split",
            confidence_cap=0.68,
            check_duplicate=False,
        )
        ev2 = self._emit_flight_contact_transfer_event(
            from_id=loose_id,
            to_id=final_receiver_id,
            start_frame=cand.start_frame,
            end_frame=frame_index,
            release_frame=cand.release_frame,
            receiver_confirm_start_frame=self._recv_confirm_start_frame or frame_index,
            receiver_confirm_frames=self._recv_streak,
            ball_start_xy=loose_ball or cand.ball_anchor_xy,
            ball_end_xy=(
                tuple(possession.ball_xy) if possession.ball_xy is not None else None
            ),
            frame_index=frame_index,
            timestamp_sec=timestamp_sec,
            possession=possession,
            track_to_team=track_to_team,
            tracks=tracks,
            source="flight_contact_source_split_fallback",
            debug_reason="loose_source_to_final_receiver",
            receiver_contact_type="confirmed_receiver",
            receiver_selection_reason="timeout_loose_source_final_receiver",
            confidence_cap=0.74,
            check_duplicate=False,
        )
        if ev1 is None or ev2 is None:
            return []

        self._dbg_gate(
            f"timeout_source_split:{cur}->{loose_id}->{final_receiver_id}",
            frame_index,
            "[PASS-FLIGHT-CONTACT] source split "
            f"{cur}->{final_receiver_id} via {loose_id}",
        )
        self._apply_emit_shared_state(
            frame_index, loose_id, final_receiver_id, clear_fsm_flight=True
        )
        self._ignored_transient_episodes = 0
        return [ev1, ev2]

    def _timeout_rebound_contact(
        self, cand: _FlightCandidate, frame_index: int
    ) -> Optional[tuple[_FlightContactEpisode, tuple[int, int, int, float, Optional[tuple[float, float]], Optional[tuple[float, float]]]]]:
        candidates: list[
            tuple[
                _FlightContactEpisode,
                tuple[int, int, int, float, Optional[tuple[float, float]], Optional[tuple[float, float]]],
            ]
        ] = []
        for ep in self._flight_contact_snapshot():
            if ep.player_id == cand.from_player_id:
                continue
            if ep.possession_frames < 1:
                continue
            if ep.min_distance_to_ball > 95.0:
                continue
            ret = self._source_return_after_flight_contact(cand, ep, frame_index)
            if ret is None:
                continue
            candidates.append((ep, ret))
        if not candidates:
            return None
        return max(
            candidates,
            key=lambda z: (
                z[1][2],
                -z[0].min_distance_to_ball,
                z[0].end_frame,
            ),
        )

    def _timeout_single_contact(
        self, cand: _FlightCandidate, frame_index: int
    ) -> Optional[_FlightContactEpisode]:
        candidates: list[_FlightContactEpisode] = []
        for ep in self._flight_contact_snapshot():
            if ep.player_id == cand.from_player_id:
                continue
            if self._invalid_intermediate_active(ep.player_id, frame_index):
                continue
            if ep.start_frame <= cand.release_frame + 1:
                continue
            md = ep.min_distance_to_ball
            strong_candidate = ep.candidate_frames >= 1 and md <= 85.0
            strong_valid = ep.valid_touch_frames >= 1 and md <= 90.0
            strong_possession = ep.possession_frames >= 1 and md <= 70.0
            if not (strong_candidate or strong_valid or strong_possession):
                continue
            if ep.candidate_frames < 1 and ep.valid_touch_frames < 1:
                continue
            candidates.append(ep)
        if not candidates:
            return None

        return min(
            candidates,
            key=lambda e: (
                e.min_distance_to_ball + 0.08 * max(0, e.start_frame - cand.release_frame),
                -e.candidate_frames,
                e.start_frame,
            ),
        )

    def _loose_preface_source_before_flight(
        self,
        *,
        cand: _FlightCandidate,
        exclude_to: int,
        lookback_frames: int = 80,
    ) -> Optional[tuple[int, int, float, Optional[tuple[float, float]]]]:
        if self._pass_emission_log:
            return None
        lb = max(0, cand.start_frame - lookback_frames)
        best: Optional[tuple[int, int, float, Optional[tuple[float, float]]]] = None
        max_d = 95.0
        for snap in self._snap_hist:
            if snap.frame_index < lb or snap.frame_index >= cand.start_frame:
                continue
            if snap.ball_xy is None:
                continue
            for tid, fxy in snap.feet.items():
                if tid in (cand.from_player_id, exclude_to):
                    continue
                d = _dist_px(fxy, snap.ball_xy)
                if d > max_d:
                    continue
                if best is None or snap.frame_index > best[1] or (
                    snap.frame_index == best[1] and d < best[2]
                ):
                    best = (tid, snap.frame_index, d, snap.ball_xy)
        return best

    def _try_emit_timeout_contact_transfer(
        self,
        *,
        cand: _FlightCandidate,
        frame_index: int,
        timestamp_sec: Optional[float],
        possession: PossessionState,
        track_to_team: dict[int, int],
        tracks: Optional["FrameTracks"],
    ) -> list[PassEvent]:
        events: list[PassEvent] = []

        rebound = self._timeout_rebound_contact(cand, frame_index)
        if rebound is not None:
            mid, ret = rebound
            ret_start, ret_end, ret_frames, _ret_min, ret_first_ball, ret_last_ball = ret
            if self._flight_contact_duplicate_blocks(
                from_id=cand.from_player_id,
                to_id=mid.player_id,
                start_frame=cand.start_frame,
                end_frame=mid.end_frame,
                frame_index=frame_index,
            ) or self._flight_contact_duplicate_blocks(
                from_id=mid.player_id,
                to_id=cand.from_player_id,
                start_frame=ret_start,
                end_frame=ret_end,
                frame_index=frame_index,
            ):
                return []
            ev1 = self._emit_flight_contact_transfer_event(
                from_id=cand.from_player_id,
                to_id=mid.player_id,
                start_frame=cand.start_frame,
                end_frame=mid.end_frame,
                release_frame=cand.release_frame,
                receiver_confirm_start_frame=mid.start_frame,
                receiver_confirm_frames=max(1, mid.frames),
                ball_start_xy=cand.ball_anchor_xy,
                ball_end_xy=mid.last_ball_xy,
                frame_index=frame_index,
                timestamp_sec=timestamp_sec,
                possession=possession,
                track_to_team=track_to_team,
                tracks=tracks,
                source="flight_contact_timeout_fallback",
                debug_reason="flight_contact_timeout_transfer",
                receiver_contact_type=self._flight_contact_label(mid),
                receiver_selection_reason="timeout_rebound_contact",
                confidence_cap=0.70,
                check_duplicate=False,
            )
            ev2 = self._emit_flight_contact_transfer_event(
                from_id=mid.player_id,
                to_id=cand.from_player_id,
                start_frame=ret_start,
                end_frame=ret_end,
                release_frame=max(mid.end_frame, ret_start),
                receiver_confirm_start_frame=ret_start,
                receiver_confirm_frames=ret_frames,
                ball_start_xy=ret_first_ball or mid.last_ball_xy,
                ball_end_xy=ret_last_ball,
                frame_index=frame_index,
                timestamp_sec=timestamp_sec,
                possession=possession,
                track_to_team=track_to_team,
                tracks=tracks,
                source="flight_contact_timeout_fallback",
                debug_reason="flight_contact_timeout_rebound_to_source",
                receiver_contact_type="flight_contact_source_reclaim",
                receiver_selection_reason="timeout_source_reclaim",
                confidence_cap=0.70,
                check_duplicate=False,
            )
            if ev1 is not None and ev2 is not None:
                events.extend([ev1, ev2])
        else:
            mid = self._timeout_single_contact(cand, frame_index)
            if mid is None:
                return []
            pre = self._loose_preface_source_before_flight(
                cand=cand, exclude_to=mid.player_id
            )
            if pre is not None:
                pre_tid, pre_frame, _pre_d, pre_ball = pre
                pre_ev = self._emit_flight_contact_transfer_event(
                    from_id=pre_tid,
                    to_id=cand.from_player_id,
                    start_frame=pre_frame,
                    end_frame=max(pre_frame, cand.start_frame - 1),
                    release_frame=pre_frame,
                    receiver_confirm_start_frame=max(pre_frame, cand.start_frame - 1),
                    receiver_confirm_frames=1,
                    ball_start_xy=pre_ball,
                    ball_end_xy=cand.ball_anchor_xy,
                    frame_index=frame_index,
                    timestamp_sec=timestamp_sec,
                    possession=possession,
                    track_to_team=track_to_team,
                    tracks=tracks,
                    source="flight_contact_timeout_fallback",
                    debug_reason="flight_contact_timeout_preface_source",
                    receiver_contact_type="flight_contact_source_bootstrap",
                    receiver_selection_reason="timeout_loose_preface_source",
                    confidence_cap=0.66,
                )
                if pre_ev is not None:
                    events.append(pre_ev)

            ev = self._emit_flight_contact_transfer_event(
                from_id=cand.from_player_id,
                to_id=mid.player_id,
                start_frame=cand.start_frame,
                end_frame=mid.end_frame,
                release_frame=cand.release_frame,
                receiver_confirm_start_frame=mid.start_frame,
                receiver_confirm_frames=max(1, mid.frames),
                ball_start_xy=cand.ball_anchor_xy,
                ball_end_xy=mid.last_ball_xy,
                frame_index=frame_index,
                timestamp_sec=timestamp_sec,
                possession=possession,
                track_to_team=track_to_team,
                tracks=tracks,
                source="flight_contact_timeout_fallback",
                debug_reason="flight_contact_timeout_transfer",
                receiver_contact_type=self._flight_contact_label(mid),
                receiver_selection_reason="timeout_best_contact",
                confidence_cap=0.70,
            )
            if ev is not None:
                events.append(ev)

        if not events:
            return []

        last_ev = events[-1]
        if last_ev.to_player_id is not None:
            self._dbg_gate(
                f"flight_timeout_transfer:{cand.from_player_id}->{last_ev.to_player_id}",
                frame_index,
                "[PASS-FLIGHT-CONTACT] timeout transfer "
                f"from={cand.from_player_id} final={last_ev.to_player_id}",
            )
            self._apply_emit_shared_state(
                frame_index,
                last_ev.from_player_id,
                last_ev.to_player_id,
                clear_fsm_flight=True,
            )
            self._ignored_transient_episodes = 0
        return events

    def _touch_hist_recv_metrics(
        self, recv_id: int, lo_frame: int, hi_frame: int
    ) -> tuple[int, int, int]:
        """Lower-body frames, valid-touch frames, bbox-overlap-with-far-foot frames."""

        cfg = self._config
        min_px = cfg.pass_bbox_only_min_footpoint_distance_px
        lb_ct = 0
        vt_ct = 0
        bo_far_ct = 0
        for snap in self._touch_hist:
            fi = snap.frame_index
            if fi < lo_frame or fi > hi_frame:
                continue
            tb = snap.tracks.get(recv_id)
            if tb is None:
                continue
            if tb.lower_body_valid:
                lb_ct += 1
            if tb.valid_touch:
                vt_ct += 1
            if tb.bbox_overlap_only and tb.d_foot > min_px:
                bo_far_ct += 1
        return lb_ct, vt_ct, bo_far_ct

    def _receiver_contact_label_for_emit(
        self, recv_id: int, cand: _FlightCandidate, frame_index: int
    ) -> str:
        cfg = self._config
        lb_ct, vt_ct, bo_far = self._touch_hist_recv_metrics(
            recv_id, cand.start_frame, frame_index
        )
        if lb_ct >= max(2, cfg.pass_receiver_confirm_frames // 2):
            return "lower_body_sustained"
        if lb_ct >= 1:
            return "lower_body"
        if vt_ct >= cfg.pass_receiver_confirm_frames:
            return "valid_touch_sustained"
        if vt_ct >= 1:
            return "valid_touch"
        if bo_far >= 1:
            return "bbox_overlap_skim"
        return "unknown"

    def _bbox_guard_chain_suppressed(self, from_tid: int, frame_index: int) -> bool:
        cfg = self._config
        if not cfg.pass_bbox_only_intermediate_guard_enabled:
            return False
        win = cfg.pass_bbox_only_chain_window_frames
        for ef, rid in self._suppressed_bbox_receiver_chain:
            if frame_index - ef > win:
                continue
            if rid == from_tid:
                return True
        return False

    def _bbox_receiver_crossing_should_suppress(
        self,
        *,
        recv_id: int,
        cand: _FlightCandidate,
        frame_index: int,
    ) -> bool:
        cfg = self._config
        if not cfg.pass_bbox_only_intermediate_guard_enabled:
            return False

        lb_ct, vt_ct, bo_far_ct = self._touch_hist_recv_metrics(
            recv_id, cand.start_frame, frame_index
        )
        if lb_ct >= 1:
            return False
        if cfg.pass_bbox_only_require_lower_body_touch:
            if vt_ct >= cfg.pass_receiver_confirm_frames:
                return False
        elif vt_ct >= 1:
            return False
        if (
            1 <= bo_far_ct <= cfg.pass_bbox_only_max_contact_frames
        ):
            return True
        return False

    def _emit_completed(
        self,
        *,
        cand: _FlightCandidate,
        receiver_id: int,
        flight_len: int,
        frame_index: int,
        timestamp_sec: Optional[float],
        possession: PossessionState,
        track_to_team: dict[int, int],
        one_touch: bool = False,
        one_touch_evidence: Optional[dict[str, Any]] = None,
        tracks: Optional["FrameTracks"] = None,
        ball_xy_for_guard: Optional[tuple[float, float]] = None,
        receiver_selection_reason: Optional[str] = None,
        receiver_score: Optional[float] = None,
    ) -> Optional[PassEvent]:
        cfg = self._config
        gx = ball_xy_for_guard
        if gx is None and possession.ball_xy is not None:
            gx = (float(possession.ball_xy[0]), float(possession.ball_xy[1]))

        receiver_bbox_intermediate_raw: Optional[int] = None
        recv_r_override: Optional[str] = None
        bbox_guard_stat = (
            "passed"
            if cfg.pass_bbox_only_intermediate_guard_enabled
            else "disabled"
        )

        if cfg.pass_invalid_intermediate_memory_enabled:
            fm0 = cand.from_player_id
            if fm0 is not None and self._invalid_intermediate_active(fm0, frame_index):
                alt = self._recover_passer_last_valid_touch_simple(
                    before_frame=cand.start_frame,
                    exclude_recv=receiver_id,
                    frame_index=frame_index,
                    track_to_team=track_to_team,
                    lookback_frames=max(
                        cfg.pass_none_source_recovery_lookback_frames,
                        cfg.pass_source_recover_lookback_frames,
                    ),
                )
                if alt is not None:
                    tid_r, nft_r = alt
                    cand.from_player_id = tid_r
                    cand.from_team_id = nft_r or track_to_team.get(tid_r)
                else:
                    self._dbg_gate(
                        f"inv_mid_sup_src:{fm0}->{receiver_id}",
                        frame_index,
                        "[PASS-INVALID-INTERMEDIATE] suppress "
                        f"{fm0}->{receiver_id} reason=source_invalid_intermediate",
                    )
                    self._print_pass_decision_suppressed(
                        frame_index=frame_index,
                        cand_from=fm0,
                        cand_to=receiver_id,
                        reason="source_invalid_intermediate",
                    )
                    return None

            if (
                self._invalid_intermediate_active(receiver_id, frame_index)
                and gx is not None
            ):
                ret_inv = self._find_skim_retarget_receiver(
                    cand=cand,
                    banned_recv_id=receiver_id,
                    frame_index=frame_index,
                    ball_xy=gx,
                    prefer_chain_window=True,
                )
                if ret_inv is not None:
                    raw_br = receiver_id
                    receiver_bbox_intermediate_raw = raw_br
                    receiver_id = ret_inv
                    recv_r_override = "chain_retargeted_from_bbox_only_intermediate"
                    bbox_guard_stat = "retargeted_receiver"
                    self._dbg_gate(
                        f"chain_rt_inv:{cand.from_player_id}:{raw_br}->{receiver_id}",
                        frame_index,
                        f"[PASS-CHAIN-RETARGET] {cand.from_player_id}->{raw_br} "
                        f"retargeted_to={receiver_id} reason=invalid_intermediate_receiver",
                    )
                else:
                    self._dbg_gate(
                        f"inv_mid_sup_rcv:{cand.from_player_id}->{receiver_id}",
                        frame_index,
                        "[PASS-INVALID-INTERMEDIATE] suppress "
                        f"{cand.from_player_id}->{receiver_id} "
                        "reason=receiver_invalid_intermediate",
                    )
                    self._print_pass_decision_suppressed(
                        frame_index=frame_index,
                        cand_from=cand.from_player_id,
                        cand_to=receiver_id,
                        reason="receiver_invalid_intermediate",
                    )
                    return None

        if cfg.pass_bbox_only_intermediate_guard_enabled:
            if self._bbox_guard_chain_suppressed(cand.from_player_id, frame_index):
                self._dbg_gate(
                    f"bbox_gate:{cand.from_player_id}->{receiver_id}:intermediate_chain",
                    frame_index,
                    "[PASS-BBOX-GATE] suppress "
                    f"{cand.from_player_id}->{receiver_id} "
                    "reason=source_bbox_only_intermediate_no_valid_touch",
                )
                self._print_pass_decision_suppressed(
                    frame_index=frame_index,
                    cand_from=cand.from_player_id,
                    cand_to=receiver_id,
                    reason="source_bbox_only_intermediate_no_valid_touch",
                )
                return None

            if cfg.pass_valid_touch_gate_enabled:
                if self._receiver_gate_skims_bbox_without_valid_touch(
                    receiver_id, cand.start_frame, frame_index
                ):
                    retarget_id: Optional[int] = None
                    if gx is not None and (
                        cfg.pass_receiver_retarget_after_bbox_only_enabled
                        or cfg.pass_chain_retarget_enabled
                    ):
                        retarget_id = self._find_skim_retarget_receiver(
                            cand=cand,
                            banned_recv_id=receiver_id,
                            frame_index=frame_index,
                            ball_xy=gx,
                            prefer_chain_window=cfg.pass_chain_retarget_enabled,
                        )
                    if retarget_id is not None:
                        raw_b = receiver_id
                        self._mark_invalid_intermediate_track(
                            raw_b, frame_index, "bbox_only_no_valid_touch"
                        )
                        receiver_bbox_intermediate_raw = (
                            raw_b
                            if receiver_bbox_intermediate_raw is None
                            else receiver_bbox_intermediate_raw
                        )
                        receiver_id = retarget_id
                        recv_r_override = (
                            "chain_retargeted_from_bbox_only_intermediate"
                            if cfg.pass_chain_retarget_enabled
                            else "retargeted_from_bbox_only_intermediate"
                        )
                        bbox_guard_stat = "retargeted_receiver"
                        tag = (
                            "[PASS-CHAIN-RETARGET]"
                            if cfg.pass_chain_retarget_enabled
                            else "[PASS-RECEIVER-RETARGET]"
                        )
                        self._dbg_gate(
                            f"skim_rt:{cand.from_player_id}->{raw_b}->{receiver_id}",
                            frame_index,
                            f"{tag} {cand.from_player_id}->{raw_b} retargeted_to="
                            f"{receiver_id} reason=bbox_only_intermediate",
                        )
                    else:
                        self._mark_invalid_intermediate_track(
                            receiver_id, frame_index, "bbox_only_no_valid_touch"
                        )
                        self._dbg_gate(
                            f"bbox_gate:{cand.from_player_id}->{receiver_id}:rcv_bbox_only",
                            frame_index,
                            "[PASS-BBOX-GATE] suppress "
                            f"{cand.from_player_id}->{receiver_id} "
                            "reason=receiver_bbox_only_no_valid_touch",
                        )
                        self._print_pass_decision_suppressed(
                            frame_index=frame_index,
                            cand_from=cand.from_player_id,
                            cand_to=receiver_id,
                            reason="receiver_bbox_only_no_valid_touch",
                        )
                        self._suppressed_bbox_receiver_chain.append(
                            (frame_index, receiver_id)
                        )
                        return None
            elif self._bbox_receiver_crossing_should_suppress(
                recv_id=receiver_id,
                cand=cand,
                frame_index=frame_index,
            ):
                self._dbg_gate(
                    f"bbox_guard:{cand.from_player_id}->{receiver_id}:receiver_bbox_only",
                    frame_index,
                    "[PASS-BBOX-GUARD] suppress "
                    f"{cand.from_player_id}->{receiver_id} reason=receiver_bbox_only",
                )
                self._print_pass_decision_suppressed(
                    frame_index=frame_index,
                    cand_from=cand.from_player_id,
                    cand_to=receiver_id,
                    reason="receiver_bbox_only",
                )
                self._suppressed_bbox_receiver_chain.append(
                    (frame_index, receiver_id)
                )
                return None

        recv_contact_label = self._receiver_contact_label_for_emit(
            receiver_id, cand, frame_index
        )

        source_retained, source_retained_rst = self._source_retains_control_guard_fsm(
            cand, receiver_id, frame_index, possession
        )
        if source_retained:
            status = source_retained_rst or "source_retains_control receiver_unconfirmed"
            self._print_pass_decision_suppressed(
                frame_index=frame_index,
                cand_from=cand.from_player_id,
                cand_to=receiver_id,
                reason=status,
            )
            return None

        dribble_blocked, dribble_rst = self._dribble_overlap_guard_fsm(
            cand, receiver_id, frame_index, gx
        )
        if dribble_blocked:
            status = dribble_rst or "dribble_overlap_not_pass"
            self._print_pass_decision_suppressed(
                frame_index=frame_index,
                cand_from=cand.from_player_id,
                cand_to=receiver_id,
                reason=status,
            )
            return None

        to_team_id = self._resolver_team(possession, receiver_id, track_to_team)

        evt_type = self._classify(
            cand.from_team_id,
            to_team_id,
            cfg.require_same_team_for_completed,
            cfg.emit_interceptions,
        )
        baseline_evt_type = evt_type

        raw_from = cand.from_team_id
        raw_to = to_team_id
        from_stable = raw_from is not None and self._team_label_stable_for_track(
            cand.from_player_id, raw_from
        )
        to_stable = raw_to is not None and self._team_label_stable_for_track(
            receiver_id, raw_to
        )
        team_stable = (raw_from is None or from_stable) and (raw_to is None or to_stable)

        debug_reason_base: Optional[str] = None
        if cfg.pass_unstable_team_as_unknown:
            bits: list[str] = []
            if raw_from is not None and not from_stable:
                bits.append("unstable_from_team")
            if raw_to is not None and not to_stable:
                bits.append("unstable_to_team")
            if bits:
                evt_type = "unknown_pass"
                debug_reason_base = ";".join(bits)

        conf = self._compute_pass_confidence(
            baseline_evt_type=baseline_evt_type,
            evt_type=evt_type,
            duration_frames=flight_len,
            raw_from=raw_from,
            raw_to=raw_to,
            from_stable=from_stable,
            to_stable=to_stable,
            ball_start=cand.ball_anchor_xy,
            ball_end=possession.ball_xy,
        )

        debug_reason = debug_reason_base
        if one_touch:
            debug_reason = (
                "one_touch"
                if debug_reason_base is None
                else f"{debug_reason_base};one_touch"
            )

        rcv_sf = (
            self._one_touch_recv_confirm_start_frame
            if one_touch
            else self._recv_confirm_start_frame
        )
        rcv_nf = (
            self._one_touch_recv_streak if one_touch else self._recv_streak
        )
        if rcv_sf is None:
            rcv_sf = frame_index

        recv_r = receiver_selection_reason
        recv_scr = receiver_score
        if recv_r is None and not one_touch:
            recv_r = self._flight_recv_pick_reason or "fsm_nearest_pick"
            if self._flight_recv_pick_score >= 0.0:
                recv_scr = round(self._flight_recv_pick_score, 6)
        if recv_r_override is not None:
            recv_r = recv_r_override
        if recv_r is None:
            recv_r = "completed_path"
        disp = None
        if cand.ball_anchor_xy is not None and possession.ball_xy is not None:
            disp = _dist_px(cand.ball_anchor_xy, possession.ball_xy)

        dur_sec = None
        ts0, ts1 = cand.start_time_sec, timestamp_sec
        if ts0 is not None and ts1 is not None:
            dur_sec = ts1 - ts0

        event = PassEvent(
            event_id=self._next_event_id,
            event_type=evt_type,
            start_frame=cand.start_frame,
            end_frame=frame_index,
            start_time_sec=cand.start_time_sec,
            end_time_sec=timestamp_sec,
            from_player_id=cand.from_player_id,
            to_player_id=receiver_id,
            from_team_id=cand.from_team_id,
            to_team_id=to_team_id,
            duration_frames=flight_len,
            duration_sec=dur_sec,
            confidence=conf,
            ball_start_xy=cand.ball_anchor_xy,
            ball_end_xy=possession.ball_xy,
            raw_from_team_id=raw_from,
            raw_to_team_id=raw_to,
            team_stable=team_stable,
            debug_reason=debug_reason,
            release_frame=cand.release_frame,
            receiver_confirm_start_frame=rcv_sf,
            receiver_confirm_frames=rcv_nf,
            ball_displacement_px=disp,
            ignored_transient_contacts=self._ignored_transient_episodes,
            pass_fsm_state="completed_one_touch" if one_touch else "completed",
            one_touch=one_touch,
            one_touch_evidence=one_touch_evidence if one_touch else None,
            post_receive_blocked=one_touch,
            receiver_selection_reason=recv_r,
            receiver_score=recv_scr,
            dribble_guard_status="passed_guard",
            receiver_contact_type=recv_contact_label,
            bbox_only_guard_status=bbox_guard_stat,
            receiver_bbox_intermediate_raw=receiver_bbox_intermediate_raw,
        )
        if self._short_interception_duel_should_suppress(event):
            self._print_pass_decision_suppressed(
                frame_index=frame_index,
                cand_from=cand.from_player_id,
                cand_to=receiver_id,
                reason="short_interception_duel_not_pass",
            )
            return None
        self._try_replace_weak_primary_source_via_valid_touch(
            event, cand, track_to_team, frame_index
        )
        rt = self._same_team_retarget_after_source_recovery(
            event=event,
            cand=cand,
            receiver_id=receiver_id,
            frame_index=frame_index,
            tracks=tracks,
            ball_xy=gx,
            track_to_team=track_to_team,
        )
        if rt is not None:
            receiver_id = rt
        self._next_event_id += 1
        self._apply_emit_shared_state(
            frame_index, cand.from_player_id, receiver_id, clear_fsm_flight=True
        )
        self._ignored_transient_episodes = 0
        event = self._finalize_emitted_pass(event)
        event = self._apply_pass_source_reliability(
            event, cand, track_to_team=track_to_team
        )
        event = self._apply_none_source_recovery(
            event, frame_index, track_to_team, cand
        )
        self._apply_pass_event_aliases(event, frame_index, track_to_team=track_to_team)
        ball_xy_tel = gx
        if ball_xy_tel is None and possession.ball_xy is not None:
            ball_xy_tel = (float(possession.ball_xy[0]), float(possession.ball_xy[1]))
        self._attach_pass_decision_telemetry(
            event,
            tracks=tracks,
            ball_xy=ball_xy_tel,
            track_to_team=track_to_team,
            frame_index=frame_index,
            emit_path_base="primary_fsm",
            flight_start_frame=cand.start_frame,
            flight_end_frame=frame_index,
            release_anchor_xy=cand.ball_anchor_xy,
        )
        self._remember_pass_emission(event)
        return event

    def _apply_emit_shared_state(
        self,
        frame_index: int,
        from_tid: Optional[int],
        receiver_id: int,
        *,
        clear_fsm_flight: bool,
    ) -> None:
        cfg = self._config
        if clear_fsm_flight:
            self._clear_flight()
            self._clear_one_touch_accumulator()
        self._cooldown_until_frame = frame_index + cfg.cooldown_frames
        self._away_run = 0
        self._away_anchor = None
        self._last_event_end_frame = frame_index
        self._last_receiver_id = receiver_id
        self._last_event_from_player_id = from_tid
        self._last_event_pair = (from_tid, receiver_id)
        if cfg.pass_post_receive_require_reconfirm:
            self._receiver_release_blocked_until[receiver_id] = (
                frame_index + cfg.pass_post_receive_settle_frames
            )
        self._release_allow_logged_tid = None
        self._co_owner = receiver_id
        self._run_tid = receiver_id
        self._same_run = 1
        self._alt_tid = None
        self._alt_run = 0

    def _remember_pass_emission(self, event: PassEvent) -> None:
        if (
            self._last_event_end_frame == event.end_frame
            and self._last_receiver_id == event.to_player_id
        ):
            self._last_event_from_player_id = event.from_player_id
            self._last_event_pair = (event.from_player_id, event.to_player_id)
        self._pass_emission_log.append(event)

    def _forget_last_remembered_event(self, event: PassEvent) -> None:
        if self._pass_emission_log and self._pass_emission_log[-1] is event:
            self._pass_emission_log.pop()

    def _event_receiver_is_weak_intermediate_candidate(self, event: PassEvent) -> bool:
        if event.pass_fsm_state != "completed":
            return False
        if event.one_touch:
            return False
        if event.from_player_id is None or event.to_player_id is None:
            return False
        rlab = (event.receiver_contact_type or "").lower()
        if rlab in ("unknown", "bbox_overlap_skim"):
            return True
        if "bbox" in rlab:
            return True
        return False

    def _pending_weak_chain_expired(self, frame_index: int) -> bool:
        ev = self._pending_weak_chain_event
        if ev is None:
            return False
        cfg = self._config
        win = max(
            cfg.pass_bbox_only_chain_window_frames,
            cfg.pass_chain_retarget_window_frames,
        )
        return frame_index - ev.end_frame > win

    def _release_expired_pending_weak_chain(
        self, frame_index: int, events: list[PassEvent]
    ) -> None:
        if not self._pending_weak_chain_expired(frame_index):
            return
        ev = self._pending_weak_chain_event
        self._pending_weak_chain_event = None
        if ev is not None:
            events.append(ev)

    def _events_form_weak_intermediate_chain(
        self, first: PassEvent, second: PassEvent
    ) -> bool:
        cfg = self._config
        if not cfg.pass_bbox_only_intermediate_guard_enabled:
            return False
        if first.to_player_id is None or second.from_player_id is None:
            return False
        if first.to_player_id != second.from_player_id:
            return False
        if second.to_player_id == first.from_player_id:
            return False
        if second.receiver_selection_reason == "owner_switch_handoff":
            if second.duration_frames > 3:
                return False
        gap = second.start_frame - first.end_frame
        if gap < 0:
            return False
        if gap > cfg.pass_bbox_only_chain_window_frames:
            return False
        if not self._event_receiver_is_weak_intermediate_candidate(first):
            return False
        if second.one_touch:
            return True
        if first.duration_frames > cfg.pass_bbox_only_chain_window_frames:
            return False
        if gap > min(6, cfg.pass_bbox_only_chain_window_frames):
            second_recv = (second.receiver_contact_type or "").lower()
            second_recv_weak = (
                second_recv in ("unknown", "bbox_overlap_skim") or "bbox" in second_recv
            )
            if not second_recv_weak:
                return False
        rlab = (second.source_contact_type or "").lower()
        return rlab in ("unknown", "stable_owner", "")

    def _retarget_pending_weak_chain(
        self, first: PassEvent, second: PassEvent, frame_index: int
    ) -> PassEvent:
        cfg = self._config
        intermediate = first.to_player_id
        old_end = first.end_frame
        first.to_player_id = second.to_player_id
        first.to_team_id = second.to_team_id
        first.raw_to_team_id = second.raw_to_team_id
        first.raw_to_player_id = second.raw_to_player_id
        if first.canonical_from_player_id is not None:
            raw_from = first.from_player_id
            first.from_player_id = first.canonical_from_player_id
            first.canonical_from_player_id = None
            reason_alias = f"chain_source_alias_output:{raw_from}->{first.from_player_id}"
            first.id_alias_reason = (
                reason_alias
                if first.id_alias_reason is None
                else f"{first.id_alias_reason};{reason_alias}"
            )
        first.end_frame = second.end_frame
        first.end_time_sec = second.end_time_sec
        first.duration_frames = max(1, first.end_frame - first.start_frame + 1)
        if first.start_time_sec is not None and first.end_time_sec is not None:
            first.duration_sec = first.end_time_sec - first.start_time_sec
        first.ball_end_xy = second.ball_end_xy
        first.receiver_confirm_start_frame = second.receiver_confirm_start_frame
        first.receiver_confirm_frames = second.receiver_confirm_frames
        first.receiver_contact_type = second.receiver_contact_type
        first.receiver_selection_reason = "weak_intermediate_chain_retargeted"
        first.receiver_score = second.receiver_score
        first.receiver_bbox_intermediate_raw = intermediate
        first.bbox_only_guard_status = "weak_intermediate_chain_retargeted"
        first.bbox_guard_checked = True
        first.bbox_guard_result = "retargeted"
        first.confidence = min(first.confidence, second.confidence, 0.82)
        first.event_type = self._classify(
            first.from_team_id,
            first.to_team_id,
            cfg.require_same_team_for_completed,
            cfg.emit_interceptions,
        )
        reason = f"weak_intermediate_chain:{intermediate}:{old_end}->{second.end_frame}"
        first.debug_reason = (
            reason if not first.debug_reason else f"{first.debug_reason};{reason}"
        )
        first.emit_path = "primary_fsm+receiver_retarget"
        first.receiver_contact_decision = "retargeted_valid_touch"
        first.invalid_intermediate_status = "weak_intermediate_chain_retargeted"
        self._mark_invalid_intermediate_track(
            intermediate, frame_index, "weak_intermediate_chain"
        )
        self._dbg_gate(
            f"weak_chain:{first.from_player_id}->{intermediate}->{first.to_player_id}",
            frame_index,
            "[PASS-WEAK-CHAIN] retarget "
            f"{first.from_player_id}->{intermediate} to {first.from_player_id}->{first.to_player_id}",
        )
        return first

    def _stage_or_emit_event(
        self, event: PassEvent, frame_index: int
    ) -> list[PassEvent]:
        out: list[PassEvent] = []
        pending = self._pending_weak_chain_event
        if pending is not None:
            if self._events_form_weak_intermediate_chain(pending, event):
                self._forget_last_remembered_event(event)
                self._pending_weak_chain_event = None
                out.append(self._retarget_pending_weak_chain(pending, event, frame_index))
                return out
            out.append(pending)
            self._pending_weak_chain_event = None

        if self._event_receiver_is_weak_intermediate_candidate(event):
            self._pending_weak_chain_event = event
            return out

        out.append(event)
        return out

    def _snap_visible_frames_count(self, tid: int, lo_f: int, hi_f: int) -> int:
        n = 0
        for snap in self._snap_hist:
            fi = snap.frame_index
            if fi < lo_f or fi > hi_f or snap.ball_xy is None:
                continue
            if snap.feet.get(tid) is not None:
                n += 1
        return n

    def _snap_min_footpoint_distance(
        self, tid: int, lo_f: int, hi_f: int
    ) -> Optional[float]:
        md: Optional[float] = None
        for snap in self._snap_hist:
            fi = snap.frame_index
            if fi < lo_f or fi > hi_f or snap.ball_xy is None:
                continue
            foot = snap.feet.get(tid)
            if foot is None:
                continue
            d = _dist_px(foot, snap.ball_xy)
            md = d if md is None else min(md, d)
        return md

    def _collect_top_receiver_candidates_debug(
        self,
        *,
        tracks: Optional["FrameTracks"],
        ball_xy: Optional[tuple[float, float]],
        exclude: int,
        frame_index: int,
        release_anchor: Optional[tuple[float, float]],
        track_to_team: dict[int, int],
        top_n: int = 5,
    ) -> list[dict[str, Any]]:
        if tracks is None or ball_xy is None:
            return []
        cfg = self._config
        raw_scored: list[tuple[int, float, str, int, int, float]] = []
        from VisionEngine.schemas.schema import ObjectRole

        search_r = float(cfg.pass_receiver_control_radius_px) * 1.45 + 40.0
        for inst in tracks.instances:
            if inst.role is not ObjectRole.PLAYER:
                continue
            tid = inst.track_id
            if tid < 0 or tid == exclude:
                continue
            bbox = tuple(float(x) for x in inst.xyxy)
            ft = foot_xy(bbox)
            d_near = _dist_px(ft, ball_xy)
            if d_near > search_r:
                continue
            d_foot, lower_ok, valid_ctrl, bbox_only_flag = classify_touch_features(
                cfg, bbox, ball_xy
            )
            qual_touch = lower_ok or valid_ctrl
            if cfg.pass_invalid_intermediate_memory_enabled and self._invalid_intermediate_active(
                tid, frame_index
            ):
                continue
            tm = track_to_team.get(tid)
            stab = tm is None or self._team_label_stable_for_track(tid, tm)
            sc = self._score_receiver_candidate_one_frame(
                track_id=tid,
                bbox=bbox,
                ball_xy=tuple(ball_xy),
                exclude=exclude,
                release_anchor=release_anchor,
                ft=ft,
                d_foot=d_foot,
                lower_body_valid=lower_ok,
                bbox_overlap_only=bbox_only_flag,
                team_id=tm,
                track_to_team=track_to_team,
                team_stable=stab,
            )
            if bbox_only_flag and not qual_touch:
                cj = "bbox_only"
            elif qual_touch:
                cj = "valid_touch"
            else:
                cj = "unknown"
            vf = self._gate_valid_snap_count(tid, frame_index - 24, frame_index)
            bf = self._gate_bbox_only_snap_count(tid, frame_index - 24, frame_index)
            raw_scored.append((tid, sc, cj, vf, bf, d_foot))
        raw_scored.sort(key=lambda z: z[1], reverse=True)
        out: list[dict[str, Any]] = []
        for tid, sc, cj, vf, bf, d_foot in raw_scored[:top_n]:
            out.append(
                {
                    "track_id": tid,
                    "score": round(sc, 4),
                    "contact_type": cj,
                    "valid_touch_frames": vf,
                    "bbox_only_frames": bf,
                    "min_footpoint_distance_px": round(d_foot, 2),
                }
            )
        return out

    def _derive_emit_path_string(self, event: PassEvent, base: str) -> str:
        b = base or "unknown"
        parts = [b]
        if event.receiver_bbox_intermediate_raw is not None or (
            event.receiver_selection_reason
            and "retarget" in (event.receiver_selection_reason or "").lower()
        ):
            parts.append("receiver_retarget")
        if (
            event.source_recovery_method
            or event.source_reliability == "recovered"
        ):
            parts.append("source_recovery")
        return "+".join(parts) if len(parts) > 1 else b

    def _map_bbox_guard_telemetry(self, status: Optional[str]) -> tuple[bool, str]:
        cfg = self._config
        checked = bool(
            cfg.pass_bbox_only_intermediate_guard_enabled
            or cfg.pass_valid_touch_gate_enabled
        )
        if status is None:
            return checked, "no_valid_touch_data"
        if status == "disabled":
            return False, "not_checked"
        if status.startswith("skipped"):
            return False, "not_checked"
        if "retarget" in status:
            return True, "retargeted"
        if status == "passed":
            return True, "passed"
        return True, "passed"

    def _attach_pass_decision_telemetry(
        self,
        event: PassEvent,
        *,
        tracks: Optional["FrameTracks"],
        ball_xy: Optional[tuple[float, float]],
        track_to_team: dict[int, int],
        frame_index: int,
        emit_path_base: str,
        flight_start_frame: int,
        flight_end_frame: int,
        release_anchor_xy: Optional[tuple[float, float]],
    ) -> None:
        cfg = self._config
        rf = event.release_frame
        if rf is None:
            rf = event.start_frame
        src_lo = max(0, rf - cfg.pass_source_reliability_lookback_frames)
        src_hi = rf + cfg.pass_source_reliability_lookahead_frames

        end_f = event.end_frame
        recv_span = max(16, cfg.pass_receiver_confirm_frames * 3)
        recv_lo = max(0, end_f - recv_span)
        recv_hi = end_f

        frm = event.from_player_id
        to_id = event.to_player_id

        if frm is not None:
            event.source_visible_frames_near_release = self._snap_visible_frames_count(
                frm, src_lo, src_hi
            )
            event.source_min_footpoint_distance_px = self._snap_min_footpoint_distance(
                frm, src_lo, src_hi
            )
            event.source_valid_touch_frames = self._gate_valid_touch_frames_episode(
                frm, flight_start_frame, flight_end_frame
            )
            event.source_bbox_only_frames = self._gate_bbox_only_frames_episode(
                frm, flight_start_frame, flight_end_frame
            )

        if to_id is not None:
            event.receiver_visible_frames_near_endpoint = (
                self._snap_visible_frames_count(to_id, recv_lo, recv_hi)
            )
            event.receiver_min_footpoint_distance_px = self._snap_min_footpoint_distance(
                to_id, recv_lo, recv_hi
            )
            event.receiver_valid_touch_frames = self._gate_valid_touch_frames_episode(
                to_id, flight_start_frame, flight_end_frame
            )
            event.receiver_bbox_only_frames = self._gate_bbox_only_frames_episode(
                to_id, flight_start_frame, flight_end_frame
            )

        event.emit_path = self._derive_emit_path_string(event, emit_path_base)

        s_reason_bits = []
        if event.source_reliability_reason:
            s_reason_bits.append(event.source_reliability_reason)
        if event.source_recovery_method:
            s_reason_bits.append(f"recovery={event.source_recovery_method}")
        event.source_decision_reason = (
            ";".join(s_reason_bits) if s_reason_bits else None
        )
        event.receiver_decision_reason = event.receiver_selection_reason

        if event.source_recovery_method or event.source_reliability == "recovered":
            event.source_contact_type = "recovered_valid_touch"
        elif frm is not None:
            sv = event.source_valid_touch_frames or 0
            sb = event.source_bbox_only_frames or 0
            if sv >= cfg.pass_valid_touch_min_frames:
                event.source_contact_type = "valid_touch"
            elif sb >= 1 and sv == 0:
                event.source_contact_type = "bbox_only"
            elif event.source_visible_frames_near_release:
                event.source_contact_type = "stable_owner"
            else:
                event.source_contact_type = "unknown"
        else:
            event.source_contact_type = "unknown"

        rlab = event.receiver_contact_type or ""
        if event.receiver_bbox_intermediate_raw is not None:
            event.receiver_contact_decision = "retargeted_valid_touch"
        elif "valid_touch" in rlab or rlab.startswith("touch_episode"):
            event.receiver_contact_decision = "valid_touch"
        elif "bbox" in rlab.lower():
            event.receiver_contact_decision = "bbox_only"
        elif (event.receiver_confirm_frames or 0) >= cfg.pass_receiver_confirm_frames:
            event.receiver_contact_decision = "confirmed_receiver"
        else:
            event.receiver_contact_decision = "unknown"

        event.candidate_receivers_debug = self._collect_top_receiver_candidates_debug(
            tracks=tracks,
            ball_xy=ball_xy,
            exclude=frm if frm is not None else -1,
            frame_index=frame_index,
            release_anchor=release_anchor_xy,
            track_to_team=track_to_team,
        )

        chk, bres = self._map_bbox_guard_telemetry(event.bbox_only_guard_status)
        event.bbox_guard_checked = chk
        event.bbox_guard_result = bres

        event.invalid_intermediate_checked = bool(
            cfg.pass_invalid_intermediate_memory_enabled
        )
        if not event.invalid_intermediate_checked:
            event.invalid_intermediate_result = "not_checked"
        elif event.invalid_intermediate_status:
            event.invalid_intermediate_result = event.invalid_intermediate_status
        else:
            event.invalid_intermediate_result = "ok"

        recover_checked = (
            cfg.pass_none_source_recovery_enabled
            or cfg.pass_source_recover_from_valid_touch_enabled
        )
        event.source_recovery_checked = recover_checked
        if not recover_checked:
            event.source_recovery_result = "not_checked"
        elif (
            event.source_recovery_method
            or event.source_reliability == "recovered"
        ):
            event.source_recovery_result = "recovered"
        elif event.from_player_id is None:
            event.source_recovery_result = "still_none"
        else:
            event.source_recovery_result = "not_needed"

        if cfg.pass_debug:
            self._print_pass_decision_block(event)

    def _gate_valid_touch_frames_episode(
        self, tid: int, lo_f: int, hi_f: int
    ) -> int:
        return self._gate_valid_snap_count(tid, lo_f, hi_f)

    def _gate_bbox_only_frames_episode(
        self, tid: int, lo_f: int, hi_f: int
    ) -> int:
        return self._gate_bbox_only_snap_count(tid, lo_f, hi_f)

    def _print_pass_decision_block(self, event: PassEvent) -> None:
        if not self._config.pass_debug:
            return
        ff = event.from_player_id
        tt = event.to_player_id
        fs = "None" if ff is None else str(ff)
        ts = "None" if tt is None else str(tt)
        cands = event.candidate_receivers_debug or []
        cand_s = ", ".join(
            f"{c['track_id']}:{c['contact_type']} score={c['score']:.2f}"
            for c in cands[:5]
        )
        self._dbg(
            f"[PASS-DECISION] frame {event.start_frame}-{event.end_frame} "
            f"{fs}->{ts} emit_path={event.emit_path or '?'}\n"
            f"  source: type={event.source_contact_type or '?'} "
            f"dist={event.source_min_footpoint_distance_px} "
            f"valid_frames={event.source_valid_touch_frames} "
            f"bbox_only_frames={event.source_bbox_only_frames}\n"
            f"  receiver: type={event.receiver_contact_decision or '?'} "
            f"dist={event.receiver_min_footpoint_distance_px} "
            f"valid_frames={event.receiver_valid_touch_frames} "
            f"bbox_only_frames={event.receiver_bbox_only_frames}\n"
            f"  candidates: {cand_s or '(none)'}\n"
            f"  guards: bbox={event.bbox_guard_result} "
            f"invalid_intermediate={event.invalid_intermediate_result} "
            f"recovery={event.source_recovery_result}"
        )

    def _print_pass_decision_suppressed(
        self,
        *,
        frame_index: int,
        cand_from: Optional[int],
        cand_to: Optional[int],
        reason: str,
        extra: str = "",
    ) -> None:
        cfg = self._config
        if not cfg.pass_debug:
            return
        fs = "None" if cand_from is None else str(cand_from)
        ts = "None" if cand_to is None else str(cand_to)
        msg = (
            f"[PASS-DECISION] suppressed frame={frame_index} {fs}->{ts} "
            f"reason={reason}"
        )
        if extra:
            msg = f"{msg} {extra}"
        self._dbg(msg)

    def _record_valid_touch_frame(
        self,
        frame_index: int,
        ball_xy: Optional[tuple[float, float]],
        tracks: Optional["FrameTracks"],
        track_to_team: dict[int, int],
    ) -> None:
        cfg = self._config
        if (
            not cfg.pass_valid_touch_gate_enabled
            and not cfg.pass_source_recover_from_valid_touch_enabled
            and not cfg.pass_receiver_retarget_after_bbox_only_enabled
            and not cfg.pass_short_valid_touch_fallback_enabled
            and not cfg.pass_none_source_recovery_enabled
            and not cfg.pass_invalid_intermediate_memory_enabled
            and not cfg.pass_chain_retarget_enabled
        ):
            return

        from VisionEngine.schemas.schema import ObjectRole

        vmap: dict[int, _ValidTouchHistTrack] = {}
        if tracks is not None and ball_xy is not None:
            bxy = (float(ball_xy[0]), float(ball_xy[1]))
            for inst in tracks.instances:
                if inst.role is not ObjectRole.PLAYER:
                    continue
                tid = inst.track_id
                if tid < 0:
                    continue
                bbox = tuple(float(x) for x in inst.xyxy)
                v_ok, bbox_only, d_cf = classify_valid_touch_gate(cfg, bbox, bxy)
                vmap[tid] = _ValidTouchHistTrack(
                    valid_touch=bool(v_ok),
                    bbox_only=bool(bbox_only),
                    d_foot=float(d_cf),
                )
                if cfg.pass_debug and bbox_only:
                    self._dbg_gate(
                        f"bbox_only_snap:{tid}:{frame_index}",
                        frame_index,
                        f"[PASS-VALID-TOUCH] bbox_only track={tid} frame={frame_index}",
                    )

        self._valid_touch_snap_hist.append(
            _ValidTouchHistFrame(
                frame_index=frame_index,
                ball_xy=tuple(ball_xy) if ball_xy is not None else None,
                tracks=dict(vmap),
            )
        )

        touching_vt = {tid for tid, ht in vmap.items() if ht.valid_touch}

        for tid in list(self._valid_touch_open.keys()):
            if tid in touching_vt:
                continue
            op = self._valid_touch_open[tid]
            op.misses += 1
            if op.misses > cfg.pass_valid_touch_max_gap_frames:
                self._valid_touch_finalize_open_episode(tid, track_to_team)

        for tid in touching_vt:
            ht = vmap[tid]
            team_snap = track_to_team.get(tid)
            if tid not in self._valid_touch_open:
                self._valid_touch_open[tid] = _ValidTouchOpenEpisode(
                    player_id=tid,
                    team_id_snap=team_snap,
                    start_frame=frame_index,
                    end_frame=frame_index,
                    misses=0,
                    contact_frames=1,
                    min_d=ht.d_foot,
                    first_ball_xy=(
                        tuple(ball_xy)
                        if ball_xy is not None
                        else None
                    ),
                    last_ball_xy=(
                        tuple(ball_xy)
                        if ball_xy is not None
                        else None
                    ),
                )
            else:
                o = self._valid_touch_open[tid]
                o.misses = 0
                o.end_frame = frame_index
                o.contact_frames += 1
                o.min_d = min(o.min_d, ht.d_foot)
                if ball_xy is not None:
                    o.last_ball_xy = tuple(ball_xy)
                if o.team_id_snap is None and team_snap is not None:
                    o.team_id_snap = team_snap

        if cfg.pass_invalid_intermediate_memory_enabled:
            for tid_v in touching_vt:
                self._invalid_intermediate_until.pop(tid_v, None)

    def _valid_touch_finalize_open_episode(
        self, tid: int, track_to_team: dict[int, int]
    ) -> None:
        o = self._valid_touch_open.pop(tid, None)
        if o is None:
            return
        team_id = track_to_team.get(tid)
        if team_id is None:
            team_id = o.team_id_snap
        ep = _ValidTouchEpisode(
            player_id=tid,
            team_id=team_id,
            start_frame=o.start_frame,
            end_frame=o.end_frame,
            contact_frames=o.contact_frames,
            min_distance_to_ball=float(o.min_d),
            first_ball_xy=o.first_ball_xy,
            last_ball_xy=o.last_ball_xy,
        )
        self._valid_touch_finished.append(ep)

    def _gate_valid_snap_count(self, player_id: int, lo_f: int, hi_f: int) -> int:
        n = 0
        for snap in self._valid_touch_snap_hist:
            fi = snap.frame_index
            if fi < lo_f or fi > hi_f:
                continue
            t = snap.tracks.get(player_id)
            if t is not None and t.valid_touch:
                n += 1
        return n

    def _gate_bbox_only_snap_count_for_threshold(
        self,
        player_id: int,
        lo_f: int,
        hi_f: int,
        min_foot_px: float,
    ) -> int:
        n = 0
        for snap in self._valid_touch_snap_hist:
            fi = snap.frame_index
            if fi < lo_f or fi > hi_f:
                continue
            t = snap.tracks.get(player_id)
            if t is not None and t.bbox_only and t.d_foot > min_foot_px:
                n += 1
        return n

    def _gate_bbox_only_snap_count(self, player_id: int, lo_f: int, hi_f: int) -> int:
        cfg = self._config
        thr = cfg.pass_valid_touch_bbox_only_footpoint_min_px
        n = 0
        for snap in self._valid_touch_snap_hist:
            fi = snap.frame_index
            if fi < lo_f or fi > hi_f:
                continue
            t = snap.tracks.get(player_id)
            if t is not None and t.bbox_only and t.d_foot > thr:
                n += 1
        return n

    def _valid_touch_qualifies_receiver(
        self, recv_id: int, lo_f: int, hi_f: int
    ) -> bool:
        cfg = self._config
        min_f = cfg.pass_valid_touch_min_frames
        for ep in self._valid_touch_finished:
            if ep.player_id != recv_id:
                continue
            if ep.contact_frames < min_f:
                continue
            ol = max(ep.start_frame, lo_f)
            oe = min(ep.end_frame, hi_f)
            if ol <= oe:
                return True
        ep_o = self._valid_touch_open.get(recv_id)
        if ep_o is not None and ep_o.contact_frames >= min_f:
            ol = max(ep_o.start_frame, lo_f)
            oe = min(ep_o.end_frame, hi_f)
            if ol <= oe:
                return True
        return False

    def _receiver_gate_skims_bbox_without_valid_touch(
        self, recv_id: int, lo_f: int, hi_f: int
    ) -> bool:
        cfg = self._config
        if cfg.pass_strict_bbox_only_classification_enabled:
            lb_ct, _vt_ct, _bo_far = self._touch_hist_recv_metrics(
                recv_id, lo_f, hi_f
            )
            if cfg.pass_invalid_intermediate_require_no_lower_body_touch and lb_ct >= 1:
                return False
            vf = self._gate_valid_snap_count(recv_id, lo_f, hi_f)
            if vf > cfg.pass_invalid_intermediate_max_valid_touch_frames:
                return False
            thr = cfg.pass_invalid_intermediate_min_footpoint_distance_px
            b = self._gate_bbox_only_snap_count_for_threshold(
                recv_id, lo_f, hi_f, thr
            )
            return b >= 1
        v = self._gate_valid_snap_count(recv_id, lo_f, hi_f)
        b = self._gate_bbox_only_snap_count(recv_id, lo_f, hi_f)
        return b >= 1 and v == 0

    def _find_skim_retarget_receiver(
        self,
        *,
        cand: _FlightCandidate,
        banned_recv_id: int,
        frame_index: int,
        ball_xy: tuple[float, float],
        prefer_chain_window: bool,
    ) -> Optional[int]:
        cfg = self._config
        chain_ok = cfg.pass_chain_retarget_enabled and prefer_chain_window
        recv_ok = cfg.pass_receiver_retarget_after_bbox_only_enabled
        if not chain_ok and not recv_ok:
            return None

        if chain_ok:
            lookahead = cfg.pass_chain_retarget_window_frames
            max_dpx = cfg.pass_chain_retarget_max_receiver_distance_px
            req_lb = cfg.pass_chain_retarget_require_valid_receiver_touch
        else:
            lookahead = cfg.pass_receiver_retarget_lookahead_frames
            max_dpx = cfg.pass_receiver_retarget_max_distance_px
            req_lb = False

        min_f = cfg.pass_valid_touch_min_frames
        lo = cand.release_frame
        hi = frame_index + lookahead

        best_id: Optional[int] = None
        best_metric = float("inf")

        def score_ep(ep: _ValidTouchEpisode) -> float:
            if ep.first_ball_xy is None:
                return float("inf")
            d = _dist_px(ep.first_ball_xy, ball_xy)
            return ep.start_frame * 0.02 + d

        def accept_pid(pid: int) -> bool:
            if pid in (cand.from_player_id, banned_recv_id):
                return False
            if cfg.pass_invalid_intermediate_memory_enabled and self._invalid_intermediate_active(
                pid, frame_index
            ):
                return False
            return True

        for ep in self._valid_touch_finished:
            if not accept_pid(ep.player_id):
                continue
            if ep.contact_frames < min_f:
                continue
            if ep.start_frame < lo or ep.start_frame > hi:
                continue
            if req_lb:
                lb_ct, _, _ = self._touch_hist_recv_metrics(
                    ep.player_id, ep.start_frame, ep.end_frame
                )
                if lb_ct < 1:
                    continue
            d = (
                float("inf")
                if ep.first_ball_xy is None
                else _dist_px(ep.first_ball_xy, ball_xy)
            )
            if d > max_dpx:
                continue
            m = score_ep(ep)
            if m < best_metric:
                best_metric = m
                best_id = ep.player_id

        for tid, eo in self._valid_touch_open.items():
            if not accept_pid(tid):
                continue
            if eo.contact_frames < min_f:
                continue
            if eo.start_frame > hi or eo.end_frame < lo:
                continue
            if req_lb:
                lb_ct, _, _ = self._touch_hist_recv_metrics(
                    tid, eo.start_frame, eo.end_frame
                )
                if lb_ct < 1:
                    continue
            ve = _ValidTouchEpisode(
                player_id=tid,
                team_id=eo.team_id_snap,
                start_frame=eo.start_frame,
                end_frame=eo.end_frame,
                contact_frames=eo.contact_frames,
                min_distance_to_ball=eo.min_d,
                first_ball_xy=eo.first_ball_xy,
                last_ball_xy=eo.last_ball_xy,
            )
            d = (
                float("inf")
                if eo.first_ball_xy is None
                else _dist_px(eo.first_ball_xy, ball_xy)
            )
            if d > max_dpx:
                continue
            m = score_ep(ve)
            if m < best_metric:
                best_metric = m
                best_id = tid

        return best_id

    def _find_retarget_valid_receiver(
        self,
        *,
        cand: _FlightCandidate,
        banned_recv_id: int,
        frame_index: int,
        ball_xy: tuple[float, float],
    ) -> Optional[int]:
        return self._find_skim_retarget_receiver(
            cand=cand,
            banned_recv_id=banned_recv_id,
            frame_index=frame_index,
            ball_xy=ball_xy,
            prefer_chain_window=False,
        )

    def _try_recover_passer_via_valid_touch(
        self,
        event: PassEvent,
        cand: _FlightCandidate,
        original_tid: int,
        track_to_team: dict[int, int],
    ) -> bool:
        cfg = self._config
        if not cfg.pass_source_recover_from_valid_touch_enabled:
            return False

        src = event.source or ""
        if (
            src == "kickoff_bootstrap_unknown_passer"
            or src == "short_valid_touch_fallback"
            or event.pass_fsm_state == "touch_to_touch_fallback"
            or event.pass_fsm_state == "short_valid_touch_fallback"
        ):
            return False

        to_id = event.to_player_id
        rf = cand.release_frame
        lb = max(0, rf - cfg.pass_source_recover_lookback_frames)
        hi_f = rf + 4

        anchor = cand.ball_anchor_xy
        max_dpx = cfg.pass_source_recover_max_distance_px

        original_tid_eff = (
            event.original_from_player_id
            if event.original_from_player_id is not None
            else event.from_player_id
        )

        candidates: list[_ValidTouchEpisode] = []
        for ep in self._valid_touch_finished:
            if cfg.pass_source_recover_require_same_primary_context and (
                original_tid_eff is not None and ep.player_id != original_tid_eff
            ):
                continue
            if ep.player_id == to_id:
                continue
            if cfg.pass_invalid_intermediate_memory_enabled and self._invalid_intermediate_active(
                ep.player_id, rf
            ):
                continue
            if ep.contact_frames < cfg.pass_valid_touch_min_frames:
                continue
            if ep.end_frame < lb or ep.start_frame > hi_f:
                continue
            if anchor is None or ep.last_ball_xy is None:
                continue
            if _dist_px(ep.last_ball_xy, anchor) > max_dpx:
                continue
            candidates.append(ep)

        for tid, eo in self._valid_touch_open.items():
            if cfg.pass_source_recover_require_same_primary_context and (
                original_tid_eff is not None and tid != original_tid_eff
            ):
                continue
            if tid == to_id:
                continue
            if cfg.pass_invalid_intermediate_memory_enabled and self._invalid_intermediate_active(
                tid, rf
            ):
                continue
            if eo.contact_frames < cfg.pass_valid_touch_min_frames:
                continue
            if eo.end_frame < lb or eo.start_frame > hi_f:
                continue
            if anchor is None or eo.last_ball_xy is None:
                continue
            if _dist_px(eo.last_ball_xy, anchor) > max_dpx:
                continue
            candidates.append(
                _ValidTouchEpisode(
                    player_id=tid,
                    team_id=track_to_team.get(tid),
                    start_frame=eo.start_frame,
                    end_frame=eo.end_frame,
                    contact_frames=eo.contact_frames,
                    min_distance_to_ball=eo.min_d,
                    first_ball_xy=eo.first_ball_xy,
                    last_ball_xy=eo.last_ball_xy,
                )
            )

        if not candidates:
            last_ev = self._pass_emission_log[-1] if self._pass_emission_log else None
            from_timeout_receiver = (
                last_ev is not None
                and last_ev.source == "flight_contact_timeout_fallback"
                and last_ev.to_player_id == cur
                and cur == self._last_receiver_id
            )
            if from_timeout_receiver:
                loose = self._recover_recent_loose_source_excluding(
                    before_frame=rf,
                    exclude_ids={cur, to_id},
                    frame_index=frame_index,
                    track_to_team=track_to_team,
                    lookback_frames=cfg.pass_none_source_recovery_lookback_frames,
                )
                if loose is not None:
                    best_pid, ft, _best_frame, best_dist, _best_ball = loose
                    _cur_vis, cur_md = self._snap_visible_ball_foot_metrics(cur, lb, hi)
                    if cur_md is not None and best_dist + 20.0 >= cur_md:
                        return False
                    old = cur
                    event.original_from_player_id = old
                    event.from_player_id = best_pid
                    event.from_team_id = ft
                    event.source_reliability = "recovered"
                    event.source_reliability_reason = (
                        "replaced_timeout_receiver_source_loose_touch"
                    )
                    event.source_recovery_method = (
                        "timeout_receiver_loose_source_replace"
                    )
                    event.confidence = min(event.confidence, 0.82)
                    event.event_type = self._classify(
                        ft,
                        event.to_team_id,
                        cfg.require_same_team_for_completed,
                        cfg.emit_interceptions,
                    )
                    cand.from_player_id = best_pid
                    cand.from_team_id = ft
                    self._dbg_gate(
                        f"source_replace_timeout_loose:{old}->{best_pid}->{to_id}:{rf}",
                        frame_index,
                        "[PASS-SOURCE-RECOVER] replaced timeout receiver source "
                        f"{old}->{to_id} as {best_pid}->{to_id}",
                    )
                    return True
            return False

        best = max(candidates, key=lambda e: (e.end_frame, e.contact_frames))

        ft = (
            track_to_team.get(best.player_id)
            if best.team_id is None
            else best.team_id
        )
        ft = ft or track_to_team.get(best.player_id)

        dest = (
            event.to_player_id if event.to_player_id is not None else "?"
        )
        event.from_player_id = best.player_id
        event.from_team_id = ft
        event.source_reliability = "recovered"
        event.source_reliability_reason = "recovered_from_recent_valid_touch"
        event.confidence = min(event.confidence, 0.82)
        if cfg.pass_debug:
            self._dbg_gate(
                f"source_recover:{original_tid}->{dest}:{best.player_id}:{rf}",
                rf,
                f"[PASS-SOURCE-RECOVER] None->{dest} "
                f"recovered_source={best.player_id} reason=recent_valid_touch",
            )

        event.event_type = self._classify(
            ft,
            event.to_team_id,
            cfg.require_same_team_for_completed,
            cfg.emit_interceptions,
        )
        return True

    def _try_replace_weak_primary_source_via_valid_touch(
        self,
        event: PassEvent,
        cand: _FlightCandidate,
        track_to_team: dict[int, int],
        frame_index: int,
    ) -> bool:
        cfg = self._config
        if not cfg.pass_source_recover_from_valid_touch_enabled:
            return False
        if event.source is not None:
            return False
        if event.pass_fsm_state not in ("completed", "completed_one_touch"):
            return False
        if event.one_touch:
            return False
        cur = event.from_player_id
        to_id = event.to_player_id
        if cur is None or to_id is None:
            return False

        rf = cand.release_frame
        lb = max(0, rf - cfg.pass_source_recover_lookback_frames)
        hi = rf + cfg.pass_source_reliability_lookahead_frames
        last_ev_pre = self._pass_emission_log[-1] if self._pass_emission_log else None
        from_timeout_receiver_pre = (
            last_ev_pre is not None
            and last_ev_pre.source == "flight_contact_timeout_fallback"
            and last_ev_pre.to_player_id == cur
            and cur == self._last_receiver_id
        )
        if from_timeout_receiver_pre:
            loose = self._recover_recent_loose_source_excluding(
                before_frame=rf,
                exclude_ids={cur, to_id},
                frame_index=frame_index,
                track_to_team=track_to_team,
                lookback_frames=cfg.pass_none_source_recovery_lookback_frames,
            )
            _cur_vis, cur_md = self._snap_visible_ball_foot_metrics(cur, lb, hi)
            if loose is not None:
                best_pid, ft, _best_frame, best_dist, _best_ball = loose
                cur_far = cur_md is None or best_dist + 20.0 < cur_md
                if cur_far:
                    old = cur
                    event.original_from_player_id = old
                    event.from_player_id = best_pid
                    event.from_team_id = ft
                    event.source_reliability = "recovered"
                    event.source_reliability_reason = (
                        "replaced_timeout_receiver_source_loose_touch"
                    )
                    event.source_recovery_method = (
                        "timeout_receiver_loose_source_replace"
                    )
                    event.confidence = min(event.confidence, 0.82)
                    event.event_type = self._classify(
                        ft,
                        event.to_team_id,
                        cfg.require_same_team_for_completed,
                        cfg.emit_interceptions,
                    )
                    cand.from_player_id = best_pid
                    cand.from_team_id = ft
                    self._dbg_gate(
                        f"source_replace_timeout_loose:{old}->{best_pid}->{to_id}:{rf}",
                        frame_index,
                        "[PASS-SOURCE-RECOVER] replaced timeout receiver source "
                        f"{old}->{to_id} as {best_pid}->{to_id}",
                    )
                    return True
        if self._gate_valid_snap_count(cur, lb, hi) >= cfg.pass_valid_touch_min_frames:
            return False

        anchor = cand.ball_anchor_xy
        if anchor is None:
            return False

        candidates: list[_ValidTouchEpisode] = []
        max_dpx = cfg.pass_source_recover_max_distance_px
        min_f = cfg.pass_valid_touch_min_frames

        preferred_source_ids = {
            tid
            for tid in (self._last_receiver_id, (self._last_event_pair or (None, None))[1])
            if tid is not None
        }

        sw = self._last_owner_switch
        if sw is not None:
            prev_owner, new_owner, switch_frame = sw
            handoff_window = max(24, cfg.pass_post_receive_settle_frames * 2)
            recent_handoff_switch = (
                self._last_event_end_frame is not None
                and switch_frame - self._last_event_end_frame <= handoff_window
            )
            allow_recent_handoff_source_replace = (
                recent_handoff_switch
                and new_owner == cur
                and prev_owner in preferred_source_ids
                and prev_owner != to_id
                and 0 <= rf - switch_frame <= cfg.pass_source_recover_lookback_frames
            )
            if (
                (not recent_handoff_switch or allow_recent_handoff_source_replace)
                and new_owner == cur
                and prev_owner in preferred_source_ids
                and prev_owner != to_id
                and 0 <= rf - switch_frame <= cfg.pass_source_recover_lookback_frames
            ):
                _vis, md = self._snap_visible_ball_foot_metrics(prev_owner, lb, hi)
                if md is not None and md <= cfg.pass_none_source_recovery_max_distance_px:
                    ft = track_to_team.get(prev_owner)
                    old = cur
                    event.original_from_player_id = old
                    event.from_player_id = prev_owner
                    event.from_team_id = ft
                    event.source_reliability = "recovered"
                    event.source_reliability_reason = (
                        "replaced_owner_switch_source_recent_receiver"
                    )
                    event.source_recovery_method = (
                        "owner_switch_previous_receiver_source_replace"
                    )
                    event.confidence = min(event.confidence, 0.82)
                    event.event_type = self._classify(
                        ft,
                        event.to_team_id,
                        cfg.require_same_team_for_completed,
                        cfg.emit_interceptions,
                    )
                    cand.from_player_id = prev_owner
                    cand.from_team_id = ft
                    self._dbg_gate(
                        f"source_replace_owner_switch:{old}->{prev_owner}->{to_id}:{rf}",
                        frame_index,
                        "[PASS-SOURCE-RECOVER] replaced owner-switch source "
                        f"{old}->{to_id} as {prev_owner}->{to_id}",
                    )
                    return True

        def add_ep(ep: _ValidTouchEpisode) -> None:
            if ep.player_id in (cur, to_id):
                return
            if ep.contact_frames < min_f:
                return
            if ep.end_frame < lb or ep.start_frame > hi:
                return
            if ep.last_ball_xy is None:
                return
            max_ep_dpx = max_dpx
            if ep.player_id in preferred_source_ids:
                max_ep_dpx = max(max_ep_dpx, cfg.pass_none_source_recovery_max_distance_px)
            if _dist_px(ep.last_ball_xy, anchor) > max_ep_dpx:
                return
            if cfg.pass_invalid_intermediate_memory_enabled and self._invalid_intermediate_active(
                ep.player_id, frame_index
            ):
                return
            candidates.append(ep)

        for ep in self._valid_touch_finished:
            add_ep(ep)
        for tid, eo in self._valid_touch_open.items():
            add_ep(
                _ValidTouchEpisode(
                    player_id=tid,
                    team_id=track_to_team.get(tid) or eo.team_id_snap,
                    start_frame=eo.start_frame,
                    end_frame=eo.end_frame,
                    contact_frames=eo.contact_frames,
                    min_distance_to_ball=eo.min_d,
                    first_ball_xy=eo.first_ball_xy,
                    last_ball_xy=eo.last_ball_xy,
                )
            )

        if not candidates:
            return False

        best = max(
            candidates,
            key=lambda e: (
                e.player_id in preferred_source_ids,
                e.contact_frames,
                e.end_frame,
            ),
        )
        best_pid = best.player_id
        canon, alias_reason = self._ephemeral_canonical_alias(
            best_pid, frame_index, track_to_team=track_to_team
        )
        if canon is not None and canon != best_pid:
            best_pid = canon
        if best_pid == cur or best_pid == to_id:
            return False

        ft = track_to_team.get(best_pid)
        if ft is None:
            ft = best.team_id
        old = cur
        event.original_from_player_id = old
        event.from_player_id = best_pid
        event.from_team_id = ft
        event.source_reliability = "recovered"
        event.source_reliability_reason = "replaced_weak_source_recent_valid_touch"
        event.source_recovery_method = "recent_valid_touch_source_replace"
        event.confidence = min(event.confidence, 0.82)
        event.event_type = self._classify(
            ft,
            event.to_team_id,
            cfg.require_same_team_for_completed,
            cfg.emit_interceptions,
        )
        cand.from_player_id = best_pid
        cand.from_team_id = ft
        if alias_reason:
            event.id_alias_reason = (
                alias_reason
                if event.id_alias_reason is None
                else f"{event.id_alias_reason};source_candidate:{alias_reason}"
            )
        self._dbg_gate(
            f"source_replace:{old}->{best_pid}->{to_id}:{rf}",
            frame_index,
            "[PASS-SOURCE-RECOVER] replaced weak source "
            f"{old}->{to_id} as {best_pid}->{to_id} reason=recent_valid_touch",
        )
        return True

    def _short_fallback_duplicate_blocks(
        self,
        start_f: int,
        end_f: int,
        from_id: Optional[int],
        to_id: int,
        frame_index: int,
        *,
        duplicate_window_frames: Optional[int] = None,
    ) -> bool:
        cfg = self._config
        w = (
            duplicate_window_frames
            if duplicate_window_frames is not None
            else cfg.pass_short_valid_touch_duplicate_window_frames
        )

        def overlap(s1: int, e1: int, s2: int, e2: int) -> bool:
            return not (e1 < s2 or s1 > e2)

        for ev in self._pass_emission_log:
            if overlap(start_f, end_f, ev.start_frame, ev.end_frame):
                self._dbg_gate(
                    f"sf_dup_ov:{from_id}->{to_id}",
                    frame_index,
                    "[PASS-SHORT-FALLBACK] skip duplicate "
                    f"{from_id}->{to_id} reason=near_existing_event",
                )
                return True
            if ev.to_player_id == to_id and abs(ev.end_frame - end_f) <= w:
                self._dbg_gate(
                    f"sf_dup_to:{from_id}->{to_id}",
                    frame_index,
                    "[PASS-SHORT-FALLBACK] skip duplicate "
                    f"{from_id}->{to_id} reason=near_existing_event",
                )
                return True
            if (
                from_id is not None
                and ev.from_player_id is not None
                and ev.from_player_id == from_id
                and ev.to_player_id == to_id
                and min(abs(ev.end_frame - end_f), abs(ev.start_frame - start_f)) <= w
            ):
                self._dbg_gate(
                    f"sf_dup_fp:{from_id}->{to_id}",
                    frame_index,
                    "[PASS-SHORT-FALLBACK] skip duplicate "
                    f"{from_id}->{to_id} reason=near_existing_event",
                )
                return True
        return False

    def _short_fallback_effect_params(self) -> tuple[int, float, int, float]:
        cfg = self._config
        if cfg.pass_short_valid_touch_fallback_recall_enabled:
            return (
                cfg.pass_short_valid_touch_fallback_max_duration_frames,
                cfg.pass_short_valid_touch_fallback_min_displacement_px,
                cfg.pass_short_valid_touch_fallback_duplicate_window_frames,
                cfg.pass_short_valid_touch_fallback_confidence_cap,
            )
        return (
            cfg.pass_short_valid_touch_max_duration_frames,
            cfg.pass_short_valid_touch_min_displacement_px,
            cfg.pass_short_valid_touch_duplicate_window_frames,
            cfg.pass_short_valid_touch_confidence_cap,
        )

    def _try_emit_short_valid_touch_fallback(
        self,
        frame_index: int,
        timestamp_sec: Optional[float],
        possession: PossessionState,
        track_to_team: dict[int, int],
        tracks: Optional["FrameTracks"] = None,
    ) -> Optional[PassEvent]:
        cfg = self._config
        if not cfg.pass_short_valid_touch_fallback_enabled:
            return None

        dur_max, disp_min, dup_w, conf_cap = self._short_fallback_effect_params()

        eps_sorted = sorted(
            list(self._valid_touch_finished),
            key=lambda e: (e.end_frame, e.start_frame),
        )

        for ia, a in enumerate(eps_sorted):
            if a.contact_frames < cfg.pass_valid_touch_min_frames:
                continue
            for b in eps_sorted[ia + 1 :]:
                if (
                    b.player_id == a.player_id
                    or b.contact_frames < cfg.pass_valid_touch_min_frames
                ):
                    continue
                if cfg.pass_invalid_intermediate_memory_enabled and (
                    self._invalid_intermediate_active(a.player_id, frame_index)
                    or self._invalid_intermediate_active(b.player_id, frame_index)
                ):
                    continue
                if cfg.pass_short_valid_touch_fallback_recall_enabled:
                    la_lb, _, _ = self._touch_hist_recv_metrics(
                        a.player_id, a.start_frame, a.end_frame
                    )
                    lb_lb, _, _ = self._touch_hist_recv_metrics(
                        b.player_id, b.start_frame, b.end_frame
                    )
                    if la_lb < 1 or lb_lb < 1:
                        continue
                if b.start_frame <= a.end_frame:
                    continue
                gap = b.start_frame - a.end_frame
                if gap < 1:
                    continue
                span_frames = b.end_frame - a.start_frame + 1
                if span_frames > dur_max:
                    continue

                if a.last_ball_xy is None or b.first_ball_xy is None:
                    continue

                disp = _dist_px(a.last_ball_xy, b.first_ball_xy)
                if disp < disp_min:
                    continue

                ca, _ra = self._ephemeral_canonical_alias(
                    a.player_id, frame_index, track_to_team=track_to_team
                )
                cb, _rb = self._ephemeral_canonical_alias(
                    b.player_id, frame_index, track_to_team=track_to_team
                )
                if ca == cb or cb == a.player_id or ca == b.player_id:
                    self._dbg_gate(
                        f"sf_alias_same:{a.player_id}->{b.player_id}",
                        frame_index,
                        "[PASS-SHORT-FALLBACK] skip "
                        f"{a.player_id}->{b.player_id} reason=alias_same_player",
                    )
                    continue

                if self._short_fallback_duplicate_blocks(
                    a.start_frame,
                    b.end_frame,
                    a.player_id,
                    b.player_id,
                    frame_index,
                    duplicate_window_frames=dup_w,
                ):
                    continue

                if self._touch_intermediary_short_valid_blocks(a, b):
                    continue

                if cfg.pass_debug:
                    self._dbg_gate(
                        f"sf_cand:{a.player_id}->{b.player_id}:{a.start_frame}",
                        frame_index,
                        "[PASS-SHORT-FALLBACK] candidate "
                        f"{a.player_id}->{b.player_id} gap={gap} disp={disp:.1f}",
                    )

                evt = self._emit_short_valid_touch_event_internal(
                    a=a,
                    b=b,
                    frame_index=frame_index,
                    timestamp_sec=timestamp_sec,
                    possession=possession,
                    track_to_team=track_to_team,
                    disp_px=disp,
                    gap_frames=gap,
                    confidence_cap_override=conf_cap,
                    tracks=tracks,
                    ball_xy=(
                        (float(possession.ball_xy[0]), float(possession.ball_xy[1]))
                        if possession.ball_xy is not None
                        else None
                    ),
                )
                if evt is None:
                    continue
                self._touch_remove_short_valid_episode_pair(a, b)

                return evt

        return None

    def _touch_remove_short_valid_episode_pair(self, a: _ValidTouchEpisode, b: _ValidTouchEpisode) -> None:
        mxlen = getattr(self._valid_touch_finished, "maxlen", None) or 128

        def _k(e: _ValidTouchEpisode) -> tuple[int, int, int]:
            return (e.player_id, e.start_frame, e.end_frame)

        ka, kb = _k(a), _k(b)
        self._valid_touch_finished = deque(
            (e for e in self._valid_touch_finished if _k(e) not in (ka, kb)),
            maxlen=mxlen,
        )

    def _touch_intermediary_short_valid_blocks(
        self, a: _ValidTouchEpisode, b: _ValidTouchEpisode
    ) -> bool:
        """Block false short-gap passes when a strong intermediary valid touch occupies the gap."""

        cfg = self._config
        lo = a.end_frame + 1
        hi = b.start_frame - 1
        if lo > hi:
            return False
        need = max(3, cfg.pass_valid_touch_min_frames + 1)
        for ep in self._valid_touch_finished:
            if ep.player_id in (a.player_id, b.player_id):
                continue
            if ep.contact_frames < need:
                continue
            mx = max(ep.start_frame, lo)
            mn = min(ep.end_frame, hi)
            if mx <= mn:
                return True
        return False

    def _emit_short_valid_touch_event_internal(
        self,
        *,
        a: _ValidTouchEpisode,
        b: _ValidTouchEpisode,
        frame_index: int,
        timestamp_sec: Optional[float],
        possession: PossessionState,
        track_to_team: dict[int, int],
        disp_px: float,
        gap_frames: int,
        confidence_cap_override: Optional[float] = None,
        tracks: Optional["FrameTracks"] = None,
        ball_xy: Optional[tuple[float, float]] = None,
    ) -> Optional[PassEvent]:
        cfg = self._config
        cand = _FlightCandidate(
            start_frame=a.start_frame,
            start_time_sec=timestamp_sec,
            from_player_id=a.player_id,
            from_team_id=track_to_team.get(a.player_id) or a.team_id,
            ball_anchor_xy=a.first_ball_xy,
            release_frame=a.end_frame,
        )

        dribble_blocked, dribble_rst = self._dribble_overlap_guard_fsm(
            cand, b.player_id, frame_index, possession.ball_xy
        )
        if dribble_blocked:
            if cfg.pass_debug:
                self._dbg_gate(
                    f"sf_drblock:{a.player_id}->{b.player_id}",
                    frame_index,
                    "[PASS-SHORT-FALLBACK] skipped "
                    f"{a.player_id}->{b.player_id} dribble_guard reason={dribble_rst}",
                )
            self._print_pass_decision_suppressed(
                frame_index=frame_index,
                cand_from=a.player_id,
                cand_to=b.player_id,
                reason=dribble_rst or "dribble_overlap_not_pass",
            )
            return None

        ft = cand.from_team_id
        tt = self._resolver_team(possession, b.player_id, track_to_team)
        if b.team_id is not None:
            tt = tt or b.team_id

        evt_type = self._classify(
            ft,
            tt,
            cfg.require_same_team_for_completed,
            cfg.emit_interceptions,
        )
        dur_sec = None
        if timestamp_sec is not None:
            dur_sec = max(1e-3, timestamp_sec)

        dur_frames = max(1, b.end_frame - a.start_frame + 1)

        conf_cap_use = (
            confidence_cap_override
            if confidence_cap_override is not None
            else cfg.pass_short_valid_touch_confidence_cap
        )

        evt = PassEvent(
            event_id=self._next_event_id,
            event_type=evt_type,
            start_frame=a.start_frame,
            end_frame=b.end_frame,
            start_time_sec=None,
            end_time_sec=timestamp_sec,
            from_player_id=a.player_id,
            to_player_id=b.player_id,
            from_team_id=ft,
            to_team_id=tt,
            duration_frames=dur_frames,
            duration_sec=dur_sec,
            confidence=conf_cap_use,
            ball_start_xy=a.first_ball_xy,
            ball_end_xy=b.last_ball_xy,
            raw_from_team_id=ft,
            raw_to_team_id=tt,
            team_stable=False,
            debug_reason="valid_touch_transition",
            source="short_valid_touch_fallback",
            release_frame=a.end_frame,
            receiver_confirm_start_frame=b.start_frame,
            receiver_confirm_frames=b.contact_frames,
            ball_displacement_px=disp_px,
            pass_fsm_state="short_valid_touch_fallback",
            dribble_guard_status="passed_guard",
            receiver_contact_type="valid_touch_episode",
            bbox_only_guard_status="skipped_short_valid_fallback",
        )
        self._next_event_id += 1

        self._apply_emit_shared_state(
            frame_index, cand.from_player_id, b.player_id, clear_fsm_flight=False
        )
        evt = self._finalize_emitted_pass(evt)
        evt = self._apply_pass_source_reliability_touch(
            evt,
            cand.release_frame,
            cand.from_player_id,
        )
        self._apply_pass_event_aliases(evt, frame_index, track_to_team=track_to_team)
        bx = ball_xy
        if bx is None and possession.ball_xy is not None:
            bx = (float(possession.ball_xy[0]), float(possession.ball_xy[1]))
        self._attach_pass_decision_telemetry(
            evt,
            tracks=tracks,
            ball_xy=bx,
            track_to_team=track_to_team,
            frame_index=frame_index,
            emit_path_base="short_valid_touch_fallback",
            flight_start_frame=a.start_frame,
            flight_end_frame=b.end_frame,
            release_anchor_xy=a.last_ball_xy,
        )
        self._remember_pass_emission(evt)

        if cfg.pass_debug:
            self._dbg_gate(
                f"sf_emit:{a.player_id}->{b.player_id}",
                frame_index,
                "[PASS-SHORT-FALLBACK] emitted "
                f"{a.player_id}->{b.player_id} reason=valid_touch_transition",
            )
        return evt

    def _record_touch_hist_frame(
        self,
        frame_index: int,
        ball_xy: Optional[tuple[float, float]],
        tracks: Optional["FrameTracks"],
        track_to_team: dict[int, int],
    ) -> None:
        cfg = self._config
        if (
            not cfg.pass_touch_fallback_enabled
            and not cfg.pass_bbox_only_intermediate_guard_enabled
            and not cfg.pass_dribble_guard_enabled
            and not cfg.pass_valid_touch_gate_enabled
            and not cfg.pass_source_recover_from_valid_touch_enabled
            and not cfg.pass_receiver_retarget_after_bbox_only_enabled
            and not cfg.pass_short_valid_touch_fallback_enabled
            and not cfg.pass_none_source_recovery_enabled
            and not cfg.pass_invalid_intermediate_memory_enabled
            and not cfg.pass_chain_retarget_enabled
            and not cfg.pass_strict_bbox_only_classification_enabled
        ):
            return

        from VisionEngine.schemas.schema import ObjectRole

        tr_map: dict[int, _TouchHistTrack] = {}
        if tracks is not None and ball_xy is not None:
            bxy = (float(ball_xy[0]), float(ball_xy[1]))
            for inst in tracks.instances:
                if inst.role is not ObjectRole.PLAYER:
                    continue
                tid = inst.track_id
                if tid < 0:
                    continue
                bbox = tuple(float(x) for x in inst.xyxy)
                ft = foot_xy(bbox)
                d_f, lower_ok, valid_touch, bbox_only_flag = classify_touch_features(
                    cfg, bbox, bxy
                )
                bh = _bbox_height_xyxy(bbox)
                dyn_r = _dynamic_touch_radius_px(cfg, bh)
                if cfg.pass_touch_use_footpoint_distance:
                    strong = lower_ok and d_f <= min(
                        dyn_r * 0.58,
                        cfg.pass_touch_strong_contact_radius_px,
                    )
                else:
                    strong = False

                tr_map[tid] = _TouchHistTrack(
                    foot_xy=(float(ft[0]), float(ft[1])),
                    bbox=(bbox[0], bbox[1], bbox[2], bbox[3]),
                    d_foot=d_f,
                    valid_touch=valid_touch,
                    strong_touch=bool(strong),
                    bbox_overlap_only=bool(bbox_only_flag),
                    lower_body_valid=bool(lower_ok),
                )

        self._touch_hist.append(
            _TouchHistFrame(
                frame_index=frame_index,
                ball_xy=(
                    tuple(ball_xy)
                    if ball_xy is not None
                    else None
                ),
                tracks=dict(tr_map),
            )
        )
        if not cfg.pass_touch_fallback_enabled:
            return

        touching = {tid for tid, ht in tr_map.items() if ht.valid_touch}

        for tid in list(self._touch_open.keys()):
            if tid in touching:
                continue
            op = self._touch_open[tid]
            op.misses += 1
            if op.misses > 1:
                self._touch_finalize_open_episode(tid, track_to_team)

        for tid in touching:
            ht = tr_map[tid]
            team_snap = track_to_team.get(tid)
            if tid not in self._touch_open:
                self._touch_open[tid] = _TouchOpenEpisode(
                    player_id=tid,
                    team_id_snap=team_snap,
                    start_frame=frame_index,
                    end_frame=frame_index,
                    misses=0,
                    contact_frames=1,
                    strong_contact_frames=1 if ht.strong_touch else 0,
                    lower_body_touch_frames=1 if ht.lower_body_valid else 0,
                    bbox_overlap_only_frames=1 if ht.bbox_overlap_only else 0,
                    min_d=ht.d_foot,
                    first_ball_xy=tuple(ball_xy) if ball_xy is not None else None,
                    last_ball_xy=tuple(ball_xy) if ball_xy is not None else None,
                )
            else:
                op = self._touch_open[tid]
                op.misses = 0
                op.end_frame = frame_index
                op.contact_frames += 1
                if ht.strong_touch:
                    op.strong_contact_frames += 1
                if ht.lower_body_valid:
                    op.lower_body_touch_frames += 1
                if ht.bbox_overlap_only:
                    op.bbox_overlap_only_frames += 1
                op.min_d = min(op.min_d, ht.d_foot)
                if ball_xy is not None:
                    op.last_ball_xy = tuple(ball_xy)
                if op.team_id_snap is None and team_snap is not None:
                    op.team_id_snap = team_snap

    def _touch_finalize_open_episode(
        self, tid: int, track_to_team: dict[int, int]
    ) -> None:
        cfg = self._config
        o = self._touch_open.pop(tid, None)
        if o is None:
            return
        team_id = track_to_team.get(tid)
        if team_id is None:
            team_id = o.team_id_snap
        ep = _TouchEpisode(
            player_id=tid,
            team_id=team_id,
            start_frame=o.start_frame,
            end_frame=o.end_frame,
            contact_frames=o.contact_frames,
            strong_contact_frames=o.strong_contact_frames,
            lower_body_touch_frames=o.lower_body_touch_frames,
            bbox_overlap_only_frames=o.bbox_overlap_only_frames,
            min_distance_to_ball=float(o.min_d),
            first_ball_xy=o.first_ball_xy,
            last_ball_xy=o.last_ball_xy,
        )
        self._touch_finished.append(ep)
        if cfg.pass_debug:
            if ep.contact_frames >= cfg.pass_touch_min_receiver_contact_frames:
                self._dbg(
                    "[PASS-TOUCH] receiver_touch "
                    f"player={tid} frames={ep.contact_frames} "
                    f"min_dist={ep.min_distance_to_ball:.1f}"
                )
            elif ep.contact_frames >= cfg.pass_touch_min_source_contact_frames:
                self._dbg(
                    "[PASS-TOUCH] source_touch "
                    f"player={tid} frames={ep.contact_frames} "
                    f"min_dist={ep.min_distance_to_ball:.1f}"
                )

    def _touch_intermediary_blocks(self, a: _TouchEpisode, b: _TouchEpisode) -> bool:
        cfg = self._config
        lo = a.end_frame + 1
        hi = b.start_frame - 1
        if lo > hi:
            return False
        need = cfg.pass_touch_min_receiver_contact_frames
        for z in self._touch_finished:
            if z.player_id in (a.player_id, b.player_id):
                continue
            if z.contact_frames < need:
                continue
            if max(z.start_frame, lo) <= min(z.end_frame, hi):
                return True
        return False

    def _touch_duplicate_blocks(
        self,
        start_f: int,
        end_f: int,
        from_id: Optional[int],
        to_id: int,
        frame_index: int = 0,
    ) -> bool:
        cfg = self._config
        w = cfg.pass_touch_duplicate_window_frames

        def overlap(s1: int, e1: int, s2: int, e2: int) -> bool:
            return not (e1 < s2 or s1 > e2)

        for ev in self._pass_emission_log:
            if overlap(start_f, end_f, ev.start_frame, ev.end_frame):
                self._dbg_gate(
                    f"touch_dup_ov:{from_id}->{to_id}",
                    frame_index,
                    "[PASS-TOUCH] skip duplicate "
                    f"{from_id}->{to_id} reason=near_existing_event",
                )
                return True
            handoff_gap = ev.start_frame - end_f
            if (
                ev.source != "touch_to_touch_fallback"
                and 0 <= handoff_gap <= cfg.pass_bbox_only_chain_window_frames
                and ev.from_player_id is not None
                and ev.from_player_id != to_id
            ):
                self._dbg_gate(
                    f"touch_dup_handoff:{from_id}->{to_id}:{ev.from_player_id}",
                    frame_index,
                    "[PASS-TOUCH] skip duplicate "
                    f"{from_id}->{to_id} reason=not_handoff_to_next_primary_source",
                )
                return True
            if ev.to_player_id == to_id and abs(ev.end_frame - end_f) <= w:
                self._dbg_gate(
                    f"touch_dup_to:{from_id}->{to_id}",
                    frame_index,
                    "[PASS-TOUCH] skip duplicate "
                    f"{from_id}->{to_id} reason=near_existing_event",
                )
                return True
            if (
                from_id is not None
                and ev.from_player_id is not None
                and ev.from_player_id == from_id
                and ev.to_player_id == to_id
                and min(abs(ev.end_frame - end_f), abs(ev.start_frame - start_f)) <= w
            ):
                self._dbg_gate(
                    f"touch_dup_fp:{from_id}->{to_id}",
                    frame_index,
                    "[PASS-TOUCH] skip duplicate "
                    f"{from_id}->{to_id} reason=near_existing_event",
                )
                return True
        return False

    def _touch_remove_episode_pair(self, a: _TouchEpisode, b: _TouchEpisode) -> None:
        def _k(e: _TouchEpisode) -> tuple[int, int, int]:
            return (e.player_id, e.start_frame, e.end_frame)

        ka, kb = _k(a), _k(b)
        mxlen = getattr(self._touch_finished, "maxlen", None) or 96
        self._touch_finished = deque(
            (
                e
                for e in self._touch_finished
                if _k(e) not in (ka, kb)
            ),
            maxlen=mxlen,
        )

    def _try_emit_touch_supplemental(
        self,
        frame_index: int,
        timestamp_sec: Optional[float],
        possession: PossessionState,
        track_to_team: dict[int, int],
        tracks: Optional["FrameTracks"] = None,
    ) -> Optional[PassEvent]:
        cfg = self._config
        if not cfg.pass_touch_fallback_enabled:
            return None

        lookahead = cfg.pass_touch_receiver_lookahead_frames
        min_src = cfg.pass_touch_min_source_contact_frames
        min_rcv = cfg.pass_touch_min_receiver_contact_frames
        max_gap = cfg.pass_touch_max_duration_frames

        ball_xy_hint = None
        if possession.ball_xy is not None:
            ball_xy_hint = (
                float(possession.ball_xy[0]),
                float(possession.ball_xy[1]),
            )

        eps_sorted = sorted(self._touch_finished, key=lambda e: (e.end_frame, e.start_frame))

        for ia, a in enumerate(eps_sorted):
            if a.contact_frames < min_src:
                continue
            for b in eps_sorted[ia + 1 :]:
                if b.player_id == a.player_id or b.contact_frames < min_rcv:
                    continue
                if b.start_frame <= a.end_frame:
                    continue
                gap = b.start_frame - a.end_frame
                if gap < 1 or gap > max_gap:
                    continue
                if frame_index < b.end_frame + lookahead:
                    continue
                if self._flight is not None and b.player_id != self._flight.from_player_id:
                    self._dbg_gate(
                        f"touch_open_flight_recv:{a.player_id}->{b.player_id}",
                        frame_index,
                        "[PASS-TOUCH] skip "
                        f"{a.player_id}->{b.player_id} reason=receiver_not_open_flight_source",
                    )
                    continue
                if a.last_ball_xy is None or b.first_ball_xy is None:
                    continue
                if self._touch_intermediary_blocks(a, b):
                    continue

                disp = _dist_px(a.last_ball_xy, b.first_ball_xy)
                if disp < cfg.pass_touch_min_displacement_px:
                    continue

                ca, _ra = self._ephemeral_canonical_alias(
                    a.player_id, frame_index, track_to_team=track_to_team
                )
                cb, _rb = self._ephemeral_canonical_alias(
                    b.player_id, frame_index, track_to_team=track_to_team
                )
                if ca == cb or cb == a.player_id or ca == b.player_id:
                    self._dbg_gate(
                        f"touch_alias_same:{a.player_id}->{b.player_id}",
                        frame_index,
                        "[PASS-TOUCH] skip "
                        f"{a.player_id}->{b.player_id} reason=alias_same_player",
                    )
                    continue

                denom = float(max(1, gap))
                avg_speed = disp / denom
                if avg_speed < cfg.pass_touch_min_avg_speed_px_per_frame:
                    continue

                if self._touch_duplicate_blocks(
                    a.start_frame, b.end_frame, a.player_id, b.player_id, frame_index
                ):
                    continue

                if cfg.pass_touch_require_lower_body_or_footpoint:
                    if (
                        a.lower_body_touch_frames < min_src
                        or b.lower_body_touch_frames < min_rcv
                    ):
                        continue

                if self._touch_pair_dribble_skip(a, b):
                    self._dbg_gate(
                        f"touch_dribble_skip:{a.player_id}->{b.player_id}",
                        frame_index,
                        "[PASS-DRIBBLE-GUARD] skip touch_fallback "
                        f"{a.player_id}->{b.player_id} reason=dribble_overlap_not_pass",
                    )
                    continue

                if cfg.pass_debug:
                    self._dbg(
                        "[PASS-TOUCH] candidate "
                        f"{a.player_id}->{b.player_id} duration={gap} "
                        f"disp={disp:.1f} speed={avg_speed:.2f}"
                    )

                ev = self._emit_touch_supplemental_event(
                    a=a,
                    b=b,
                    frame_index=frame_index,
                    timestamp_sec=timestamp_sec,
                    possession=possession,
                    track_to_team=track_to_team,
                    disp_px=disp,
                    avg_speed=avg_speed,
                    tracks=tracks,
                    ball_xy=ball_xy_hint,
                )
                self._touch_remove_episode_pair(a, b)
                if cfg.pass_debug:
                    self._dbg(
                        "[PASS-TOUCH] emitted "
                        f"{a.player_id}->{b.player_id} "
                        f"source=touch_to_touch_fallback conf={ev.confidence:.2f}"
                    )
                return ev

        return None

    def _emit_touch_supplemental_event(
        self,
        *,
        a: _TouchEpisode,
        b: _TouchEpisode,
        frame_index: int,
        timestamp_sec: Optional[float],
        possession: PossessionState,
        track_to_team: dict[int, int],
        disp_px: float,
        avg_speed: float,
        tracks: Optional["FrameTracks"] = None,
        ball_xy: Optional[tuple[float, float]] = None,
    ) -> PassEvent:
        cfg = self._config
        from_team = a.team_id
        to_team = b.team_id
        to_team_res = self._resolver_team(possession, b.player_id, track_to_team)
        if to_team is None:
            to_team = to_team_res
        elif to_team_res is not None:
            to_team = to_team_res

        evt_type = self._classify(
            from_team,
            to_team,
            cfg.require_same_team_for_completed,
            cfg.emit_interceptions,
        )

        conf = 0.60
        if a.strong_contact_frames > 0:
            conf += 0.10
        if b.strong_contact_frames > 0:
            conf += 0.10
        if from_team is not None and to_team is not None:
            conf += 0.05
        if disp_px >= 2.0 * cfg.pass_touch_min_displacement_px:
            conf += 0.05
        conf = min(conf, cfg.pass_touch_confidence_cap)

        duration_frames = b.end_frame - a.start_frame + 1

        event = PassEvent(
            event_id=self._next_event_id,
            event_type=evt_type,
            start_frame=a.start_frame,
            end_frame=b.end_frame,
            start_time_sec=None,
            end_time_sec=timestamp_sec,
            from_player_id=a.player_id,
            to_player_id=b.player_id,
            from_team_id=from_team,
            to_team_id=to_team,
            duration_frames=duration_frames,
            duration_sec=None,
            confidence=conf,
            ball_start_xy=a.first_ball_xy,
            ball_end_xy=b.last_ball_xy,
            raw_from_team_id=from_team,
            raw_to_team_id=to_team,
            team_stable=False,
            debug_reason="raw_touch_transition",
            release_frame=a.end_frame,
            receiver_confirm_start_frame=b.start_frame,
            receiver_confirm_frames=b.contact_frames,
            ball_displacement_px=disp_px,
            touch_source_frames=a.contact_frames,
            touch_receiver_frames=b.contact_frames,
            touch_source_min_distance_px=a.min_distance_to_ball,
            touch_receiver_min_distance_px=b.min_distance_to_ball,
            avg_ball_speed_px_per_frame=avg_speed,
            pass_fsm_state="touch_to_touch_fallback",
            source="touch_to_touch_fallback",
            receiver_selection_reason="touch_fallback_episode_pair",
            dribble_guard_status="passed_guard",
            receiver_contact_type=(
                "touch_episode_lb"
                if b.lower_body_touch_frames >= cfg.pass_touch_min_receiver_contact_frames
                else "touch_episode_mixed"
            ),
            bbox_only_guard_status="skipped_touch_fallback",
        )
        self._next_event_id += 1
        self._apply_emit_shared_state(
            frame_index, a.player_id, b.player_id, clear_fsm_flight=False
        )
        event = self._finalize_emitted_pass(event)
        event = self._apply_pass_source_reliability_touch(
            event, a.end_frame, a.player_id
        )
        self._apply_pass_event_aliases(event, frame_index, track_to_team=track_to_team)
        bx = ball_xy
        if bx is None and possession.ball_xy is not None:
            bx = (float(possession.ball_xy[0]), float(possession.ball_xy[1]))
        self._attach_pass_decision_telemetry(
            event,
            tracks=tracks,
            ball_xy=bx,
            track_to_team=track_to_team,
            frame_index=frame_index,
            emit_path_base="touch_to_touch_fallback",
            flight_start_frame=a.start_frame,
            flight_end_frame=b.end_frame,
            release_anchor_xy=a.last_ball_xy,
        )
        self._remember_pass_emission(event)
        return event

    def _append_touch_fallback_if_needed(
        self,
        frame_index: int,
        timestamp_sec: Optional[float],
        events: list[PassEvent],
        possession: PossessionState,
        track_to_team: dict[int, int],
        tracks: Optional["FrameTracks"] = None,
    ) -> None:
        if events:
            return
        ev = self._try_emit_touch_supplemental(
            frame_index,
            timestamp_sec,
            possession,
            track_to_team,
            tracks,
        )
        if ev is not None:
            events.extend(self._stage_or_emit_event(ev, frame_index))
            return
        sf = self._try_emit_short_valid_touch_fallback(
            frame_index,
            timestamp_sec,
            possession,
            track_to_team,
            tracks,
        )
        if sf is not None:
            events.extend(self._stage_or_emit_event(sf, frame_index))

    def update(
        self,
        frame_index: int,
        timestamp_sec: Optional[float],
        possession: PossessionState,
        track_to_team: dict[int, int],
        tracks: Optional["FrameTracks"] = None,
    ) -> list[PassEvent]:
        events: list[PassEvent] = []
        cfg = self._config
        present_ids = self._gather_present_track_ids(tracks)
        self._consume_disappearing_tracks(frame_index, present_ids, track_to_team)
        self._maybe_alias_first_seen_players(
            frame_index, tracks, present_ids, track_to_team
        )
        try:
            self._record_teams_for_frame(track_to_team)
            self._update_foot_cache(tracks)
            self._expire_receive_blocks(frame_index)
            self._prune_invalid_intermediate(frame_index)

            ball_xy = possession.ball_xy
            poss_tid = possession.track_id

            self._record_pass_frame_snap(frame_index, ball_xy)
            self._record_touch_hist_frame(frame_index, ball_xy, tracks, track_to_team)
            self._record_valid_touch_frame(
                frame_index, ball_xy, tracks, track_to_team
            )
            self._release_expired_pending_weak_chain(frame_index, events)
            if events:
                return events
            ret_ev = self._try_emit_pending_return_pass(
                frame_index=frame_index,
                timestamp_sec=timestamp_sec,
                possession=possession,
                track_to_team=track_to_team,
                tracks=tracks,
            )
            if ret_ev is not None:
                events.append(ret_ev)
                return events

            owner_switch: Optional[tuple[int, int]] = None
            if self._flight is None and frame_index >= self._cooldown_until_frame:
                owner_switch = self._update_controlled_owner(
                    poss_tid, frame_index=frame_index
                )
                if owner_switch is not None:
                    prev_owner, new_owner = owner_switch
                    ev_sw = self._try_emit_owner_switch_handoff(
                        prev_owner=prev_owner,
                        new_owner=new_owner,
                        frame_index=frame_index,
                        timestamp_sec=timestamp_sec,
                        possession=possession,
                        track_to_team=track_to_team,
                        tracks=tracks,
                    )
                    if ev_sw is not None:
                        events.extend(self._stage_or_emit_event(ev_sw, frame_index))
                        self._append_touch_fallback_if_needed(
                            frame_index,
                            timestamp_sec,
                            events,
                            possession,
                            track_to_team,
                            tracks,
                        )
                        return events
                    ev_prox = self._try_emit_owner_switch_proximity_fallback(
                        prev_owner=prev_owner,
                        new_owner=new_owner,
                        frame_index=frame_index,
                        timestamp_sec=timestamp_sec,
                        possession=possession,
                        track_to_team=track_to_team,
                        tracks=tracks,
                    )
                    if ev_prox is not None:
                        events.append(ev_prox)
                        return events

            if self._flight is not None:
                cand = self._flight
                flight_len = frame_index - cand.start_frame + 1

                if flight_len > cfg.pass_candidate_timeout_frames:
                    self._dbg(
                        "[PASS-FSM] flight_timeout "
                        f"from={cand.from_player_id} frames={flight_len} — discard"
                    )
                    timeout_events = self._try_emit_timeout_contact_transfer(
                        cand=cand,
                        frame_index=frame_index,
                        timestamp_sec=timestamp_sec,
                        possession=possession,
                        track_to_team=track_to_team,
                        tracks=tracks,
                    )
                    if timeout_events:
                        for tev in timeout_events:
                            events.extend(self._stage_or_emit_event(tev, frame_index))
                        self._append_touch_fallback_if_needed(
                            frame_index,
                            timestamp_sec,
                            events,
                            possession,
                            track_to_team,
                            tracks,
                        )
                        return events
                    self._clear_flight()
                    self._away_run = 0
                    self._away_anchor = None
                    self._append_touch_fallback_if_needed(
                        frame_index, timestamp_sec, events, possession, track_to_team, tracks
                    )
                    return events

                if ball_xy is not None:
                    f_from = self._foot_of(cand.from_player_id, possession)
                    if f_from is not None:
                        d_back = _dist_px(ball_xy, f_from)
                        if d_back < cfg.pass_release_min_distance_px:
                            self._reclaim_run += 1
                        else:
                            self._reclaim_run = 0
                        if self._reclaim_run >= cfg.same_player_grace_frames and flight_len > cfg.same_player_grace_frames:
                            self._dbg(
                                "[PASS-FSM] passer_reclaim_cancel "
                                f"from={cand.from_player_id} frame={frame_index}"
                            )
                            self._clear_flight()
                            self._away_run = 0
                            self._away_anchor = None
                            self._append_touch_fallback_if_needed(
                                frame_index,
                                timestamp_sec,
                                events,
                                possession,
                                track_to_team,
                                tracks,
                            )
                            return events

                cand_recv = None
                if ball_xy is not None and tracks is not None:
                    cand_recv, rreason, rscore = self._pick_receiver_for_pass(
                        tracks=tracks,
                        ball_xy=ball_xy,
                        exclude=cand.from_player_id,
                        frame_index=frame_index,
                        release_anchor=cand.ball_anchor_xy,
                        radius_px=None,
                        track_to_team=track_to_team,
                    )
                    self._flight_recv_pick_reason = rreason
                    self._flight_recv_pick_score = float(rscore)

                self._record_flight_contact_frame(
                    cand=cand,
                    poss_tid=poss_tid,
                    cand_recv=cand_recv,
                    frame_index=frame_index,
                    ball_xy=ball_xy,
                    track_to_team=track_to_team,
                )

                self._advance_transient_overlap(
                    poss_tid, cand.from_player_id, cand_recv, frame_index
                )

                rx = cand_recv
                if rx is None:
                    self._recv_gap += 1
                    if self._recv_gap > cfg.pass_receiver_max_gap_frames:
                        if self._recv_streak > 0 and cfg.pass_debug:
                            self._dbg(
                                "[PASS-FSM] receiver_streak_reset gap="
                                f"{self._recv_gap} frame={frame_index}"
                            )
                        self._recv_streak = 0
                        self._recv_tid = None
                        self._recv_confirm_start_frame = None
                elif self._recv_tid is None:
                    self._recv_tid = rx
                    self._recv_streak = 1
                    self._recv_gap = 0
                    self._recv_confirm_start_frame = frame_index
                    if cfg.pass_debug:
                        self._dbg(
                            f"[PASS-FSM] receiver_candidate={rx} frames=1 "
                            f"@ frame={frame_index}"
                        )
                elif rx == self._recv_tid:
                    self._recv_streak += 1
                    self._recv_gap = 0
                    if cfg.pass_debug and self._recv_streak == cfg.pass_receiver_confirm_frames:
                        self._dbg(
                            f"[PASS-FSM] receiver_candidate={rx} frames="
                            f"{self._recv_streak} confirmed"
                        )
                else:
                    self._recv_tid = rx
                    self._recv_streak = 1
                    self._recv_gap = 0
                    self._recv_confirm_start_frame = frame_index
                    if cfg.pass_debug:
                        self._dbg(
                            f"[PASS-FSM] receiver_candidate={rx} frames=1 "
                            f"@ frame={frame_index}"
                        )

                ball_travel_ok = True
                if (
                    cand.ball_anchor_xy is not None
                    and ball_xy is not None
                    and cfg.pass_release_min_ball_displacement_px > 0
                ):
                    ball_travel_ok = (
                        _dist_px(cand.ball_anchor_xy, ball_xy)
                        >= cfg.pass_release_min_ball_displacement_px
                    )

                emit_ok = (
                    self._recv_streak >= cfg.pass_receiver_confirm_frames
                    and self._recv_tid is not None
                    and self._recv_tid != cand.from_player_id
                    and flight_len >= cfg.min_pass_frames
                    and ball_xy is not None
                    and ball_travel_ok
                )

                if emit_ok:
                    source_split_events = self._try_emit_timeout_source_loose_split(
                        cand=cand,
                        final_receiver_id=self._recv_tid,  # type: ignore[arg-type]
                        frame_index=frame_index,
                        timestamp_sec=timestamp_sec,
                        possession=possession,
                        track_to_team=track_to_team,
                        tracks=tracks,
                    )
                    if source_split_events:
                        for ssev in source_split_events:
                            events.extend(self._stage_or_emit_event(ssev, frame_index))
                        self._append_touch_fallback_if_needed(
                            frame_index,
                            timestamp_sec,
                            events,
                            possession,
                            track_to_team,
                            tracks,
                        )
                        return events
                    split_events = self._try_emit_flight_contact_split(
                        cand=cand,
                        final_receiver_id=self._recv_tid,  # type: ignore[arg-type]
                        frame_index=frame_index,
                        timestamp_sec=timestamp_sec,
                        possession=possession,
                        track_to_team=track_to_team,
                        tracks=tracks,
                    )
                    if split_events:
                        for sev in split_events:
                            events.extend(self._stage_or_emit_event(sev, frame_index))
                        self._append_touch_fallback_if_needed(
                            frame_index,
                            timestamp_sec,
                            events,
                            possession,
                            track_to_team,
                            tracks,
                        )
                        return events
                    gx = tuple(ball_xy) if ball_xy is not None else None
                    ev = self._emit_completed(
                        cand=cand,
                        receiver_id=self._recv_tid,  # type: ignore[arg-type]
                        flight_len=flight_len,
                        frame_index=frame_index,
                        timestamp_sec=timestamp_sec,
                        possession=possession,
                        track_to_team=track_to_team,
                        tracks=tracks,
                        ball_xy_for_guard=gx,
                    )
                    if ev is None:
                        self._clear_flight()
                        self._away_run = 0
                        self._away_anchor = None
                        self._append_touch_fallback_if_needed(
                            frame_index, timestamp_sec, events, possession, track_to_team, tracks
                        )
                        return events
                    events.extend(self._stage_or_emit_event(ev, frame_index))
                    self._append_touch_fallback_if_needed(
                        frame_index, timestamp_sec, events, possession, track_to_team, tracks
                    )
                    return events

                self._append_touch_fallback_if_needed(
                    frame_index, timestamp_sec, events, possession, track_to_team, tracks
                )
                return events

            # ------- No in-flight candidate: armed release -------
            owner = self._co_owner
            if owner is None or ball_xy is None:
                self._away_run = 0
                self._away_anchor = None
                self._append_touch_fallback_if_needed(
                    frame_index, timestamp_sec, events, possession, track_to_team, tracks
                )
                return events

            blocked = self._post_receive_blocked(owner, frame_index)

            if blocked and cfg.pass_allow_one_touch_release:
                ot_ev = self._try_emit_one_touch(
                    frame_index=frame_index,
                    timestamp_sec=timestamp_sec,
                    possession=possession,
                    track_to_team=track_to_team,
                    tracks=tracks,
                )
                if ot_ev is not None:
                    events.extend(self._stage_or_emit_event(ot_ev, frame_index))
                    self._append_touch_fallback_if_needed(
                        frame_index, timestamp_sec, events, possession, track_to_team, tracks
                    )
                    return events

            f_own = self._foot_of(owner, possession)
            min_d = (
                cfg.pass_release_min_distance_px
                if f_own is not None
                else float("inf")
            )
            far = False
            if f_own is not None:
                far = _dist_px(ball_xy, f_own) > min_d
            elif poss_tid != owner:
                far = True

            if blocked:
                sf = self._last_event_end_frame
                episode = sf if sf is not None else -1
                if (
                    owner == self._last_receiver_id
                    and sf is not None
                    and far
                    and cfg.pass_debug
                    and self._suppress_release_episode_logged != episode
                ):
                    since = frame_index - sf
                    self._dbg(
                        "[PASS-FSM] suppress immediate receiver release "
                        f"owner={owner} frame={frame_index} since_last_event={since} "
                        f"settle={cfg.pass_post_receive_settle_frames}"
                    )
                    self._suppress_release_episode_logged = episode
                self._away_run = 0
                self._away_anchor = None
                self._append_touch_fallback_if_needed(
                    frame_index, timestamp_sec, events, possession, track_to_team, tracks
                )
                return events

            if far:
                self._away_run += 1
                if self._away_anchor is None:
                    self._away_anchor = ball_xy
                    self._dbg(
                        f"[PASS-FSM] release started from={owner} frame={frame_index}"
                    )
            else:
                self._away_run = 0
                self._away_anchor = None

            disp = (
                _dist_px(ball_xy, self._away_anchor)
                if self._away_anchor is not None
                else 0.0
            )

            if (
                self._away_run >= cfg.pass_release_min_frames_away
                and disp >= cfg.pass_release_min_ball_displacement_px
            ):
                from_team_snap = track_to_team.get(owner)
                self._reset_flight_contacts()
                self._flight = _FlightCandidate(
                    start_frame=frame_index,
                    start_time_sec=timestamp_sec,
                    from_player_id=owner,
                    from_team_id=from_team_snap,
                    ball_anchor_xy=self._away_anchor,
                    release_frame=frame_index,
                )
                self._flight_recv_pick_reason = ""
                self._flight_recv_pick_score = -1.0
                self._recv_tid = None
                self._recv_streak = 0
                self._recv_gap = 0
                self._recv_confirm_start_frame = None
                self._ignored_transient_episodes = 0
                self._away_run = 0
                self._away_anchor = None

                stable_n = cfg.pass_owner_min_stable_frames
                self._dbg(
                    "[PASS-FSM] in_flight_open "
                    f"from={owner} release_frame={frame_index} stable_owner_frames>={stable_n}"
                )
            self._append_touch_fallback_if_needed(
                frame_index, timestamp_sec, events, possession, track_to_team, tracks
            )
            return events
        finally:
            self._snapshot_tracking_state_after_frame(
                frame_index, tracks, track_to_team, present_ids
            )

    def flush(self, final_frame_index: int, final_timestamp_sec: Optional[float]) -> list[PassEvent]:
        del final_frame_index, final_timestamp_sec
        tail: list[PassEvent] = []
        if self._pending_weak_chain_event is not None:
            tail.append(self._pending_weak_chain_event)
            self._pending_weak_chain_event = None
        if self._flight is not None and self._config.pass_debug:
            self._dbg("[PASS-FSM] flush: discard open flight candidate")
        self._clear_flight()
        self._clear_one_touch_accumulator()
        self._touch_hist.clear()
        self._touch_open.clear()
        self._touch_finished.clear()
        self._suppressed_bbox_receiver_chain.clear()
        self._valid_touch_snap_hist.clear()
        self._valid_touch_open.clear()
        self._valid_touch_finished.clear()
        return tail


def format_pass_terminal_for_debug(ev: PassEvent) -> str:
    dest = ev.to_player_id if ev.to_player_id is not None else "?"
    frm = ev.from_player_id
    frm_s = "None" if frm is None else str(frm)
    rf, rt = ev.raw_from_team_id, ev.raw_to_team_id
    rsf = "?" if rf is None else str(rf)
    rst = "?" if rt is None else str(rt)

    src = ev.source or ""
    if src == "touch_to_touch_fallback":
        orig_s = ""
        if ev.original_from_player_id is not None and ev.from_player_id is None:
            orig_s = f" original_from={ev.original_from_player_id}"
        dr = f" reason={ev.debug_reason}" if ev.debug_reason else ""
        return (
            f"[PASS] frame {ev.start_frame}-{ev.end_frame} "
            f"{ev.event_type} {frm_s} -> {dest}{orig_s}{dr} "
            f"source={src} conf={ev.confidence:.2f}"
        )

    if src == "kickoff_bootstrap_unknown_passer":
        return (
            f"[PASS] frame {ev.start_frame}-{ev.end_frame} "
            f"{ev.event_type} {frm_s} -> {dest} "
            f"source={src} conf={ev.confidence:.2f}"
        )

    if (
        ev.event_type == "unknown_pass"
        and ev.original_from_player_id is not None
        and src != "kickoff_bootstrap_unknown_passer"
        and src != "touch_to_touch_fallback"
    ):
        rs = f" reason={ev.debug_reason}" if ev.debug_reason else ""
        return (
            f"[PASS] frame {ev.start_frame}-{ev.end_frame} "
            f"{ev.event_type} {frm_s} -> {dest} "
            f"original_from={ev.original_from_player_id}{rs} "
            f"conf={ev.confidence:.2f}"
        )

    return (
        f"[PASS] frame {ev.start_frame}-{ev.end_frame} "
        f"{ev.event_type} {frm_s} -> {dest} raw_teams={rsf}->{rst} "
        f"team_stable={bool(ev.team_stable)} conf={ev.confidence:.2f}"
    )
