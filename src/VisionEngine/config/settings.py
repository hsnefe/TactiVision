"""Runtime configuration for the analysis pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from VisionEngine.schemas.schema import ObjectRole


@dataclass
class Settings:
    """Paths, model options, and feature flags."""

    input_video: Path
    output_video: Path
    mode: str = "normal"
    """Pipeline mode: ``normal`` (broadcast tracking pipeline) or ``field``."""

    model_path: str = "yolo11n.pt"
    """Default YOLO11 nano model path."""

    conf_threshold: float = 0.22
    """Main ``track()`` confidence; slightly below 0.25 to help marginal detections."""

    iou_threshold: float = 0.45
    tracker_config: str = "bytetrack.yaml"
    """Ultralytics tracker YAML name (under ultralytics/cfg/trackers/)."""

    inference_imgsz: int = 1280
    """Letterbox size for both ``track`` and ball-only ``predict`` (larger helps tiny ball)."""

    ball_conf_threshold: float = 0.40
    """Lower threshold on ball-class-only ``predict`` pass for recall / yellow debug overlay."""

    ball_max_gap_frames: int = 15
    """When track loses the ball, extrapolate position for this many frames (blue overlay)."""

    ball_debug_overlay: bool = True
    """Draw yellow raw / red tracked / blue estimated ball diagnostics on the output video."""

    ball_roi_recovery: bool = True
    """Activates Region of Interest (ROI) scanning when the ball is lost."""

    ball_roi_conf: float = 0.08
    """Low confidence threshold for ball ROI scan."""

    ball_roi_size: int = 300
    """Pixel size (width/height) of the cropped region to search for the ball."""

    roi_recovery_enabled: bool = False
    """
    If True, enable the aggressive identity-preservation pipeline for players:
    ``_relabel_lost_players_from_primary_nearby`` rebinds new tracker IDs to
    recently lost IDs near the same location, and a second low-threshold
    person-only ``predict`` is run on ROIs around lost players to inject
    synthetic tracks with the same ``track_id``.

    Default is ``False`` so that in (team-)classification runs a player who is
    lost by ByteTrack and re-detected with a new ID is rendered immediately,
    matching ``--debug_persons`` behavior. Enable with ``--roi-recovery`` when
    ID continuity across short occlusions matters more than freshness.
    """

    roi_conf_threshold: float = 0.08
    """Confidence for ROI-only person ``predict`` (typically below main ``conf_threshold``)."""

    roi_margin_ratio: float = 0.4
    """Expand last bbox by this fraction of width/height before cropping (per side)."""

    roi_inference_imgsz: int = 1280
    """Letterbox size for ROI ``predict`` (larger helps small / weak person boxes)."""

    roi_max_per_frame: int = 8
    """Max lost player ROIs to evaluate per frame (cost control)."""

    roi_skip_near_border: bool = True
    """Skip ROI recovery when the last bbox touches the frame border (likely out of scope)."""

    roi_border_margin_px: int = 20
    """Pixel margin for border touch test (same idea as debug player scope)."""

    roi_min_iou_with_last: float = 0.08
    """Reject ROI detections with IoU below this vs the last bbox (wild false positives)."""

    roi_max_iou_with_other_track: float = 0.45
    """Skip recovery if the candidate overlaps another track's box this strongly."""

    roi_max_lost_streak: int = 120
    """Stop ROI attempts after this many consecutive primary misses for the same track id."""

    class_mapping_preset: str = "coco_football"
    """Preset for YOLO class id -> role: ``coco_football`` or ``football_three_class``."""

    class_role_overrides: Optional[dict[int, "ObjectRole"]] = None
    """Optional per-class index overrides after preset (e.g. from CLI)."""

    only_mapped_classes: bool = True
    """If True, drop detections whose role is ``OTHER`` after mapping."""

    tracks_log_path: Optional[Path] = None
    """If set, append per-frame tracking records as JSON Lines."""

    pass_detection_enabled: bool = False
    """When True, run the finite-state pass detector on possession + track teams."""

    pass_events_jsonl_path: Optional[Path] = None
    """Destination JSONL for :class:`~VisionEngine.EventAnalytics.pass_detector.PassEvent` rows."""

    throwin_detection_enabled: bool = False
    """When True, run the separate throw-in event detector."""

    throwin_events_jsonl_path: Optional[Path] = None
    """Destination JSONL for throw-in event rows."""

    throwin_debug: bool = False
    """Print throw-in detector diagnostics and emitted events to terminal."""

    pass_min_possession_frames: int = 3
    pass_min_pass_frames: int = 2
    pass_max_pass_frames: int = 90
    pass_cooldown_frames: int = 10
    pass_same_player_grace_frames: int = 4
    pass_emit_interceptions: bool = True
    pass_require_same_team_for_completed: bool = True

    pass_team_stability_window: int = 10
    """Per-track team-id deque length inside :class:`~VisionEngine.EventAnalytics.pass_detector.PassDetector`."""

    pass_team_min_stable_count: int = 6
    """Minimum agreeing labels needed to treat ``from_team`` / ``to_team`` as temporally stable."""

    pass_unstable_team_as_unknown: bool = True
    """Prefer ``unknown_pass`` over ``intercepted_pass`` when team labels jitter on either player."""

    pass_kickoff_bootstrap_guard_enabled: bool = True
    """Output-only fix: first kickoff-era ``intercepted_pass`` may have unreliable passer IDs."""

    pass_kickoff_bootstrap_guard_max_frame: int = 70
    """Only applies the kickoff guard when ``start_frame <=`` this value."""

    pass_owner_min_stable_frames: int = 5
    """Consecutive possession frames defining a controlled ball owner."""

    pass_owner_switch_min_frames: int = 5
    """Frames a new possession id must persist (after transient window) before owner switch."""

    pass_transient_contact_max_frames: int = 4
    """Possession glimpses shorter than this are ignored for ownership / pass branching."""

    pass_release_min_frames_away: int = 3
    """Frames the ball must stay away from the owner's foot-point before ``release``."""

    pass_release_min_distance_px: float = 25.0
    """Distance (px) ball -> owner foot exceeding this counts as ``away``."""

    pass_release_min_ball_displacement_px: float = 20.0
    """Min ball displacement vs first ``away`` frame before arming pass candidate."""

    pass_receiver_confirm_frames: int = 5
    """Frames a receiver must stay as foot-radius candidate (with gap tolerance)."""

    pass_receiver_max_gap_frames: int = 2
    """Missed-foot-radius frames before receiver streak resets."""

    pass_receiver_control_radius_px: float = 70.0
    """Foot-point to ball center distance for inbound receiver hypothesis."""

    pass_candidate_timeout_frames: int = 90
    """In-flight candidate horizon without emission (discard, no forced pass)."""

    pass_post_receive_settle_frames: int = 12
    """After a completed pass: block the receiver's *normal* release path for N frames."""

    pass_post_receive_require_reconfirm: bool = True
    """When False: skip settle blocking (immediate normal release resumes)."""

    pass_allow_one_touch_release: bool = True
    """Allow a tightened one-touch pass during the settle block window."""

    pass_one_touch_window_frames: int = 8
    """One-touch path only while frame <= pass_end_frame + N."""

    pass_one_touch_min_away_frames: int = 3
    """Ball must leave receiver's tight control radius for N consecutive frames."""

    pass_one_touch_min_outgoing_displacement_px: float = 30.0
    """Min displacement from recorded contact anchoring."""

    pass_one_touch_receiver_confirm_frames: int = 4
    """Inbound receiver persisted under tight radius before emitting one-touch pass."""

    pass_one_touch_control_radius_px: float = 75.0
    """Receiver foot-ball proximity tolerance for establishing contact vs receiver."""

    pass_source_reliability_enabled: bool = True
    """Verify passer foot-ball proximity near ``release_frame`` before emitting."""

    pass_source_release_lookback_frames: int = 8
    """Frames before ``release_frame`` included when sampling passer-ball distance."""

    pass_source_max_release_distance_px: float = 115.0
    """Max passer-foot to ball-center distance (px) for a ``reliable`` source."""

    pass_source_unknown_if_unreliable: bool = True
    """Emit ``unknown_pass`` with preserved ``original_from_player_id`` when unreliable."""

    pass_touch_fallback_enabled: bool = True
    """Raw footpoint touch-to-touch pass recovery when the main FSM misses."""

    pass_touch_history_frames: int = 80
    """Rolling frames kept for touch episode construction."""

    pass_touch_source_lookback_frames: int = 14
    """Reserved for temporal filtering around source touch episodes."""

    pass_touch_receiver_lookahead_frames: int = 12
    """Defer supplemental emit until this many frames after receiver touch ends."""

    pass_touch_contact_radius_px: float = 90.0
    """Footpoint-to-ball distance for a nominal touch frame."""

    pass_touch_strong_contact_radius_px: float = 65.0
    """Strong touch threshold (footpoint–ball)."""

    pass_touch_min_source_contact_frames: int = 2

    pass_touch_min_receiver_contact_frames: int = 3

    pass_touch_min_displacement_px: float = 25.0

    pass_touch_max_duration_frames: int = 50
    """Max frames between source episode end and receiver episode start."""

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

    pass_debug: bool = False
    """Print pass-detector diagnostics and emitted pass lines to the terminal."""

    ball_roi_scan_debug: bool = False
    """Verbose terminal notices when ball ROI recovery scan runs."""

    debug_tracking: bool = False
    """Log extra inference / parse diagnostics."""

    debug_persons: bool = False
    """
    First-track style pipeline: only ``YOLO.track()`` boxes + role-colored draw.

    Disables ball-only raw ``predict``, ROI player recovery, ball debug overlay,
    temporal ball estimate, and jersey team coloring (see ``main.py`` wiring).
    """

    debug_player_tracking: bool = False
    """Emit terminal events for player recognition/loss/found transitions."""

    debug_only1_player: bool = False
    """If True, lock to the first recognized player id and log only that player."""

    debug_player_scope_margin_px: int = 20
    """Border margin for out-of-scope detection based on last visible bbox."""

    possession_proximity_px: float = 80.0
    """Max distance (pixels) from ball center to assign possession to a player."""

    team_classification_enabled: bool = True
    """If False, skip jersey-based team assignment (all players uncolored by team)."""

    team_torso_y0_ratio: float = 0.05
    """Top of torso crop: fraction of box height from box top (below hair/head)."""

    team_torso_y1_ratio: float = 0.45
    """Bottom of torso crop: fraction of box height from box top (upper chest / shirt)."""

    team_grass_hsv_lower: tuple[int, int, int] = (35, 40, 40)
    """OpenCV HSV lower bound for pitch green to mask out (H 0–179)."""

    team_grass_hsv_upper: tuple[int, int, int] = (85, 255, 255)
    """OpenCV HSV upper bound for pitch green."""

    team_grass_morph_kernel: int = 3
    """Odd kernel size for morphological open on grass mask (0 to disable)."""

    team_kmeans_min_samples: int = 4
    """Minimum players in frame to fit KMeans; below this, use prior team centers."""

    team_history_frames: int = 21
    """Per-track deque length for majority-vote stabilization (odd recommended)."""

    team_center_ema_alpha: float = 0.15
    """EMA weight to update global Lab team centers from each frame's KMeans centers."""

    enable_camera_pan: bool = True
    enable_top_down: bool = True
    """When True, run optional homography / mini-pitch mapping if implemented."""

    # Optional four image corners for pitch quad (x, y) in pixel coords; None = not configured.
    pitch_corners_image: Optional[tuple[tuple[float, float], ...]] = field(default=None)
    """Four points TL, TR, BR, BL in image space for homography (optional)."""

    def __post_init__(self) -> None:
        self.input_video = Path(self.input_video)
        self.output_video = Path(self.output_video)
        if self.tracks_log_path is not None:
            self.tracks_log_path = Path(self.tracks_log_path)
        if self.pass_events_jsonl_path is not None:
            self.pass_events_jsonl_path = Path(self.pass_events_jsonl_path)
        if self.throwin_events_jsonl_path is not None:
            self.throwin_events_jsonl_path = Path(self.throwin_events_jsonl_path)
