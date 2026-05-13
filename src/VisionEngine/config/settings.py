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

    field_detection_enabled: bool = False
    """
    When True, run the Roboflow ``football-field-detection-f07vi`` keypoint
    pose model to (a) build a pitch polygon and filter off-field PLAYER/REFEREE
    detections, and (b) compute an image->pitch homography that powers the
    existing :class:`FieldMapper`.
    """

    field_model_path: Optional[Path] = field(default=Path("models/field-keypoints.pt"))
    """Local Ultralytics ``.pt`` weights for the pitch keypoint model. If the
    file does not exist and ``field_use_remote_fallback`` is True, the system
    falls back to the Roboflow Inference HTTP API."""

    field_model_id: str = "football-field-detection-f07vi/15"
    """Roboflow Universe model id used for the hosted Inference API fallback."""

    field_model_imgsz: int = 1280
    """Letterbox size for the keypoint model ``predict`` pass."""

    field_model_conf: float = 0.30
    """Confidence threshold for the keypoint pose ``predict`` pass."""

    field_kp_min_conf: float = 0.50
    """Per-keypoint visibility threshold; below this a keypoint is treated as
    not visible and excluded from homography solving."""

    field_recompute_every_n_frames: int = 30
    """Re-detect pitch keypoints every N frames; between recomputes the cached
    polygon and homography are reused (broadcast cameras pan slowly relative
    to per-frame detector cost)."""

    field_mask_expand_ratio: float = 0.05
    """Expand the pitch polygon by this fraction of its bbox diagonal before
    point-in-polygon tests, so players right on the touchline aren't dropped."""

    field_use_remote_fallback: bool = True
    """If True and ``field_model_path`` is missing, fall back to the Roboflow
    hosted Inference API using ``roboflow_api_key_env``."""

    roboflow_api_key_env: str = "ROBOFLOW_API_KEY"
    """Environment variable name read for the Roboflow API key (used by the
    download helper and by the hosted inference fallback)."""

    def __post_init__(self) -> None:
        self.input_video = Path(self.input_video)
        self.output_video = Path(self.output_video)
        if self.tracks_log_path is not None:
            self.tracks_log_path = Path(self.tracks_log_path)
        if self.field_model_path is not None:
            self.field_model_path = Path(self.field_model_path)
