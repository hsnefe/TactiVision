"""Runtime configuration for the analysis pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from tactivision.tracking.schema import ObjectRole


@dataclass
class Settings:
    """Paths, model options, and feature flags."""

    input_video: Path
    output_video: Path
    model_path: str = "yolov8s.pt"
    """Default YOLOv8 small — better small-object recall than ``yolov8n.pt``."""

    conf_threshold: float = 0.22
    """Main ``track()`` confidence; slightly below 0.25 to help marginal detections."""

    iou_threshold: float = 0.45
    tracker_config: str = "bytetrack.yaml"
    """Ultralytics tracker YAML name (under ultralytics/cfg/trackers/)."""

    inference_imgsz: int = 1280
    """Letterbox size for both ``track`` and ball-only ``predict`` (larger helps tiny ball)."""

    ball_conf_threshold: float = 0.12
    """Lower threshold on ball-class-only ``predict`` pass for recall / yellow debug overlay."""

    ball_max_gap_frames: int = 8
    """When track loses the ball, extrapolate position for this many frames (blue overlay)."""

    ball_debug_overlay: bool = True
    """Draw yellow raw / red tracked / blue estimated ball diagnostics on the output video."""

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
