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
    model_path: str = "yolov8n.pt"
    conf_threshold: float = 0.25
    iou_threshold: float = 0.45
    tracker_config: str = "bytetrack.yaml"
    """Ultralytics tracker YAML name (under ultralytics/cfg/trackers/)."""

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

    possession_proximity_px: float = 80.0
    """Max distance (pixels) from ball center to assign possession to a player."""

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
