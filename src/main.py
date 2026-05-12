#!/usr/bin/env python3
"""CLI entry: football broadcast analysis (YOLO + tracking + optional JSONL log)."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from VisionEngine.config.class_mapping import parse_class_role_overrides
from VisionEngine.config.settings import Settings
from VisionEngine.pipeline.runner import AnalysisPipeline


def _default_output_path(input_path: Path) -> Path:
    stem = input_path.stem
    return input_path.with_name(f"{stem}_analyzed.mp4")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Football video analysis: YOLO + tracking + teams + possession."
    )
    p.add_argument(
        "--video",
        type=Path,
        required=True,
        help="Path to input broadcast video.",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path for annotated MP4 (default: <input_stem>_analyzed.mp4).",
    )
    p.add_argument(
        "--model",
        type=str,
        default="yolo11n.pt",
        help="Ultralytics YOLO weights file (default: yolo11n.pt).",
    )
    p.add_argument(
        "--conf",
        type=float,
        default=0.22,
        help="Main track() confidence threshold.",
    )
    p.add_argument(
        "--imgsz",
        type=int,
        default=1280,
        help="Letterbox inference size (larger improves tiny ball recall, slower).",
    )
    p.add_argument(
        "--ball-conf",
        type=float,
        default=0.35,
        help="Lower threshold for ball-class-only raw predict pass (yellow debug boxes).",
    )
    p.add_argument(
        "--ball-gap",
        type=int,
        default=15,
        help="Max frames to extrapolate ball position when track drops (blue circle).",
    )
    p.add_argument(
        "--no-ball-debug",
        action="store_true",
        help="Disable yellow/red/blue ball debug overlay (faster draw).",
    )
    p.add_argument(
        "--roi-recovery",
        action="store_true",
        help=(
            "Enable the aggressive identity-preservation pipeline for players "
            "(relabel new IDs to recently-lost nearby IDs + ROI second-pass "
            "person predict). Off by default so a player re-detected with a "
            "new ID is rendered immediately (matches --debug_persons)."
        ),
    )
    p.add_argument(
        "--no-roi-recovery",
        action="store_true",
        help=(
            "Deprecated alias: ROI recovery is now off by default. Kept for "
            "backward compatibility; takes precedence over --roi-recovery."
        ),
    )
    p.add_argument(
        "--roi-conf",
        type=float,
        default=0.08,
        help="ROI-only person predict confidence (lower than main --conf).",
    )
    p.add_argument(
        "--roi-margin",
        type=float,
        default=0.4,
        help="Expand last bbox by this fraction of w/h before ROI crop.",
    )
    p.add_argument(
        "--roi-imgsz",
        type=int,
        default=1280,
        help="Letterbox size for ROI person predict.",
    )
    p.add_argument(
        "--roi-max",
        type=int,
        default=8,
        help="Max ROI recoveries per frame.",
    )
    p.add_argument(
        "--roi-max-lost-streak",
        type=int,
        default=120,
        help="Drop ROI state after this many consecutive primary misses for a track id.",
    )
    p.add_argument(
        "--roi-no-border-skip",
        action="store_true",
        help="Allow ROI recovery even when the last bbox touches the frame border.",
    )
    p.add_argument(
        "--iou",
        type=float,
        default=0.45,
        help="Detection IoU threshold.",
    )
    p.add_argument(
        "--tracker",
        type=str,
        default="bytetrack.yaml",
        help="Tracker config name for Ultralytics.",
    )
    p.add_argument(
        "--class-preset",
        type=str,
        default="coco_football",
        choices=("coco_football", "football_three_class"),
        help="How to map YOLO class names to player/referee/ball roles.",
    )
    p.add_argument(
        "--class-role-override",
        type=str,
        default=None,
        help='Optional overrides like "32:ball,0:player" (class index:role).',
    )
    p.add_argument(
        "--keep-other-classes",
        action="store_true",
        help="Keep YOLO classes mapped to role 'other' (default: drop them).",
    )
    p.add_argument(
        "--tracks-jsonl",
        type=Path,
        default=None,
        help="Optional path to write per-frame tracking JSON Lines.",
    )
    p.add_argument(
        "--debug-tracking",
        action="store_true",
        help="Print verbose tracker diagnostics.",
    )
    p.add_argument(
        "--debug_persons",
        action="store_true",
        help=(
            "First-track style run: only YOLO track() boxes + role colors. "
            "Disables ball raw predict, ROI recovery, ball debug overlay, "
            "ball gap estimate, and team jersey colors."
        ),
    )
    p.add_argument(
        "--debug_player_tracking",
        action="store_true",
        help="Print player recognized/out-of-scope/track-loss/found events to terminal.",
    )
    p.add_argument(
        "--debug_only1_player",
        action="store_true",
        help="With --debug_player_tracking, log only the first recognized player id.",
    )
    p.add_argument(
        "--possession-radius",
        type=float,
        default=80.0,
        help="Max pixel distance ball-to-player for possession.",
    )
    p.add_argument(
        "--no-camera-pan",
        action="store_true",
        help="Disable camera pan estimation stage.",
    )
    p.add_argument(
        "--no-topdown",
        action="store_true",
        help="Disable optional top-down / homography stage.",
    )
    p.add_argument(
        "--no-teams",
        action="store_true",
        help="Disable jersey-based team colors (players use default role color).",
    )
    p.add_argument(
        "--team-history",
        type=int,
        default=21,
        help="Frames of majority-vote history per track for team stabilization (odd).",
    )
    p.add_argument(
        "--no-ball-roi-recovery",
        action="store_false",
        dest="ball_roi_recovery",
        help="Disables Region of Interest (ROI) scanning when the ball is lost.",
    )
    p.add_argument(
        "--ball-roi-conf",
        type=float,
        default=0.08,
        help="Low confidence threshold for ball ROI scan.",
    )
    p.add_argument(
        "--ball-roi-size",
        type=int,
        default=300,
        help="Pixel size (width/height) of the cropped region to search for the ball.",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    video_path: Path = args.video.expanduser().resolve()
    if not video_path.is_file():
        print(f"Input video not found: {video_path}", file=sys.stderr)
        return 1

    output_path = args.output
    if output_path is None:
        output_path = _default_output_path(video_path)
    else:
        output_path = output_path.expanduser().resolve()

    if args.debug_tracking:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    overrides = parse_class_role_overrides(args.class_role_override)

    debug_persons = bool(args.debug_persons)
    ball_debug = not args.no_ball_debug and not debug_persons
    roi_on = bool(args.roi_recovery) and not args.no_roi_recovery and not debug_persons
    teams_on = not args.no_teams and not debug_persons

    settings = Settings(
        input_video=video_path,
        output_video=output_path,
        model_path=args.model,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        tracker_config=args.tracker,
        inference_imgsz=args.imgsz,
        ball_conf_threshold=args.ball_conf,
        ball_max_gap_frames=args.ball_gap,
        ball_debug_overlay=ball_debug,
        roi_recovery_enabled=roi_on,
        debug_persons=debug_persons,
        roi_conf_threshold=args.roi_conf,
        roi_margin_ratio=args.roi_margin,
        roi_inference_imgsz=args.roi_imgsz,
        roi_max_per_frame=args.roi_max,
        roi_max_lost_streak=args.roi_max_lost_streak,
        roi_skip_near_border=not args.roi_no_border_skip,
        class_mapping_preset=args.class_preset,
        class_role_overrides=overrides,
        only_mapped_classes=not args.keep_other_classes,
        tracks_log_path=args.tracks_jsonl,
        debug_tracking=args.debug_tracking,
        debug_player_tracking=args.debug_player_tracking,
        debug_only1_player=args.debug_only1_player,
        possession_proximity_px=args.possession_radius,
        enable_camera_pan=not args.no_camera_pan,
        enable_top_down=not args.no_topdown,
        team_classification_enabled=teams_on,
        team_history_frames=args.team_history,
        ball_roi_recovery=args.ball_roi_recovery,
        ball_roi_conf=args.ball_roi_conf,
        ball_roi_size=args.ball_roi_size,
    )

    pipeline = AnalysisPipeline(settings)
    out = pipeline.run()
    print(f"Wrote: {out}")
    if settings.tracks_log_path:
        print(f"Tracks log: {settings.tracks_log_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
