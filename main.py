#!/usr/bin/env python3
"""CLI entry: football broadcast analysis (YOLO + tracking + optional JSONL log)."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from tactivision.config.class_mapping import parse_class_role_overrides
from tactivision.config.settings import Settings
from tactivision.pipeline.runner import AnalysisPipeline


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
        default="yolov8s.pt",
        help="Ultralytics YOLO weights (yolov8s recommended over nano for small ball).",
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
        default=0.12,
        help="Lower threshold for ball-class-only raw predict pass (yellow debug boxes).",
    )
    p.add_argument(
        "--ball-gap",
        type=int,
        default=8,
        help="Max frames to extrapolate ball position when track drops (blue circle).",
    )
    p.add_argument(
        "--no-ball-debug",
        action="store_true",
        help="Disable yellow/red/blue ball debug overlay (faster draw).",
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
        ball_debug_overlay=not args.no_ball_debug,
        class_mapping_preset=args.class_preset,
        class_role_overrides=overrides,
        only_mapped_classes=not args.keep_other_classes,
        tracks_log_path=args.tracks_jsonl,
        debug_tracking=args.debug_tracking,
        possession_proximity_px=args.possession_radius,
        enable_camera_pan=not args.no_camera_pan,
        enable_top_down=not args.no_topdown,
    )

    pipeline = AnalysisPipeline(settings)
    out = pipeline.run()
    print(f"Wrote: {out}")
    if settings.tracks_log_path:
        print(f"Tracks log: {settings.tracks_log_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
