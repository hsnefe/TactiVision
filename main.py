#!/usr/bin/env python3
"""CLI entry: football broadcast analysis (starter scaffolding)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from tactivision.config.settings import Settings
from tactivision.pipeline.runner import AnalysisPipeline


def _default_output_path(input_path: Path) -> Path:
    stem = input_path.stem
    return input_path.with_name(f"{stem}_analyzed.mp4")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Football video analysis: YOLO + tracking + teams + possession (WIP)."
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
        default="yolov8n.pt",
        help="Ultralytics YOLO weights path or model name.",
    )
    p.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Detection confidence threshold.",
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

    settings = Settings(
        input_video=video_path,
        output_video=output_path,
        model_path=args.model,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        tracker_config=args.tracker,
        possession_proximity_px=args.possession_radius,
        enable_camera_pan=not args.no_camera_pan,
        enable_top_down=not args.no_topdown,
    )

    pipeline = AnalysisPipeline(settings)
    out = pipeline.run()
    print(f"Wrote: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
