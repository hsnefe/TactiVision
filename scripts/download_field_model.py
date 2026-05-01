#!/usr/bin/env python3
"""Download Roboflow ``football-field-detection-f07vi`` weights for offline use.

Run this once before using ``--mode field`` if you want fully offline pitch
keypoint detection. The script reads your Roboflow API key from the
``ROBOFLOW_API_KEY`` environment variable, downloads the v15 dataset
(which includes the trained YOLOv8-pose weights file ``best.pt``), and
copies the weights into ``models/field-keypoints.pt`` at the repo root.

Usage:
    set ROBOFLOW_API_KEY=...        # PowerShell: $env:ROBOFLOW_API_KEY="..."
    python scripts/download_field_model.py
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

DEFAULT_WORKSPACE = "roboflow-jvuqo"
DEFAULT_PROJECT = "football-field-detection-f07vi"
DEFAULT_VERSION = 15
DEFAULT_OUTPUT = Path("models/field-keypoints.pt")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workspace", default=DEFAULT_WORKSPACE)
    p.add_argument("--project", default=DEFAULT_PROJECT)
    p.add_argument("--version", type=int, default=DEFAULT_VERSION)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument(
        "--api-key-env",
        default="ROBOFLOW_API_KEY",
        help="Environment variable to read the Roboflow API key from.",
    )
    return p


def _resolve_api_key(env_var: str) -> str:
    api_key = os.environ.get(env_var, "").strip()
    if not api_key:
        print(
            f"Error: ${env_var} is not set. Get a key from "
            "https://app.roboflow.com/settings/api and export it before running this script.",
            file=sys.stderr,
        )
        sys.exit(2)
    return api_key


def _download_weights(
    workspace: str,
    project: str,
    version: int,
    api_key: str,
) -> Path:
    try:
        from roboflow import Roboflow
    except ImportError:
        print(
            "Error: the `roboflow` package is required. Install with:\n"
            "    pip install roboflow",
            file=sys.stderr,
        )
        sys.exit(2)

    rf = Roboflow(api_key=api_key)
    project_obj = rf.workspace(workspace).project(project)
    version_obj = project_obj.version(version)
    dataset = version_obj.download("yolov8")
    location = Path(dataset.location)

    candidates = [
        location / "weights" / "best.pt",
        location / "best.pt",
        location / "runs" / "pose" / "train" / "weights" / "best.pt",
    ]
    for cand in candidates:
        if cand.is_file():
            return cand
    matches = list(location.rglob("*.pt"))
    if matches:
        return matches[0]
    print(
        f"Error: could not locate a .pt weights file under {location}.",
        file=sys.stderr,
    )
    sys.exit(2)


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    api_key = _resolve_api_key(args.api_key_env)
    weights_path = _download_weights(args.workspace, args.project, args.version, api_key)

    output: Path = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(weights_path, output)
    print(f"Wrote {output} (source: {weights_path})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
