#!/usr/bin/env python3
"""Football pitch keypoint pose: prepare data.yaml, train, validate, and run video inference.

This script expects a Roboflow-style YOLOv8 pose export (or compatible layout):
  <dataset_root>/
    data.yaml
    train/images, train/labels
    valid/ or val/ with images & labels
    test/ (optional)

It rewrites ``data.yaml`` so ``path``, ``train``, ``val``, and ``test`` point at real
directories on disk (YOLO resolves ``train``/``val``/``test`` relative to ``path``).

Usage examples:

    # Default desktop dataset folder, train + val + test inference video
    python scripts/football_pose_train_pipeline.py \\
        --dataset ~/Desktop/football-field-detection.v12i.yolov8 \\
        --video ~/Desktop/sample_match.mp4

    # ZIP archive (extracted next to the zip unless --extract-to is set)
    python scripts/football_pose_train_pipeline.py \\
        --dataset ~/Desktop/football-field-detection.v12i.yolov8.zip \\
        --video ~/Desktop/clips/clip01.mp4

    # Skip training (only patch yaml + optional inference with existing weights)
    python scripts/football_pose_train_pipeline.py --dataset DATA --weights runs/pose/foo/weights/best.pt --video VID --no-train

Requires: ``pip install ultralytics`` (and a compatible PyTorch build for GPU).
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import zipfile
from pathlib import Path
from typing import Any

import numpy as np

# Default augmentations for pitch keypoints (passed through to ``model.train()``).
TRAIN_AUGMENT_KWARGS: dict[str, float] = {
    "degrees": 5.0,
    "translate": 0.1,
    "scale": 0.5,
    "shear": 2.0,
    "perspective": 0.0005,
    "hsv_h": 0.015,
    "hsv_s": 0.7,
    "hsv_v": 0.4,
    "fliplr": 0.0,
    "flipud": 0.0,
    "mosaic": 0.0,
    "mixup": 0.0,
}

# Keypoint confidence below this is written as ``null`` coordinates in JSON (still listed in ``confidence``).
DEFAULT_KPT_CONF_THRESHOLD = 0.6

# ---------------------------------------------------------------------------
# Optional heavy imports — fail fast with a clear message if ultralytics/torch missing.
# ---------------------------------------------------------------------------


def _require_yaml() -> Any:
    try:
        import yaml  # type: ignore
    except ImportError as e:
        print(
            "Error: PyYAML is required. Install with: pip install pyyaml",
            file=sys.stderr,
        )
        raise SystemExit(2) from e
    return yaml


def _require_ultralytics():
    try:
        from ultralytics import YOLO
    except ImportError as e:
        print(
            "Error: ultralytics is required. Install with: pip install ultralytics",
            file=sys.stderr,
        )
        raise SystemExit(2) from e
    return YOLO


def _resolve_device(preference: str) -> str | int:
    """Return a device string for Ultralytics (``'cpu'``, ``'0'``, ``'0,1'``, etc.)."""
    pref = preference.strip().lower()
    if pref in ("cpu", "mps"):
        return pref
    try:
        import torch
    except ImportError:
        return "cpu"

    if pref == "auto":
        if torch.cuda.is_available():
            return "0"
        # Apple Silicon MPS (pose is supported in recent torch; ultralytics will fall back if unsupported)
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    if pref.startswith("cuda") or pref.isdigit():
        return pref
    return "cpu"


# ---------------------------------------------------------------------------
# Dataset location: ZIP extraction + locating data.yaml
# ---------------------------------------------------------------------------


def ensure_dataset_dir(dataset_arg: Path, extract_to: Path | None) -> Path:
    """Return a directory containing ``data.yaml``; extract ZIP if needed."""
    dataset_arg = dataset_arg.expanduser().resolve()
    if not dataset_arg.exists():
        print(f"Error: dataset path does not exist: {dataset_arg}", file=sys.stderr)
        raise SystemExit(2)

    if dataset_arg.is_dir():
        return dataset_arg

    if dataset_arg.suffix.lower() == ".zip":
        dest = (extract_to.expanduser().resolve() if extract_to else dataset_arg.with_suffix(""))
        if dest.exists() and any(dest.iterdir()):
            print(f"Using existing extract directory: {dest}")
        else:
            dest.mkdir(parents=True, exist_ok=True)
            print(f"Extracting {dataset_arg} -> {dest}")
            with zipfile.ZipFile(dataset_arg, "r") as zf:
                zf.extractall(dest)
        return dest

    print(f"Error: unsupported dataset path (not a directory or .zip): {dataset_arg}", file=sys.stderr)
    raise SystemExit(2)


def find_data_yaml(root: Path) -> Path:
    """Find ``data.yaml`` or ``data.yml`` under the dataset root."""
    for name in ("data.yaml", "data.yml"):
        p = root / name
        if p.is_file():
            return p
    # Some exports nest yaml one level deep
    for candidate in root.rglob("data.y*ml"):
        if candidate.is_file():
            return candidate
    print(f"Error: could not find data.yaml under {root}", file=sys.stderr)
    raise SystemExit(2)


# ---------------------------------------------------------------------------
# Path rewriting for YOLO data.yaml
# ---------------------------------------------------------------------------


def _detect_image_folder_for_split(dataset_root: Path, split: str) -> str | None:
    """Infer ``train``/``val``/``test`` path segments relative to ``dataset_root``."""
    split_aliases = {
        "train": ["train"],
        "val": ["val", "valid"],
        "test": ["test"],
    }[split]

    for alias in split_aliases:
        base = dataset_root / alias
        if not base.is_dir():
            continue
        imgs = base / "images"
        if imgs.is_dir():
            return f"{alias}/images"
        # Rare layouts: images directly under split name
        return alias
    return None


def _resolve_old_train_path(
    yaml_path: Path,
    data: dict[str, Any],
    split_key: str,
) -> Path | None:
    """Resolve where an existing YAML split pointed on disk before we rewrite it."""
    raw = data.get(split_key)
    if split_key == "val" and (not raw or not isinstance(raw, str)):
        raw = data.get("valid")
    if not raw or not isinstance(raw, str):
        return None

    path_key = data.get("path")
    roots: list[Path] = []
    if path_key:
        pk = Path(path_key)
        roots.append(pk if pk.is_absolute() else (yaml_path.parent / pk).resolve())
    roots.append(yaml_path.parent.resolve())

    for root in roots:
        candidate = (root / raw).resolve()
        if candidate.is_dir():
            return candidate
    # Try raw relative to yaml only
    cand2 = (yaml_path.parent / raw).resolve()
    if cand2.is_dir():
        return cand2
    return None


def patch_data_yaml(yaml_path: Path, dataset_root: Path, overwrite: bool = True) -> dict[str, Any]:
    """Load ``data.yaml``, set ``path`` and split folders, write back if ``overwrite``."""
    yaml = _require_yaml()
    dataset_root = dataset_root.resolve()
    yaml_path = yaml_path.resolve()

    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        print(f"Error: expected mapping in {yaml_path}", file=sys.stderr)
        raise SystemExit(2)

    # Ultralytics expects ``val``; Roboflow exports often use ``valid``.
    if "val" not in data and isinstance(data.get("valid"), str):
        data["val"] = data["valid"]

    # Resolve / validate each split; prefer detection if existing path is wrong.
    splits = ("train", "val", "test")
    new_splits: dict[str, str] = {}
    for split in splits:
        resolved_old = _resolve_old_train_path(yaml_path, data, split)
        rel: str | None = None
        if resolved_old and resolved_old.exists():
            try:
                rel = str(resolved_old.resolve().relative_to(dataset_root))
            except ValueError:
                rel = None
        if rel is None:
            guessed = _detect_image_folder_for_split(dataset_root, split)
            if guessed:
                rel = guessed
        if rel is not None:
            new_splits[split] = rel

    # ``test`` is optional — only set if we found images (keeps val() happy if no test set)
    if "test" in new_splits:
        test_dir = dataset_root / new_splits["test"]
        if not test_dir.is_dir():
            del new_splits["test"]

    data["path"] = str(dataset_root)
    for k, v in new_splits.items():
        data[k] = v
    # Avoid duplicate split keys in the written file.
    if "valid" in data and "val" in data:
        del data["valid"]

    # Basic sanity check
    for sk in ("train", "val"):
        if sk not in data:
            print(
                f"Error: could not resolve '{sk}' images folder under {dataset_root}",
                file=sys.stderr,
            )
            raise SystemExit(2)
        if not (dataset_root / data[sk]).is_dir():
            print(
                f"Error: {sk} path does not exist: {dataset_root / data[sk]}",
                file=sys.stderr,
            )
            raise SystemExit(2)

    if overwrite:
        backup = yaml_path.with_suffix(yaml_path.suffix + ".bak")
        if yaml_path.exists() and not backup.exists():
            try:
                shutil.copy2(yaml_path, backup)
            except OSError:
                pass
        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, sort_keys=False, allow_unicode=True)
        print(f"Wrote updated dataset config: {yaml_path}")
    return data


# ---------------------------------------------------------------------------
# Train / val / predict
# ---------------------------------------------------------------------------


def run_train(
    YOLO,
    data_yaml: Path,
    model_name: str,
    epochs: int,
    imgsz: int,
    batch: int,
    device: str | int,
    project: Path,
    run_name: str,
    workers: int,
    resume: bool,
):
    """Train a pose model; return path to ``best.pt`` and the Ultralytics metrics object."""
    model = YOLO(model_name)
    train_kw: dict[str, Any] = {
        "data": str(data_yaml),
        "epochs": epochs,
        "imgsz": imgsz,
        "batch": batch,
        "device": device,
        "project": str(project),
        "name": run_name,
        "workers": workers,
        **TRAIN_AUGMENT_KWARGS,
    }
    if resume:
        train_kw["resume"] = True
    results = model.train(**train_kw)
    save_dir = Path(getattr(results, "save_dir", project / run_name))
    best_pt = save_dir / "weights" / "best.pt"
    if not best_pt.is_file():
        # Older / alternate layouts
        alt = project / run_name / "weights" / "best.pt"
        if alt.is_file():
            best_pt = alt
        else:
            print(f"Warning: expected weights not found at {best_pt}", file=sys.stderr)
    return best_pt, results


def run_validate(YOLO, weights: Path, data_yaml: Path, imgsz: int, batch: int, device: str | int):
    """Evaluate metrics on the val split using saved weights."""
    model = YOLO(str(weights))
    return model.val(
        data=str(data_yaml),
        imgsz=imgsz,
        batch=batch,
        device=device,
        verbose=True,
    )


def print_validation_metrics(metrics: Any) -> None:
    """Print pose / box validation metrics so results stay visible in the console."""
    sep = "=" * 72
    print(f"\n{sep}\nValidation summary\n{sep}", flush=True)

    results_dict = getattr(metrics, "results_dict", None)
    if callable(results_dict):
        try:
            results_dict = results_dict()
        except TypeError:
            results_dict = None
    if isinstance(results_dict, dict) and results_dict:
        print("\nresults_dict:", flush=True)
        for key in sorted(results_dict.keys()):
            print(f"  {key}: {results_dict[key]}", flush=True)

    speed = getattr(metrics, "speed", None)
    if speed is not None:
        print(f"\nInference speed: {speed}", flush=True)

    for sub_name in ("box", "pose"):
        sub = getattr(metrics, sub_name, None)
        if sub is None:
            continue
        print(f"\nmetrics.{sub_name}:", flush=True)
        for attr in (
            "p",
            "r",
            "f1",
            "mp",
            "map50",
            "map",
            "maps",
            "precision",
            "recall",
        ):
            if hasattr(sub, attr):
                print(f"  {attr}: {getattr(sub, attr)}", flush=True)

    summary_fn = getattr(metrics, "summary", None)
    if callable(summary_fn):
        try:
            print(f"\n{summary_fn()}", flush=True)
        except Exception:
            pass

    print(sep + "\n", flush=True)


def _pose_best_index(keypoints: Any) -> int:
    """If multiple pose instances exist, pick the one with highest mean keypoint confidence."""
    xy = getattr(keypoints, "xy", None)
    if xy is None:
        return -1
    n = int(xy.shape[0]) if hasattr(xy, "shape") else 0
    if n <= 0:
        return -1
    if n == 1:
        return 0
    conf = getattr(keypoints, "conf", None)
    if conf is None:
        return 0
    c = conf.cpu().numpy() if hasattr(conf, "cpu") else np.asarray(conf)
    mean_c = c.mean(axis=-1)
    return int(np.argmax(mean_c))


def extract_frame_pose_dict(
    result: Any,
    conf_threshold: float,
) -> dict[str, Any]:
    """Build one JSON object for a single frame (without ``frame`` index)."""
    empty = {"keypoints": [None] * 32, "confidence": [0.0] * 32}

    kps = getattr(result, "keypoints", None)
    out_kp: list[Any] = []
    out_cf: list[float] = []

    if kps is None:
        return empty

    idx = _pose_best_index(kps)
    if idx < 0:
        return empty

    orig_shape = getattr(result, "orig_shape", None)
    if orig_shape is not None and len(orig_shape) >= 2:
        h_f, w_f = float(orig_shape[0]), float(orig_shape[1])
    else:
        w_f, h_f = 1.0, 1.0

    xy = kps.xy[idx]
    conf = kps.conf[idx] if hasattr(kps, "conf") and kps.conf is not None else None

    xy_np = xy.cpu().numpy() if hasattr(xy, "cpu") else xy
    conf_np = conf.cpu().numpy() if conf is not None and hasattr(conf, "cpu") else conf

    xy_np = np.asarray(xy_np, dtype=np.float32)
    n = xy_np.shape[0]
    conf_np = np.asarray(conf_np, dtype=np.float32).reshape(-1) if conf_np is not None else np.ones(n, dtype=np.float32)

    for i in range(n):
        x, y = float(xy_np[i, 0]), float(xy_np[i, 1])
        x, y = x / w_f, y / h_f
        c = float(conf_np[i]) if i < len(conf_np) else 0.0
        out_cf.append(c)
        if c >= conf_threshold:
            out_kp.append([x, y])
        else:
            out_kp.append(None)

    return {"keypoints": out_kp, "confidence": out_cf}


def run_predict(
    YOLO,
    weights: Path,
    video: Path,
    imgsz: int,
    device: str | int,
    project: Path,
    name: str,
    json_out: Path,
    kpt_conf_threshold: float,
):
    """Run pose inference on a video, save overlays, and write filtered keypoints to JSON."""
    model = YOLO(str(weights))
    predict_results = model.predict(
        source=str(video),
        imgsz=imgsz,
        device=device,
        save=True,
        project=str(project),
        name=name,
        stream=True,
    )

    frames_out: list[dict[str, Any]] = []
    for frame_idx, r in enumerate(predict_results):
        d = extract_frame_pose_dict(r, conf_threshold=kpt_conf_threshold)
        frames_out.append(
            {
                "frame": frame_idx,
                "keypoints": d["keypoints"],
                "confidence": d["confidence"],
            }
        )

    json_out.parent.mkdir(parents=True, exist_ok=True)
    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(frames_out, f, indent=2, allow_nan=False)
    print(f"Saved keypoint JSON ({len(frames_out)} frames): {json_out}", flush=True)

    return json_out


def build_arg_parser() -> argparse.ArgumentParser:
    desktop = Path.home() / "Desktop"
    default_dataset = desktop / "football-field-detection.v12i.yolov8"

    p = argparse.ArgumentParser(
        description="Prepare data.yaml, train YOLO pose, validate, and run video inference.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--dataset",
        type=Path,
        default=default_dataset,
        help="Path to dataset folder or .zip (Roboflow export).",
    )
    p.add_argument(
        "--extract-to",
        type=Path,
        default=None,
        help="If --dataset is a zip, extract here instead of next to the archive.",
    )
    p.add_argument(
        "--model",
        type=str,
        default="yolo11m-pose.pt",
        help="Ultralytics pose weights (built-in name or local .pt).",
    )
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--imgsz", type=int, default=960)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument(
        "--device",
        type=str,
        default="auto",
        help="auto | cpu | mps | 0 | 0,1 | cuda:0 — passed to Ultralytics.",
    )
    p.add_argument("--workers", type=int, default=8, help="DataLoader workers.")
    p.add_argument(
        "--project",
        type=Path,
        default=Path("runs/pose"),
        help="Ultralytics project directory for train/predict runs.",
    )
    p.add_argument("--name", type=str, default="football_field", help="Run name under --project.")
    p.add_argument("--resume", action="store_true", help="Resume last training run (Ultralytics).")
    p.add_argument("--no-train", action="store_true", help="Skip training (still patch yaml).")
    p.add_argument(
        "--weights",
        type=Path,
        default=None,
        help="Explicit weights for val/predict when --no-train or after external training.",
    )
    p.add_argument(
        "--video",
        type=Path,
        default=None,
        help="Local video file for inference (required unless --skip-predict).",
    )
    p.add_argument(
        "--predict-name",
        type=str,
        default="predict_video",
        help="Subfolder under --project for saved prediction outputs.",
    )
    p.add_argument("--skip-val", action="store_true", help="Skip model.val() after training.")
    p.add_argument("--skip-predict", action="store_true", help="Skip video inference.")
    p.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Output JSON path for per-frame keypoints (default: <project>/<predict-name>/pitch_keypoints.json).",
    )
    p.add_argument(
        "--kpt-conf-threshold",
        type=float,
        default=DEFAULT_KPT_CONF_THRESHOLD,
        help="Keypoints with lower confidence are written as null coordinates in JSON.",
    )
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    YOLO = _require_ultralytics()

    dataset_root = ensure_dataset_dir(args.dataset, args.extract_to)
    data_yaml = find_data_yaml(dataset_root)
    # If yaml lived in a subfolder, dataset_root for YOLO should still be folder containing splits.
    # Roboflow usually has yaml at root — patch in place.
    patch_data_yaml(data_yaml, dataset_root, overwrite=True)

    device = _resolve_device(args.device)
    print(f"Using device: {device}")

    best_pt: Path | None = None
    if not args.no_train:
        best_pt, _train_metrics = run_train(
            YOLO,
            data_yaml=data_yaml,
            model_name=args.model,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=device,
            project=args.project,
            run_name=args.name,
            workers=args.workers,
            resume=args.resume,
        )
        print(f"Training finished. Best weights: {best_pt}")
    else:
        best_pt = args.weights

    if best_pt is None and args.weights is not None:
        best_pt = args.weights

    if best_pt is None or not Path(best_pt).is_file():
        print(
            "Error: no weights found. Train first or pass --weights /path/to/best.pt",
            file=sys.stderr,
        )
        raise SystemExit(2)
    best_pt = Path(best_pt).expanduser().resolve()

    if not args.skip_val:
        print("Running validation (model.val)…", flush=True)
        val_metrics = run_validate(
            YOLO,
            weights=best_pt,
            data_yaml=data_yaml,
            imgsz=args.imgsz,
            batch=args.batch,
            device=device,
        )
        print_validation_metrics(val_metrics)

    if args.skip_predict:
        print("Done.")
        return

    if args.video is None:
        print("Error: --video is required for inference (or pass --skip-predict).", file=sys.stderr)
        raise SystemExit(2)

    video_path = args.video.expanduser().resolve()
    if not video_path.is_file():
        print(f"Error: video not found: {video_path}", file=sys.stderr)
        raise SystemExit(2)
    print(f"Running inference on {video_path} …", flush=True)
    json_path = args.json_out
    if json_path is None:
        json_path = args.project / args.predict_name / "pitch_keypoints.json"
    else:
        json_path = json_path.expanduser().resolve()

    run_predict(
        YOLO,
        weights=best_pt,
        video=video_path,
        imgsz=args.imgsz,
        device=device,
        project=args.project,
        name=args.predict_name,
        json_out=json_path,
        kpt_conf_threshold=args.kpt_conf_threshold,
    )
    out_dir = args.project / args.predict_name
    print(f"Saved prediction visualizations under: {out_dir}", flush=True)

    print("Done.")


if __name__ == "__main__":
    main()