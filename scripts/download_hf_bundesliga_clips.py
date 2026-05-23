#!/usr/bin/env python3
"""Download Bundesliga/Clips MP4s from Hugging Face (dbal0503/Bundesliga) into data/hf_bundesliga_clips."""

from __future__ import annotations

import argparse
import csv
import fnmatch
import sys
from collections import defaultdict
from pathlib import Path


CLIPS_PREFIX = "Bundesliga/Clips/"
DEFAULT_MANIFEST = Path("data/hf_bundesliga_clips_manifest.csv")
DEFAULT_OUTPUT = Path("data/hf_bundesliga_clips")


def normalize_repo_path(path: str) -> str:
    return path.replace("\\", "/")


def expected_local_clip(output_dir: Path, repo_path: str) -> Path:
    """Local path mirrored under ``output_dir`` (same layout as repo-relative ``repo_path``)."""

    return (output_dir.resolve() / normalize_repo_path(repo_path)).resolve()


def list_clips_mp4(api: object, repo_id: str, pattern: str) -> list[str]:
    """List dataset repo MP4 paths matching ``pattern`` (fnmatch, forward slashes)."""

    files: list[str] = api.list_repo_files(repo_id, repo_type="dataset")
    pat = pattern.replace("\\", "/")
    out: list[str] = []
    for f in files:
        fn = normalize_repo_path(f)
        if not fn.lower().endswith(".mp4"):
            continue
        if not fnmatch.fnmatch(fn, pat):
            continue
        out.append(fn)
    return sorted(out)


def list_all_bundesliga_clips_mp4(api: object, repo_id: str) -> list[str]:
    """Every ``.mp4`` whose path starts with ``Bundesliga/Clips/`` (any depth)."""

    files: list[str] = api.list_repo_files(repo_id, repo_type="dataset")
    prefix = CLIPS_PREFIX
    out: list[str] = []
    for f in files:
        fn = normalize_repo_path(f)
        if not fn.lower().endswith(".mp4"):
            continue
        if not fn.startswith(prefix):
            continue
        out.append(fn)
    return sorted(out)


def clip_prefix(repo_path: str) -> str:
    stem = Path(repo_path).stem
    if "_" in stem:
        return stem.split("_", 1)[0]
    return stem[:6] if len(stem) >= 6 else stem


def select_clips_first_n(paths: list[str], limit: int) -> tuple[list[str], str]:
    if limit <= 0:
        return [], "limit_zero"
    sel = paths[:limit]
    return sel, "first_n_sorted_paths"


def select_clips_balanced(paths: list[str], limit: int) -> tuple[list[str], str]:
    """Round-robin across prefix groups (e.g. 08fd33, 0a2d9b) until ``limit`` clips."""

    if limit <= 0:
        return [], "limit_zero"

    by_pfx: dict[str, list[str]] = defaultdict(list)
    for p in paths:
        by_pfx[clip_prefix(p)].append(p)
    for k in by_pfx:
        by_pfx[k].sort()

    prefixes = sorted(by_pfx.keys())
    if not prefixes:
        return [], "no_inputs"

    selected: list[str] = []
    round_idx = 0
    while len(selected) < limit:
        progressed = False
        for pfx in prefixes:
            if len(selected) >= limit:
                break
            bucket = by_pfx[pfx]
            if round_idx < len(bucket):
                selected.append(bucket[round_idx])
                progressed = True
        if not progressed:
            break
        round_idx += 1

    note = "balanced_prefix_round_robin" if len(prefixes) > 1 else "single_prefix_sequential"
    return selected, note


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Download Bundesliga clip MP4s from Hugging Face (dbal0503/Bundesliga). "
            "Writes data/hf_bundesliga_clips_manifest.csv by default."
        )
    )
    ap.add_argument("--repo-id", type=str, default="dbal0503/Bundesliga")
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT,
    )
    ap.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help="CSV manifest path (default: data/hf_bundesliga_clips_manifest.csv).",
    )
    ap.add_argument(
        "--all",
        action="store_true",
        help=f"Download every MP4 under {CLIPS_PREFIX!r} (any subfolder depth).",
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Max clips to include after selection. "
            "Without --all, default is 10. With --all, default is unlimited."
        ),
    )
    ap.add_argument(
        "--resume",
        action="store_true",
        help="Skip re-download when the target file already exists (still manifests it).",
    )
    ap.add_argument(
        "--skip-existing",
        action="store_true",
        help="Same behavior as --resume.",
    )
    ap.add_argument(
        "--pattern",
        type=str,
        default=f"{CLIPS_PREFIX}*.mp4",
        help="Ignored when --all. fnmatch pattern for repo-relative paths.",
    )
    ap.add_argument(
        "--selection",
        choices=("balanced", "first"),
        default="balanced",
        help="How to narrow to --limit: balanced prefixes or first N lexicographic.",
    )
    args = ap.parse_args()

    try:
        from huggingface_hub import HfApi, hf_hub_download  # type: ignore[import-untyped]
    except ImportError:
        print(
            "Install huggingface_hub: pip install huggingface_hub",
            file=sys.stderr,
        )
        return 1

    skip_if_present = args.resume or args.skip_existing

    if args.limit is None:
        effective_limit: int | None = None if args.all else 10
    else:
        effective_limit = args.limit

    api = HfApi()
    if args.all:
        all_matches = list_all_bundesliga_clips_mp4(api, args.repo_id)
    else:
        all_matches = list_clips_mp4(api, args.repo_id, args.pattern)

    total_remote = len(all_matches)

    if not all_matches:
        hint = CLIPS_PREFIX if args.all else args.pattern
        print(
            f"No MP4 paths matched ({hint!r}) under repo {args.repo_id!r}.",
            file=sys.stderr,
        )
        return 1

    lim = effective_limit if effective_limit is not None else len(all_matches)
    if args.selection == "first":
        chosen, strat_note = select_clips_first_n(all_matches, lim)
    else:
        chosen, strat_note = select_clips_balanced(all_matches, lim)

    if not chosen:
        print("No clips selected (--limit 0 or empty selection).", file=sys.stderr)
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)

    base_note = strat_note + (
        "; manual_test_media_only—not aligned with train.csv video_id rows"
    )

    manifest_rows: list[dict[str, str]] = []
    n_existing = 0
    n_downloaded = 0
    n_failed = 0

    for repo_path in chosen:
        pfx = clip_prefix(repo_path)
        stem = Path(repo_path).stem
        notes = base_note
        local_target = expected_local_clip(args.output_dir, repo_path)
        clip_path = ""

        if skip_if_present and local_target.is_file() and local_target.stat().st_size > 0:
            n_existing += 1
            clip_path = str(local_target)
            notes = f"skipped_existing; {base_note}"
        else:
            try:
                local_file = hf_hub_download(
                    repo_id=args.repo_id,
                    filename=repo_path,
                    repo_type="dataset",
                    local_dir=str(args.output_dir.resolve()),
                )
                clip_path = str(Path(local_file).resolve())
                n_downloaded += 1
                notes = base_note
            except Exception as exc:  # noqa: BLE001
                n_failed += 1
                notes = f"download_failed: {exc}; {base_note}"

        manifest_rows.append(
            {
                "clip_path": clip_path,
                "repo_path": repo_path,
                "clip_stem": stem,
                "prefix": pfx,
                "notes": notes,
            }
        )

    fieldnames = ["clip_path", "repo_path", "clip_stem", "prefix", "notes"]
    with args.manifest.open("w", newline="", encoding="utf-8") as fp:
        w = csv.DictWriter(fp, fieldnames=fieldnames)
        w.writeheader()
        for row in manifest_rows:
            w.writerow(row)

    print(f"total remote mp4 files: {total_remote}")
    print(f"already existing:       {n_existing}")
    print(f"downloaded:             {n_downloaded}")
    print(f"failed:                 {n_failed}")
    print(f"manifest rows:          {len(manifest_rows)}")
    print(f"Output dir: {args.output_dir.resolve()}")
    print(f"Manifest:   {args.manifest.resolve()}")

    if n_failed and not any(r["clip_path"] for r in manifest_rows):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
