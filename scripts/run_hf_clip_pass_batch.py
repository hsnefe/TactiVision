#!/usr/bin/env python3
"""Batch-run TactiVision pass detection on HF Bundesliga clips from a manifest CSV."""

from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path


DEFAULT_MANIFEST = Path("data/hf_bundesliga_clips_manifest.csv")
DEFAULT_OUTPUT_ROOT = Path("outputs/hf_clip_pass_tests_all_debug")


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def sanitize_clip_stem(name: str, *, max_len: int = 120) -> str:
    """Folder/file token: spaces → _, drop (), keep only [A-Za-z0-9_-]."""

    s = name.strip()
    if not s:
        return "clip"
    s = s.replace(" ", "_")
    s = s.replace("(", "").replace(")", "")
    s = re.sub(r"[^a-zA-Z0-9_-]+", "", s)
    return (s or "clip")[:max_len]


def load_manifest(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with path.open(newline="", encoding="utf-8") as fp:
        reader = csv.DictReader(fp)
        for r in reader:
            rows.append({k: (v or "").strip() for k, v in r.items()})
    return rows


def resolve_clip_path(root: Path, manifest_path: Path, clip_path_str: str) -> Path | None:
    clip_path_str = (clip_path_str or "").strip()
    if not clip_path_str:
        return None
    p = Path(clip_path_str).expanduser()
    if p.is_file():
        return p.resolve()
    cand = root / clip_path_str
    if cand.is_file():
        return cand.resolve()
    alt = manifest_path.parent / clip_path_str
    if alt.is_file():
        return alt.resolve()
    return None


def count_pass_events_jsonl(path: Path) -> tuple[int, str]:
    if not path.is_file():
        return -1, "missing_jsonl"
    n = 0
    try:
        with path.open(encoding="utf-8") as fp:
            for line in fp:
                line = line.strip()
                if not line:
                    continue
                n += 1
        return n, ""
    except OSError as e:
        return -1, f"read_error:{e}"


def effective_limit(limit: int | None, use_all: bool) -> int | None:
    """None = no cap (only with --all). Without --all, default first N rows."""

    if limit is not None:
        return limit
    if use_all:
        return None
    return 5


def select_manifest_slice(
    rows: list[dict[str, str]],
    *,
    start_index: int,
    limit: int | None,
    use_all: bool,
) -> list[dict[str, str]]:
    eff = effective_limit(limit, use_all)
    tail = rows[start_index:]
    if eff is None:
        return tail
    return tail[:eff]


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Run TactiVision pass detection (--pass-debug) on HF clip manifest rows."
        )
    )
    ap.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
    )
    ap.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
    )
    ap.add_argument(
        "--all",
        action="store_true",
        help="Process every row in the manifest (after --start-index), optional --limit caps.",
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="Max rows to process after --start-index. With --all, default is unlimited; "
        "without --all, default is 5.",
    )
    ap.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="0-based offset into the manifest before applying --limit.",
    )
    ap.add_argument(
        "--resume",
        action="store_true",
        help="Skip clips whose output video, JSONL, and log already exist.",
    )
    ap.add_argument(
        "--skip-existing",
        action="store_true",
        help="Same as --resume.",
    )
    ap.add_argument(
        "--python",
        type=str,
        default="python3",
        metavar="EXE",
        help="Python executable to run src/main.py.",
    )
    args = ap.parse_args()

    skip_if_done = args.resume or args.skip_existing

    root = repo_root()
    manifest_path = args.manifest
    if not manifest_path.is_file():
        manifest_path = (root / manifest_path).resolve()
    if not manifest_path.is_file():
        print(f"Manifest not found: {args.manifest}", file=sys.stderr)
        return 1

    all_rows = load_manifest(manifest_path)
    out_root = args.output_root
    if not out_root.is_absolute():
        out_root = (root / out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    main_py = root / "src" / "main.py"
    if not main_py.is_file():
        print(f"Expected {main_py}", file=sys.stderr)
        return 1

    if args.start_index < 0:
        print("--start-index must be >= 0", file=sys.stderr)
        return 1
    if args.start_index > len(all_rows):
        print(
            f"--start-index={args.start_index} past manifest length {len(all_rows)}.",
            file=sys.stderr,
        )
        return 1

    selected = select_manifest_slice(
        all_rows,
        start_index=args.start_index,
        limit=args.limit,
        use_all=args.all,
    )
    total = len(selected)

    env_base = os.environ.copy()
    env_base["PYTHONPATH"] = str((root / "src").resolve())

    dirname_counts: dict[str, int] = defaultdict(int)
    summary_rows: list[dict[str, str]] = []

    for i, r in enumerate(selected, start=1):
        clip_path_str = (r.get("clip_path") or "").strip()
        stem_raw = (r.get("clip_stem") or "").strip()

        vid_path = resolve_clip_path(root, manifest_path, clip_path_str)
        if vid_path is None:
            if not stem_raw and clip_path_str:
                stem_raw = Path(clip_path_str).stem
            elif not stem_raw:
                stem_raw = "missing"
            safe_base = sanitize_clip_stem(stem_raw)
            n = dirname_counts[safe_base]
            dirname_counts[safe_base] += 1
            safe = safe_base if n == 0 else f"{safe_base}_{n}"

            print(f"[{i}/{total}] Running {safe} ...", flush=True)
            notes = "missing_or_invalid_clip_path" if not clip_path_str else "clip_file_not_found"
            summary_rows.append(
                {
                    "clip_path": clip_path_str,
                    "clip_stem": stem_raw,
                    "safe_clip_stem": safe,
                    "output_video": "",
                    "passes_jsonl": "",
                    "log_path": "",
                    "return_code": "-1",
                    "detected_pass_count": "-1",
                    "notes": notes,
                }
            )
            print(
                f"[{i}/{total}] Done return_code=-1 detected_pass_count=-1",
                flush=True,
            )
            continue

        stem_raw = stem_raw or vid_path.stem
        safe_base = sanitize_clip_stem(stem_raw)
        n = dirname_counts[safe_base]
        dirname_counts[safe_base] += 1
        safe = safe_base if n == 0 else f"{safe_base}_{n}"

        folder = out_root / safe
        folder.mkdir(parents=True, exist_ok=True)

        out_video = folder / f"{safe}_pass.mp4"
        passes_jsonl = folder / f"{safe}_passes.jsonl"
        log_path = folder / f"{safe}_pass.log"

        clip_path = str(vid_path.resolve())

        print(f"[{i}/{total}] Running {safe} ...", flush=True)

        if skip_if_done and out_video.is_file() and passes_jsonl.is_file() and log_path.is_file():
            det_count, _ = count_pass_events_jsonl(passes_jsonl)
            notes = "skipped_existing"
            summary_rows.append(
                {
                    "clip_path": clip_path,
                    "clip_stem": stem_raw,
                    "safe_clip_stem": safe,
                    "output_video": str(out_video),
                    "passes_jsonl": str(passes_jsonl),
                    "log_path": str(log_path),
                    "return_code": "0",
                    "detected_pass_count": str(det_count),
                    "notes": notes,
                }
            )
            print(
                f"[{i}/{total}] Done return_code=0 detected_pass_count={det_count}",
                flush=True,
            )
            continue

        cmd = [
            args.python,
            str(main_py),
            "--video",
            clip_path,
            "--output",
            str(out_video),
            "--mode",
            "normal",
            "--pass-detection",
            "--passes-jsonl",
            str(passes_jsonl),
            "--pass-debug",
            "--no-topdown",
        ]

        notes_parts: list[str] = []
        try:
            proc = subprocess.run(
                cmd,
                cwd=str(root),
                env=env_base,
                capture_output=True,
                text=True,
                check=False,
            )
            log_body = proc.stdout or ""
            if proc.stderr:
                log_body += ("\n" if log_body else "") + "--- stderr ---\n" + proc.stderr
            log_path.write_text(log_body, encoding="utf-8")
            rc = proc.returncode
            if rc != 0:
                notes_parts.append(f"nonzero_exit_{rc}")
        except OSError as e:
            rc = -1
            log_path.write_text(f"Failed to spawn subprocess: {e}", encoding="utf-8")
            notes_parts.append(f"os_error:{e}")

        det_count, count_note = count_pass_events_jsonl(passes_jsonl)
        if count_note:
            notes_parts.append(count_note)

        summary_rows.append(
            {
                "clip_path": clip_path,
                "clip_stem": stem_raw,
                "safe_clip_stem": safe,
                "output_video": str(out_video),
                "passes_jsonl": str(passes_jsonl),
                "log_path": str(log_path),
                "return_code": str(rc),
                "detected_pass_count": str(det_count),
                "notes": ";".join(notes_parts),
            }
        )

        print(
            f"[{i}/{total}] Done return_code={rc} detected_pass_count={det_count}",
            flush=True,
        )

    summary_csv = out_root / "batch_summary.csv"
    fields = [
        "clip_path",
        "clip_stem",
        "safe_clip_stem",
        "output_video",
        "passes_jsonl",
        "log_path",
        "return_code",
        "detected_pass_count",
        "notes",
    ]
    with summary_csv.open("w", newline="", encoding="utf-8") as fp:
        w = csv.DictWriter(fp, fieldnames=fields)
        w.writeheader()
        for sr in summary_rows:
            w.writerow(sr)

    print(
        f"Wrote {len(summary_rows)} row(s); summary: {summary_csv}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
