#!/usr/bin/env python3
"""Inspect Hugging Face dataset dbal0503/Bundesliga (splits, schema, samples, local mirrors)."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


def _row_to_search_blob(row: dict[str, Any]) -> str:
    """Flatten primitive fields for substring search."""

    parts: list[str] = []

    def walk(val: Any) -> None:
        if isinstance(val, str):
            parts.append(val)
        elif isinstance(val, dict):
            for v in val.values():
                walk(v)
        elif isinstance(val, (list, tuple)):
            for v in val:
                walk(v)
        elif isinstance(val, (int, float, bool)) or val is None:
            parts.append(str(val))

    for v in row.values():
        walk(v)
    blob = "\n".join(parts).lower()
    return blob


def _mentions_event_keywords(blob: str) -> bool:
    keys = ("pass", "cross", "throwin", "throw-in", "throw_in")
    return any(k in blob for k in keys)


def _guess_video_columns(columns: list[str]) -> list[str]:
    """Columns likely naming a fixture / clip."""

    guesses: list[str] = []
    for c in columns:
        lc = c.lower()
        if any(
            frag in lc
            for frag in (
                "video_id",
                "videoid",
                "clip_id",
                "game_id",
                "match_id",
                "fixture_id",
                "video_file",
                "filename",
                "url",
                "video",
            )
        ):
            guesses.append(c)
    return guesses


def _safe_add(unique: set[Any], val: Any) -> None:
    try:
        unique.add(val)
    except TypeError:
        unique.add(json.dumps(val, sort_keys=True, ensure_ascii=False, default=str))


def _scan_split(
    split_ds: Iterable,
    *,
    columns: list[str],
    sample_rows: int,
    scan_cap: int,
    keyword_hit_cap: int = 40,
    max_distinct_per_column: int = 8_192,
) -> tuple[
    list[dict[str, Any]],
    dict[str, set[Any]],
    dict[str, set[Any]],
    list[tuple[int, dict[str, Any]]],
]:
    """One forward pass: streaming-safe."""

    samples: list[dict[str, Any]] = []
    evt_uniques: dict[str, set[Any]] = defaultdict(set)
    vid_uniques: dict[str, set[Any]] = defaultdict(set)
    kw_hits: list[tuple[int, dict[str, Any]]] = []

    evt_cols = _guess_event_column(columns or [])[:8]
    vid_cols = _guess_video_columns(columns or [])[:12]

    for idx, ex in enumerate(split_ds):
        if idx >= scan_cap:
            break
        row = dict(ex) if hasattr(ex, "keys") else ex  # type: ignore[arg-type]
        if not isinstance(row, dict):
            row = {"_value": row}

        if idx < sample_rows:
            samples.append(row)

        blob = _row_to_search_blob(row)
        if _mentions_event_keywords(blob) and len(kw_hits) < keyword_hit_cap:
            kw_hits.append((idx, row))

        for col in evt_cols:
            if col not in row:
                continue
            if len(evt_uniques[col]) >= max_distinct_per_column:
                continue
            _safe_add(evt_uniques[col], row[col])

        for col in vid_cols:
            if col not in row:
                continue
            if len(vid_uniques[col]) >= max_distinct_per_column:
                continue
            _safe_add(vid_uniques[col], row[col])

    return samples, dict(evt_uniques), dict(vid_uniques), kw_hits


def _guess_event_column(columns: list[str]) -> list[str]:
    guesses: list[str] = []
    for c in columns:
        cl = c.lower()
        if any(
            t in cl
            for t in ("event", "label", "action", "verb", "type", "segment", "class")
        ):
            guesses.append(c)
    return guesses or columns





def _inspect_local_dir(local_dir: Path, *, max_list: int = 50) -> list[str]:
    lines: list[str] = []
    if not local_dir.exists():
        lines.append(f"[local-dir] Missing: {local_dir}")
        return lines

    paths = sorted(local_dir.rglob("*"))
    files = [p for p in paths if p.is_file()]
    dirs = [p for p in paths if p.is_dir()]

    lines.append(f"[local-dir] Exists: {local_dir}")
    lines.append(f"[local-dir] File count (recursive): {len(files)}")
    lines.append(f"[local-dir] Dir count (recursive): {len(dirs)}")

    by_ext: Counter[str] = Counter()
    for p in files:
        ext = p.suffix.lower() if p.suffix else "<no_suffix>"
        by_ext[ext] += 1

    lines.append("[local-dir] Extensions (count, ext):")
    for ext, cnt in by_ext.most_common():
        lines.append(f"    {cnt:8d}  {ext}")

    lines.append(f"[local-dir] First {max_list} files:")
    for p in files[:max_list]:
        try:
            rel = p.relative_to(local_dir.resolve())
        except ValueError:
            rel = p
        try:
            size = p.stat().st_size
        except OSError:
            size = -1
        lines.append(f"    {rel}  ({size} bytes)")

    # Heuristic video presence
    video_ext = {".mp4", ".mkv", ".webm", ".avi", ".mov", ".ts", ".m4v"}
    n_vid = sum(1 for p in files if p.suffix.lower() in video_ext)
    lines.append(f"[local-dir] Video-ish files (*.mp4,.mkv,...): {n_vid}")

    return lines


def _load_dataset_smart(name: str, local_dir: Path | None) -> tuple[Any, str]:
    """Return (dataset_dict_or_single, mode_description)."""

    from datasets import load_dataset  # type: ignore[import-untyped]

    kwargs: dict[str, Any] = {}
    maybe_cache = Path(local_dir).resolve() if local_dir else None
    if maybe_cache and maybe_cache.is_dir():
        # Common HF cache layouts; harmless if unused.
        kwargs["cache_dir"] = str(maybe_cache)

    errs: list[str] = []

    def try_once(*, streaming: bool) -> Any:
        last_exc: Exception | None = None
        for tc in (False, True):
            try:
                return load_dataset(
                    name,
                    streaming=streaming,
                    trust_remote_code=tc,
                    **kwargs,
                )
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
        if last_exc is not None:
            raise last_exc
        raise RuntimeError("load_dataset failed without exception detail")

    try:
        ds = try_once(streaming=True)
        return ds, "streaming=True"
    except Exception as e:  # noqa: BLE001 broad for HF robustness
        errs.append(f"streaming=True failed: {type(e).__name__}: {e}")

    try:
        ds = try_once(streaming=False)
        return ds, "streaming=False (fallback)"
    except Exception as e:  # noqa: BLE001
        errs.append(f"streaming=False failed: {type(e).__name__}: {e}")
        joined = "\n".join(errs)
        raise RuntimeError(
            "Could not load dataset with datasets.load_dataset.\n" + joined
        ) from e


def _split_names(ds: Any) -> list[str]:
    if hasattr(ds, "keys"):
        return list(ds.keys())
    return ["data"]


def _get_split(ds: Any, split: str):
    try:
        return ds[split]
    except Exception:  # noqa: BLE001
        return ds


def _column_names(ds_split: Any) -> list[str]:
    if hasattr(ds_split, "column_names"):
        return list(ds_split.column_names)
    if hasattr(ds_split, "features") and hasattr(ds_split.features, "keys"):
        return list(ds_split.features.keys())
    return []


def main() -> int:
    p = argparse.ArgumentParser(
        description="Inspect HF dataset dbal0503/Bundesliga-like configs."
    )
    p.add_argument(
        "--dataset-name",
        type=str,
        default="dbal0503/Bundesliga",
        help="HF dataset hub id.",
    )
    p.add_argument(
        "--local-dir",
        type=Path,
        default=Path("data/hf_bundesliga"),
        help="Local cache/mirror folder to inspect recursively.",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path("data/hf_bundesliga_schema.txt"),
        help="Written report.",
    )
    p.add_argument(
        "--sample-rows",
        type=int,
        default=30,
        help="Examples per split to materialize.",
    )
    args = p.parse_args()

    try:
        ds, mode = _load_dataset_smart(args.dataset_name, args.local_dir)
    except ImportError:
        print(
            "Missing dependency `datasets`. Install with: pip install datasets",
            file=sys.stderr,
        )
        return 1
    except Exception as exc:  # noqa: BLE001
        print(f"Failed to load dataset: {exc}", file=sys.stderr)
        return 1

    lines: list[str] = []
    lines.append("=" * 72)
    lines.append(f"Hugging Face dataset inspect: {args.dataset_name}")
    lines.append(f"load_mode: {mode}")
    lines.append(f"resolved local-dir arg: {args.local_dir}")
    lines.append("=" * 72)

    splits = _split_names(ds)
    lines.append(f"splits ({len(splits)}): {splits}")

    scan_cap = max(args.sample_rows * 2000, 10_000)

    for split in splits:
        lines.append("")
        lines.append(f"--- split: {split!r} ---")
        split_ds = _get_split(ds, split)

        cols = _column_names(split_ds)
        lines.append(f"columns ({len(cols)}): {cols}")

        guessed_evt = _guess_event_column(cols or [])
        guessed_vid = _guess_video_columns(cols or [])
        lines.append(f"guess event-like columns for uniques scan: {guessed_evt}")
        lines.append(f"guess video-id columns for uniques scan: {guessed_vid}")

        sampled, evt_uniques, vid_uniques, kw_hits = _scan_split(
            split_ds,
            columns=cols or [],
            sample_rows=args.sample_rows,
            scan_cap=scan_cap,
        )

        lines.append(f"[scan_cap={scan_cap}] materialized_first={len(sampled)} samples")
        lines.append(f"first {len(sampled)} sample rows:")
        for i, row in enumerate(sampled):
            try:
                pretty = json.dumps(row, ensure_ascii=False, default=str, indent=2)
            except Exception:  # noqa: BLE001
                pretty = str(row)
            lines.append(f"[row {i}]")
            if len(pretty) > 8000:
                pretty = pretty[:8000] + "\n… (truncated) …"
            lines.append(pretty)

        lines.append("")
        lines.append("(unique-ish event columns; bounded distinct set per scan window)")
        for col in guessed_evt[:5]:
            vals = evt_uniques.get(col, set())
            if not vals:
                continue
            lines.append(f"    {col!r}: {len(vals)} distinct scanned")
            preview = sorted(
                vals,
                key=lambda x: (str(type(x).__name__), str(x)),
            )[:120]
            for v in preview:
                s = repr(v)
                if len(s) > 200:
                    s = s[:200] + "…"
                lines.append(f"      - {s}")

        lines.append("")
        lines.append("(unique-ish video-id / clip columns)")
        if not guessed_vid:
            lines.append("    (no columns matched video-id heuristics)")
        else:
            for col in guessed_vid:
                vals = vid_uniques.get(col, set())
                if not vals:
                    lines.append(f"    {col!r}: (no populated values seen in scan window)")
                    continue
                lines.append(f"    {col!r}: {len(vals)} distinct in scan window")
                preview = sorted(
                    vals,
                    key=lambda x: (str(type(x).__name__), str(x)),
                )[:80]
                for v in preview:
                    s = repr(v)
                    if len(s) > 200:
                        s = s[:200] + "…"
                    lines.append(f"      - {s}")

        lines.append("")
        lines.append(
            f"examples where row blob matches pass/cross/throw-in keywords "
            f"(first up to {len(kw_hits)} hits inside scan_cap):"
        )
        for gi, (row_idx, row) in enumerate(kw_hits):
            lines.append(f"  keyword_hit[{gi}] global_row_idx={row_idx}")
            try:
                snippet = json.dumps(row, ensure_ascii=False, default=str)
            except Exception:  # noqa: BLE001
                snippet = str(row)
            if len(snippet) > 6000:
                snippet = snippet[:6000] + "\n… (truncated) …"
            lines.append(snippet)
            lines.append("")

    lines.append("")
    lines.extend(_inspect_local_dir(args.local_dir.expanduser(), max_list=50))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = "\n".join(lines) + "\n"
    args.output.write_text(report, encoding="utf-8")
    print(report, end="")
    print(f"(wrote report to {args.output})", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
