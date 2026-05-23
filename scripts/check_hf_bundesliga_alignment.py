#!/usr/bin/env python3
"""Verify Hugging Face Bundesliga annotation CSV rows vs MP4 paths in the HF dataset repo."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path


def _normalize_csv_field_name(name: str) -> str:
    return name.lstrip("\ufeff").strip()


def read_csv_records(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    """Parse CSV using utf-8-sig; normalize header BOM and BOM-prefixed field names."""

    text = path.read_text(encoding="utf-8-sig")
    reader = csv.DictReader(text.splitlines())
    raw_headers = reader.fieldnames or []

    aliases: dict[str, str] = {
        _normalize_csv_field_name(h): h for h in raw_headers
    }

    def row_norm(row: dict[str, str]) -> dict[str, str]:
        out: dict[str, str] = {}
        for k, v in row.items():
            if k is None:
                continue
            nk = _normalize_csv_field_name(k)
            out[nk] = v if isinstance(v, str) else ""  # type: ignore[unreachable]
        return out

    headers = [_normalize_csv_field_name(h) for h in raw_headers if h is not None]
    records = [row_norm(r) for r in reader if any((v or "").strip() for v in r.values())]
    return headers, records


def _build_id_variants(raw: str) -> set[str]:
    """Candidate strings for ID matching (annotation video_id or MP4 stem)."""

    s = (raw or "").strip()
    if not s:
        return set()

    out: set[str] = set()
    sl = s.lower()
    no_us = s.replace("_", "")
    no_us_l = sl.replace("_", "")

    def add(x: str) -> None:
        if x:
            out.add(x)
            if len(x) >= 6:
                out.add(x[:6])
                out.add(x[-6:])

    add(s)
    add(sl)
    add(no_us)
    add(no_us_l)

    if "_" in s:
        add(s.split("_", 1)[0])
        add(s.rsplit("_", 1)[0])
        add(sl.split("_", 1)[0])
        add(sl.rsplit("_", 1)[0])
    return out


def _row_text_blob(row: dict[str, str]) -> str:
    return "\n".join(str(v) for v in row.values()).lower()


def _event_attributes_text(row: dict[str, str]) -> str:
    """Best-effort text for event_attributes column (raw or JSON string)."""

    for key in ("event_attributes", "eventattribute", "attributes", "attributes_json"):
        if key in row and row[key]:
            val = row[key].strip()
            try:
                obj = json.loads(val)
                return json.dumps(obj, ensure_ascii=False).lower()
            except json.JSONDecodeError:
                return val.lower()
    return ""


def _event_field(row: dict[str, str]) -> str:
    for key in ("event", "label", "action", "event_type", "type"):
        if key in row and row[key]:
            return row[key].strip().lower()
    return ""


def is_pass_related_row(row: dict[str, str]) -> bool:
    eat = _event_attributes_text(row)
    ev = _event_field(row)
    throwin_labels = {"throwin", "throw-in", "throw_in"}
    if "pass" in eat:
        return True
    if "cross" in eat:
        return True
    if ev in throwin_labels and "pass" in eat:
        return True
    return False


def _pick_time_numeric(row: dict[str, str]) -> list[float]:
    """Collect numeric time-like values from row for min/max range."""

    nums: list[float] = []
    for k, v in row.items():
        kl = k.lower()
        if not any(
            frag in kl
            for frag in (
                "time",
                "sec",
                "second",
                "minute",
                "frame",
                "timestamp",
                "start",
                "end",
                "offset",
            )
        ):
            continue
        if not v or not str(v).strip():
            continue
        m = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", str(v))
        for g in m:
            try:
                nums.append(float(g))
            except ValueError:
                continue
    return nums


@dataclass
class VideoAgg:
    events: int = 0
    pass_related: int = 0
    time_values: list[float] = field(default_factory=list)


def resolve_video_id_key(rows: list[dict[str, str]]) -> str:
    if not rows:
        return "video_id"
    sample = rows[0]
    if "video_id" in sample and any((r.get("video_id") or "").strip() for r in rows[:50]):
        return "video_id"
    for k in sample.keys():
        lk = k.lower()
        if "video" in lk and "id" in lk:
            return k
    return "video_id"


def aggregate_by_video(records: list[dict[str, str]]) -> tuple[dict[str, VideoAgg], str]:
    vid_key = resolve_video_id_key(records)
    agg: dict[str, VideoAgg] = defaultdict(VideoAgg)
    for row in records:
        vid = (row.get(vid_key) or "").strip()
        if not vid:
            continue
        g = agg[vid]
        g.events += 1
        if is_pass_related_row(row):
            g.pass_related += 1
        g.time_values.extend(_pick_time_numeric(row))
    return dict(agg), vid_key


@dataclass
class Mp4Entry:
    rel_path: str
    stem: str
    variants: set[str]


def list_repo_mp4s(repo_id: str) -> tuple[list[str], list[str]]:
    from huggingface_hub import HfApi  # type: ignore[import-untyped]

    api = HfApi()
    files = api.list_repo_files(repo_id, repo_type="dataset")
    mp4 = [f for f in files if f.lower().endswith(".mp4")]
    norm = [f.replace("\\", "/") for f in mp4]
    under_clips = [f for f in norm if "bundesliga/clips/" in f.lower()]
    return norm, under_clips


def build_mp4_entries(paths: list[str]) -> list[Mp4Entry]:
    out: list[Mp4Entry] = []
    for p in paths:
        stem = Path(p).stem
        out.append(Mp4Entry(rel_path=p, stem=stem, variants=_build_id_variants(stem)))
    return out


def match_video_id(
    video_id: str,
    entries: list[Mp4Entry],
) -> tuple[list[str], str, str]:
    """
    Return (matched_paths, match_method, notes).

    Tries, in order: exact stem, normalized set intersection, substring, underscore-prefix.
    """

    vid_stem_exact = video_id.strip()
    vid_vars = _build_id_variants(vid_stem_exact)
    notes: list[str] = []

    # 1) exact stem
    exact = [e.rel_path for e in entries if e.stem == vid_stem_exact]
    if exact:
        return exact, "exact_stem", ""

    # 2) normalized – any shared variant (exact token equality in variant sets)
    norm_hits: list[str] = []
    for e in entries:
        if vid_vars & e.variants:
            norm_hits.append(e.rel_path)
    if norm_hits:
        return norm_hits, "normalized_intersection", ""

    # 3) substring (avoid trivial 1–2 char noise)
    sub_hits: list[str] = []
    seen: set[str] = set()
    vlow = vid_stem_exact.lower()
    for e in entries:
        el = e.stem.lower()
        hit = False
        if len(vlow) >= 3 and (vlow in el or el in vlow):
            hit = True
        elif len(el) >= 3 and (el in vlow or vlow in el):
            hit = True
        if hit and e.rel_path not in seen:
            seen.add(e.rel_path)
            sub_hits.append(e.rel_path)
    if sub_hits:
        return sub_hits, "substring", "substring match can be ambiguous"

    # 4) prefix before first underscore
    if "_" in vid_stem_exact:
        pre = vid_stem_exact.split("_", 1)[0].lower()
        pre_hits = [
            e.rel_path
            for e in entries
            if e.stem.lower().split("_", 1)[0] == pre and len(pre) >= 3
        ]
        if pre_hits:
            return pre_hits, "underscore_prefix", "matched on token before first '_'"

    notes.append("no_mp4_match")
    return [], "none", "; ".join(notes) if notes else "no match"


def alignment_verdict(matched: int, total: int) -> str:
    if total <= 0:
        return "no"
    if matched == total:
        return "yes"
    if matched == 0:
        return "no"
    return "partial"


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Check train.csv video_id rows vs MP4 paths in Hugging Face dataset repo."
    )
    ap.add_argument("--hf-dir", type=Path, default=Path("data/hf_bundesliga"))
    ap.add_argument("--repo-id", type=str, default="dbal0503/Bundesliga")
    ap.add_argument(
        "--output-report",
        type=Path,
        default=Path("data/hf_bundesliga_alignment_report.txt"),
    )
    ap.add_argument(
        "--output-csv",
        type=Path,
        default=Path("data/hf_bundesliga_alignment.csv"),
    )
    args = ap.parse_args()

    train_path = args.hf_dir / "train.csv"
    clips_path = args.hf_dir / "clips.csv"

    if not train_path.is_file():
        print(f"ERROR: missing {train_path}", file=sys.stderr)
        return 1

    try:
        import huggingface_hub  # noqa: F401
    except ImportError:
        print(
            "Missing dependency `huggingface_hub`. Install: pip install huggingface_hub",
            file=sys.stderr,
        )
        return 1

    _, train_rows = read_csv_records(train_path)
    clips_note = ""
    if clips_path.is_file():
        _, clip_rows = read_csv_records(clips_path)
        if clip_rows:
            cvk = resolve_video_id_key(clip_rows)
            uclip = len(
                {
                    (r.get(cvk) or "").strip()
                    for r in clip_rows
                    if (r.get(cvk) or "").strip()
                }
            )
            clips_note = (
                f"clips.csv: {len(clip_rows)} rows, {uclip} distinct {cvk!r}.\n"
            )
        else:
            clips_note = "clips.csv present but empty.\n"
    else:
        clips_note = "clips.csv not found (optional).\n"

    by_video, video_id_column = aggregate_by_video(train_rows)
    unique_vids = sorted(by_video.keys())

    all_mp4, clips_mp4 = list_repo_mp4s(args.repo_id)
    use_clips_primary = len(clips_mp4) > 0
    mp4_entries_clips = build_mp4_entries(clips_mp4) if use_clips_primary else []
    mp4_entries_all = build_mp4_entries(all_mp4)
    primary_entries = mp4_entries_clips if use_clips_primary else mp4_entries_all

    report_lines: list[str] = []
    report_lines.append("=" * 72)
    report_lines.append("Bundesliga HF annotations vs repository MP4 alignment check")
    report_lines.append(f"hf-dir: {args.hf_dir}")
    report_lines.append(f"repo-id: {args.repo_id}")
    report_lines.append(f"train.csv rows: {len(train_rows)}")
    report_lines.append(clips_note.rstrip())
    report_lines.append(
        f"unique annotation video_id: {len(unique_vids)} "
        f"(column used: '{video_id_column}', BOM-safe headers)"
    )
    report_lines.append(f"MP4 files in repo (all): {len(all_mp4)}")
    report_lines.append(f"MP4 under Bundesliga/Clips/: {len(clips_mp4)}")
    if len(all_mp4) > 1:
        report_lines.append("sample MP4 paths (first 25):")
        for pth in sorted(all_mp4)[:25]:
            report_lines.append(f"    {pth}")
    elif len(all_mp4) == 1:
        report_lines.append(f"single MP4: {all_mp4[0]}")

    csv_rows_out: list[dict[str, str]] = []
    matched_count = 0
    unmatched: list[str] = []

    for vid in unique_vids:
        agg = by_video[vid]
        times = sorted(agg.time_values)
        tmin = f"{times[0]:.6g}" if times else ""
        tmax = f"{times[-1]:.6g}" if times else ""

        paths_trial, method, note = match_video_id(vid, primary_entries)
        if (
            not paths_trial
            and use_clips_primary
            and primary_entries is not mp4_entries_all
        ):
            paths_fb, method_fb, note_fb = match_video_id(vid, mp4_entries_all)
            if paths_fb:
                paths_trial = paths_fb
                method = f"{method_fb}_fallback_all_repo_mp4"
                note = note_fb or note

        if paths_trial:
            matched_count += 1
        else:
            unmatched.append(vid)

        csv_rows_out.append(
            {
                "video_id": vid,
                "annotation_event_count": str(agg.events),
                "pass_related_count": str(agg.pass_related),
                "time_min": tmin,
                "time_max": tmax,
                "matched_mp4_count": str(len(paths_trial)),
                "matched_mp4_paths": ";".join(paths_trial[:50]),
                "match_method": method,
                "notes": note,
            }
        )

    total = len(unique_vids)
    verdict = alignment_verdict(matched_count, total)
    pct = (100.0 * matched_count / total) if total else 0.0

    report_lines.append("")
    report_lines.append(f"matched video_id: {matched_count} / {total} ({pct:.1f}%)")
    report_lines.append(f"unmatched video_id: {len(unmatched)}")
    if unmatched:
        report_lines.append("unmatched IDs (truncated at 120):")
        for u in unmatched[:120]:
            report_lines.append(f"    {u}")
        if len(unmatched) > 120:
            report_lines.append(f"    … {len(unmatched) - 120} more")

    report_lines.append("")
    report_lines.append(f"ALIGNED: {verdict}")

    warning = False
    if matched_count == 0 or matched_count < max(3, int(0.05 * total)):
        warning = True
    if total > 0 and pct < 5.0:
        warning = True

    warn_msg = (
        "WARNING: annotations and clips may not be aligned; "
        "do not run timing evaluation yet."
    )
    if warning:
        report_lines.append("")
        report_lines.append(warn_msg)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "video_id",
        "annotation_event_count",
        "pass_related_count",
        "time_min",
        "time_max",
        "matched_mp4_count",
        "matched_mp4_paths",
        "match_method",
        "notes",
    ]
    with args.output_csv.open("w", newline="", encoding="utf-8") as fp:
        wr = csv.DictWriter(fp, fieldnames=fieldnames)
        wr.writeheader()
        for row in csv_rows_out:
            wr.writerow(row)

    report_text = "\n".join(report_lines) + "\n"
    args.output_report.write_text(report_text, encoding="utf-8")

    print(report_text, end="")
    print(f"[wrote] {args.output_report}", file=sys.stderr)
    print(f"[wrote] {args.output_csv}", file=sys.stderr)
    if warning:
        print(warn_msg, file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
