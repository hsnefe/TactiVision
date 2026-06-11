"""VisionEngine pas tespitini çalıştırıp event listesi döndürür.

İki mod:
  * Gerçek: ``src/main.py`` CLI'ı subprocess olarak çağrılır (--pass-detection),
    üretilen ``*_passes.jsonl`` okunur.
  * Mock: ağır YOLO bağımlılıkları yoksa ya da TACTIVISION_MOCK=1 ise örnek,
    gerçekçi event akışı üretilir (uçtan uca demo için).
"""

from __future__ import annotations

import json
import random
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from BackendAPI.Infrastructure.config import Settings


@dataclass
class VisionResult:
    """Pas tespiti çıktısı."""

    events: list[dict[str, Any]]
    used_mock: bool
    # YOLO'nun ürettiği kutucuklu/işlenmiş video (yalnızca gerçek modda dolu).
    annotated_video: Optional[Path] = None


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if line:
                events.append(json.loads(line))
    return events


def run_pass_detection(video_path: Path, out_dir: Path, settings: Settings) -> VisionResult:
    """Senin YOLO programını (src/main.py) çalıştırıp pas event'lerini döndürür."""

    if settings.vision_mock:
        return VisionResult(_mock_events(video_path), used_mock=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    annotated = out_dir / f"{video_path.stem}_analyzed.mp4"
    passes_jsonl = out_dir / f"{video_path.stem}_passes.jsonl"

    python_exe = settings.python_exe or sys.executable
    main_py = settings.root_dir / "src" / "main.py"

    cmd = [
        python_exe,
        str(main_py),
        "--video", str(video_path),
        "--output", str(annotated),
        "--model", settings.yolo_model,
        "--pass-detection",
        "--passes-jsonl", str(passes_jsonl),
    ]

    try:
        proc = subprocess.run(
            cmd,
            cwd=str(settings.root_dir / "src"),
            capture_output=True,
            text=True,
            timeout=60 * 60,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        # Bağımlılık/ağırlık yoksa demoyu bozma: mock'a düş.
        return VisionResult(_mock_events(video_path), used_mock=True)

    if proc.returncode != 0 or not passes_jsonl.exists():
        # Pipeline başarısız (ör. torch/ultralytics kurulu değil) -> mock.
        return VisionResult(_mock_events(video_path), used_mock=True)

    annotated_out = annotated if annotated.exists() else None
    return VisionResult(_read_jsonl(passes_jsonl), used_mock=False, annotated_video=annotated_out)


def _mock_events(video_path: Path) -> list[dict[str, Any]]:
    """Gerçekçi, çeşitli pas event'leri üret (deterministik seed = dosya adı)."""

    rng = random.Random(hash(video_path.name) & 0xFFFFFFFF)
    event_types = (
        ["completed_pass"] * 7 + ["intercepted_pass"] * 2 + ["unknown_pass"]
    )
    players_a = [4, 7, 8, 10, 11, 9]
    players_b = [3, 5, 6, 14, 17]

    events: list[dict[str, Any]] = []
    t = round(rng.uniform(2.0, 5.0), 2)
    for i in range(1, rng.randint(9, 15)):
        etype = rng.choice(event_types)
        from_team = rng.choice([0, 1])
        from_pool = players_a if from_team == 0 else players_b
        if etype == "intercepted_pass":
            to_team = 1 - from_team
            to_pool = players_b if to_team == 1 else players_a
        else:
            to_team = from_team
            to_pool = from_pool
        frm = rng.choice(from_pool)
        to = rng.choice([p for p in to_pool if p != frm])
        dur = round(rng.uniform(0.4, 1.6), 2)
        events.append(
            {
                "event_id": i,
                "event_type": etype,
                "start_frame": int(t * 25),
                "end_frame": int((t + dur) * 25),
                "start_time_sec": t,
                "end_time_sec": round(t + dur, 2),
                "from_player_id": frm,
                "to_player_id": None if etype == "unknown_pass" else to,
                "from_team_id": from_team,
                "to_team_id": None if etype == "unknown_pass" else to_team,
                "duration_sec": dur,
                "confidence": round(rng.uniform(0.55, 0.95), 2),
                "ball_start_xy": [round(rng.uniform(100, 1100), 1), round(rng.uniform(100, 600), 1)],
                "ball_end_xy": [round(rng.uniform(100, 1100), 1), round(rng.uniform(100, 600), 1)],
            }
        )
        t = round(t + dur + rng.uniform(1.5, 6.0), 2)
    return events
