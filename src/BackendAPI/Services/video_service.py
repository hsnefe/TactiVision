"""Spiker sesini (mp3) yüklenen videonun üzerine gömer (mp4 üretir).

Sistemde ffmpeg gerekmez: ``imageio-ffmpeg`` ile gelen taşınabilir ffmpeg
binary'si kullanılır. Paket yoksa adım atlanır (ses ayrı dosya olarak kalır).

Senkron: spiker sesi videodan uzunsa, ses ``atempo`` ile videoya TAM oturacak
şekilde hızlandırılır. Böylece video donmaz, anlatım yarıda kesilmez.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import Optional


def _ffmpeg_exe() -> Optional[str]:
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


_DUR_RE = re.compile(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)")


def probe_duration(path: Path) -> Optional[float]:
    """Bir medya dosyasının süresini saniye olarak döndürür (ffmpeg stderr'inden)."""
    ffmpeg = _ffmpeg_exe()
    if ffmpeg is None or not Path(path).exists():
        return None
    try:
        proc = subprocess.run([ffmpeg, "-i", str(path)], capture_output=True, text=True, timeout=60)
    except Exception:
        return None
    m = _DUR_RE.search(proc.stderr or "")
    if not m:
        return None
    h, mn, s = m.groups()
    return int(h) * 3600 + int(mn) * 60 + float(s)


def _atempo_chain(factor: float) -> Optional[str]:
    """factor (>=1) için atempo filtre zinciri. atempo tek başına 0.5–2.0 aralığını
    destekler; daha büyük hızlanma için zincirlenir."""
    if factor <= 1.001:
        return None
    parts: list[str] = []
    f = factor
    while f > 2.0:
        parts.append("atempo=2.0")
        f /= 2.0
    parts.append(f"atempo={f:.4f}")
    return ",".join(parts)


def compose_narrated_video(video_path: Path, audio_path: Path, out_path: Path) -> bool:
    """Videoyu spiker sesiyle birleştir. Ses videoya sığacak şekilde senkronlanır.

    Önce orijinal saha sesi (kısık) + spiker sesini KARIŞTIRMAYI dener; video
    sesi yoksa anlatımı tek ses kanalı yapar.
    """

    ffmpeg = _ffmpeg_exe()
    if ffmpeg is None or not video_path.exists() or not audio_path.exists():
        return False
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Senkron: ses videodan uzunsa, videoya tam oturacak tempo hesapla.
    v_dur = probe_duration(video_path)
    a_dur = probe_duration(audio_path)
    tempo = None
    if v_dur and a_dur and v_dur > 0:
        # %2 pay bırak ki ses videodan çok az kısa bitsin (taşmasın).
        factor = a_dur / (v_dur * 0.98)
        tempo = _atempo_chain(factor)

    spk_pre = f"{tempo}," if tempo else ""

    # 1) Karıştır: orijinal saha sesi (kısık) + spiker anlatımı (baskın).
    mix_cmd = [
        ffmpeg, "-y",
        "-i", str(video_path),
        "-i", str(audio_path),
        "-filter_complex",
        f"[0:a]volume=0.35[bg];[1:a]{spk_pre}volume=1.6[spk];"
        "[bg][spk]amix=inputs=2:duration=first:dropout_transition=2[aout]",
        "-map", "0:v:0", "-map", "[aout]",
        "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
        str(out_path),
    ]
    # 2) Yedek: video sesi yoksa anlatımı (gerekirse hızlandırılmış) ses kanalı yap.
    replace_filter = f"[1:a]{tempo}[aout]" if tempo else None
    if replace_filter:
        replace_cmd = [
            ffmpeg, "-y",
            "-i", str(video_path), "-i", str(audio_path),
            "-filter_complex", replace_filter,
            "-map", "0:v:0", "-map", "[aout]",
            "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
            str(out_path),
        ]
    else:
        replace_cmd = [
            ffmpeg, "-y",
            "-i", str(video_path), "-i", str(audio_path),
            "-map", "0:v:0", "-map", "1:a:0",
            "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
            str(out_path),
        ]

    for cmd in (mix_cmd, replace_cmd):
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        except Exception:
            continue
        if proc.returncode == 0 and out_path.exists() and out_path.stat().st_size > 0:
            return True
    return False
