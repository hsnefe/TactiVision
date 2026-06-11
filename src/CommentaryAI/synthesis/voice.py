"""Metni TTS ile insan sesine (mp3) çevirir.

Edge TTS (Microsoft Neural sesleri) kullanılır — ücretsiz, doğal Türkçe ses,
ffmpeg gerektirmez. Paket yoksa ses adımı atlanır (metin yine de döner).
"""

from __future__ import annotations

import asyncio
from pathlib import Path


def is_available() -> bool:
    try:
        import edge_tts  # noqa: F401
        return True
    except Exception:
        return False


def synthesize(text: str, out_path: Path, *, voice: str, rate: str = "+0%") -> bool:
    """Metni seslendirip ``out_path`` (mp3) olarak kaydet. Başarı durumunu döndür."""

    if not text.strip() or not is_available():
        return False

    import edge_tts

    async def _run() -> None:
        communicate = edge_tts.Communicate(text, voice=voice, rate=rate)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        await communicate.save(str(out_path))

    try:
        asyncio.run(_run())
    except Exception:
        return False
    return out_path.exists() and out_path.stat().st_size > 0
