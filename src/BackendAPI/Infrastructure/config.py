"""Çalışma zamanı ayarları (ortam değişkenlerinden okunur)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _root() -> Path:
    # src/BackendAPI/Infrastructure/config.py -> proje kökü
    return Path(__file__).resolve().parents[3]


def _bool(name: str, default: bool) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class Settings:
    """Tüm backend ayarları tek yerde."""

    # Dizinler
    root_dir: Path = field(default_factory=_root)
    data_dir: Path = field(default_factory=lambda: _root() / "data")

    # VisionEngine
    # MOCK açıkken ağır YOLO pipeline yerine örnek event üretilir (hızlı demo / CI).
    vision_mock: bool = field(default_factory=lambda: _bool("TACTIVISION_MOCK", True))
    allow_mock_fallback: bool = field(default_factory=lambda: _bool("TACTIVISION_ALLOW_MOCK_FALLBACK", False))
    yolo_model: str = field(default_factory=lambda: os.getenv("TACTIVISION_YOLO_MODEL", "yolo11n.pt"))
    python_exe: str = field(default_factory=lambda: os.getenv("TACTIVISION_PYTHON", ""))

    # LLM (AI spiker)
    # Sağlayıcı: "gemini" | "template"
    llm_provider: str = field(default_factory=lambda: os.getenv("TACTIVISION_LLM", "gemini").lower())
    gemini_api_key: str = field(default_factory=lambda: os.getenv("GEMINI_API_KEY", ""))
    gemini_model: str = field(default_factory=lambda: os.getenv("GEMINI_MODEL", "gemini-2.5-flash"))

    # TTS (Edge TTS Türkçe spiker sesi)
    tts_voice: str = field(default_factory=lambda: os.getenv("TACTIVISION_TTS_VOICE", "tr-TR-AhmetNeural"))
    tts_rate: str = field(default_factory=lambda: os.getenv("TACTIVISION_TTS_RATE", "+12%"))

    def __post_init__(self) -> None:
        self.uploads_dir.mkdir(parents=True, exist_ok=True)
        self.outputs_dir.mkdir(parents=True, exist_ok=True)

    @property
    def uploads_dir(self) -> Path:
        return self.data_dir / "uploads"

    @property
    def outputs_dir(self) -> Path:
        return self.data_dir / "outputs"

    @property
    def effective_provider(self) -> str:
        """Anahtar yoksa otomatik olarak şablona düş."""
        if self.llm_provider == "gemini" and not self.gemini_api_key:
            return "template"
        return self.llm_provider


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
