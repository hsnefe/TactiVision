"""TactiVision Backend API giriş noktası.

Çalıştırma (src/ dizininden):
    uvicorn BackendAPI.app:app --reload --port 8000
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from BackendAPI.Controllers.videos import router as videos_router
from BackendAPI.Infrastructure.config import get_settings

app = FastAPI(
    title="TactiVision API",
    description="Video -> pas tespiti -> AI spiker anlatımı -> TTS sesi",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # geliştirme; üretimde frontend origin'i ile sınırla
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(videos_router)


@app.get("/api/health")
async def health() -> dict:
    s = get_settings()
    return {
        "status": "ok",
        "vision_mock": s.vision_mock,
        "llm_provider": s.effective_provider,
        "tts_voice": s.tts_voice,
    }


# Derlenmiş frontend (src/WebFrontEnd/dist) varsa onu da bu sunucudan servis et.
# Böylece tek adres yeterli olur: http://localhost:8000
# (API yolları yukarıda tanımlı olduğundan onlara dokunmaz.)
_DIST = Path(__file__).resolve().parents[1] / "WebFrontEnd" / "dist"
if _DIST.is_dir():
    app.mount("/", StaticFiles(directory=str(_DIST), html=True), name="web")
