"""Bir işin uçtan uca akışı: video -> pas tespiti -> spiker metni -> TTS sesi.

Arka planda (BackgroundTasks) çalışır ve job durumunu adım adım günceller.
"""

from __future__ import annotations

import traceback

from BackendAPI.Infrastructure.config import Settings
from BackendAPI.Infrastructure.job_store import JobStore
from BackendAPI.Models.schemas import JobStage
from BackendAPI.Services.vision_service import run_pass_detection
from BackendAPI.Services.video_service import compose_narrated_video, probe_duration
from CommentaryAI.llm_client import generate_commentary
from CommentaryAI.synthesis import voice as tts


def process_job(job_id: str, store: JobStore, settings: Settings) -> None:
    job = store.get(job_id)
    if job is None:
        return

    try:
        # 1) VisionEngine: pas tespiti
        store.update(job_id, stage=JobStage.PROCESSING, progress=10,
                     message="Video analiz ediliyor (pas tespiti)...")
        vres = run_pass_detection(job.video_path, settings.outputs_dir, settings)
        events = vres.events
        store.update(job_id, events=events, mock=vres.used_mock, progress=55,
                     message=f"{len(events)} pas olayı bulundu.")

        # 2) CommentaryAI: spiker metni
        store.update(job_id, stage=JobStage.COMMENTARY, progress=65,
                     message="AI spiker anlatımı yazılıyor...")
        # Anlatım videoya sığsın diye video süresini ölçüp LLM'e bütçe olarak ver.
        video_seconds = probe_duration(job.video_path)
        result = generate_commentary(
            events,
            settings.effective_provider,
            gemini_api_key=settings.gemini_api_key,
            gemini_model=settings.gemini_model,
            target_seconds=video_seconds,
        )
        store.update(job_id, commentary_text=result.text,
                     commentary_provider=result.provider, progress=80,
                     message=f"Anlatım hazır ({result.provider}).")

        # 3) TTS: sese çevir
        store.update(job_id, stage=JobStage.SYNTHESIZING, progress=85,
                     message="Spiker sesi üretiliyor (TTS)...")
        audio_path = settings.outputs_dir / f"{job_id}.mp3"
        ok = tts.synthesize(result.text, audio_path, voice=settings.tts_voice, rate=settings.tts_rate)
        if ok:
            store.update(job_id, audio_path=audio_path)

        # 4) Spiker sesini videonun üzerine göm (mp4).
        video_ok = False
        if ok:
            store.update(job_id, progress=92, message="Spiker sesi videoya gömülüyor...")
            # Ses HAM (kutucuksuz) yüklenen videonun üzerine eklenir.
            # YOLO'nun işlediği video yalnızca pas event'lerini üretmek için kullanılır.
            narrated = settings.outputs_dir / f"{job_id}_narrated.mp4"
            video_ok = compose_narrated_video(job.video_path, audio_path, narrated)
            if video_ok:
                store.update(job_id, narrated_video_path=narrated)

        if video_ok:
            final_msg = "Tamamlandı — spikerli video hazır."
        elif ok:
            final_msg = "Tamamlandı (ses hazır, videoya gömülemedi)."
        else:
            final_msg = "Tamamlandı (ses üretilemedi, metin hazır)."
        store.update(job_id, stage=JobStage.DONE, progress=100, message=final_msg)

    except Exception as exc:  # işin tamamı çökmesin, hatayı job'a yaz
        store.update(job_id, stage=JobStage.FAILED, error=f"{exc}",
                     message="İşlem sırasında hata oluştu.")
        traceback.print_exc()
