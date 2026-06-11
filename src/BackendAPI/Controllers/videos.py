"""Video yükleme, iş durumu, event'ler, anlatım ve ses uçları."""

from __future__ import annotations

import shutil
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, UploadFile, File
from fastapi.responses import FileResponse

from BackendAPI.Infrastructure.config import Settings, get_settings
from BackendAPI.Infrastructure.job_store import Job, JobStore, get_job_store
from BackendAPI.Models.schemas import (
    CommentaryResponse,
    EventsResponse,
    JobResponse,
    PassEventModel,
)
from BackendAPI.Services.pipeline_service import process_job

router = APIRouter(prefix="/api", tags=["videos"])

ALLOWED_EXT = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


def _to_response(job: Job) -> JobResponse:
    return JobResponse(
        job_id=job.job_id,
        filename=job.filename,
        stage=job.stage,
        progress=job.progress,
        message=job.message,
        mock=job.mock,
        event_count=len(job.events),
        has_video=bool(job.narrated_video_path and job.narrated_video_path.exists()),
        error=job.error,
    )


@router.post("/videos", response_model=JobResponse)
async def upload_video(
    background: BackgroundTasks,
    file: UploadFile = File(...),
    store: JobStore = Depends(get_job_store),
    settings: Settings = Depends(get_settings),
) -> JobResponse:
    ext = Path(file.filename or "").suffix.lower()
    if ext not in ALLOWED_EXT:
        raise HTTPException(400, f"Desteklenmeyen dosya türü: {ext or '?'}. İzin verilen: {sorted(ALLOWED_EXT)}")

    job = store.create(filename=file.filename or "video", video_path=Path())
    dest = settings.uploads_dir / f"{job.job_id}{ext}"
    with dest.open("wb") as out:
        shutil.copyfileobj(file.file, out)
    store.update(job.job_id, video_path=dest)

    # Uçtan uca işlemi arka planda başlat.
    background.add_task(process_job, job.job_id, store, settings)
    return _to_response(store.get(job.job_id))


@router.get("/jobs/{job_id}", response_model=JobResponse)
async def get_job(job_id: str, store: JobStore = Depends(get_job_store)) -> JobResponse:
    job = store.get(job_id)
    if job is None:
        raise HTTPException(404, "İş bulunamadı.")
    return _to_response(job)


@router.get("/jobs/{job_id}/events", response_model=EventsResponse)
async def get_events(job_id: str, store: JobStore = Depends(get_job_store)) -> EventsResponse:
    job = store.get(job_id)
    if job is None:
        raise HTTPException(404, "İş bulunamadı.")
    events = [PassEventModel(**{k: ev.get(k) for k in PassEventModel.model_fields}) for ev in job.events]
    return EventsResponse(job_id=job_id, events=events)


@router.get("/jobs/{job_id}/commentary", response_model=CommentaryResponse)
async def get_commentary(job_id: str, store: JobStore = Depends(get_job_store)) -> CommentaryResponse:
    job = store.get(job_id)
    if job is None:
        raise HTTPException(404, "İş bulunamadı.")
    if not job.commentary_text:
        raise HTTPException(409, "Anlatım henüz hazır değil.")
    audio_url = f"/api/jobs/{job_id}/audio" if job.audio_path else None
    has_video = bool(job.narrated_video_path and job.narrated_video_path.exists())
    video_url = f"/api/jobs/{job_id}/video" if has_video else None
    return CommentaryResponse(
        job_id=job_id,
        provider=job.commentary_provider or "template",
        text=job.commentary_text,
        audio_url=audio_url,
        video_url=video_url,
    )


@router.get("/jobs/{job_id}/audio")
async def get_audio(job_id: str, store: JobStore = Depends(get_job_store)) -> FileResponse:
    job = store.get(job_id)
    if job is None or not job.audio_path or not job.audio_path.exists():
        raise HTTPException(404, "Ses bulunamadı.")
    return FileResponse(str(job.audio_path), media_type="audio/mpeg", filename=f"{job_id}.mp3")


@router.get("/jobs/{job_id}/video")
async def get_video(job_id: str, store: JobStore = Depends(get_job_store)) -> FileResponse:
    """Spiker sesi gömülmüş videoyu döndürür (tarayıcıda oynatılır/indirilir)."""
    job = store.get(job_id)
    if job is None or not job.narrated_video_path or not job.narrated_video_path.exists():
        raise HTTPException(404, "Video bulunamadı.")
    # FileResponse Range isteklerini destekler -> tarayıcıda akıcı oynatma.
    return FileResponse(str(job.narrated_video_path), media_type="video/mp4",
                        filename=f"{job_id}_spikerli.mp4")
