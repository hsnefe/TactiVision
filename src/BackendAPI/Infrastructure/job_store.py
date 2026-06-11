"""Basit, thread-safe in-memory iş (job) deposu.

Üretimde bunun yerine Redis/DB konabilir; arayüz aynı kalır.
"""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from BackendAPI.Models.schemas import JobStage


@dataclass
class Job:
    job_id: str
    filename: str
    video_path: Path
    stage: JobStage = JobStage.QUEUED
    progress: int = 0
    message: str = ""
    mock: bool = False
    error: Optional[str] = None

    events: list[dict[str, Any]] = field(default_factory=list)
    commentary_text: Optional[str] = None
    commentary_provider: Optional[str] = None
    audio_path: Optional[Path] = None
    narrated_video_path: Optional[Path] = None


class JobStore:
    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    def create(self, filename: str, video_path: Path) -> Job:
        job_id = uuid.uuid4().hex[:12]
        job = Job(job_id=job_id, filename=filename, video_path=video_path)
        with self._lock:
            self._jobs[job_id] = job
        return job

    def get(self, job_id: str) -> Optional[Job]:
        with self._lock:
            return self._jobs.get(job_id)

    def update(self, job_id: str, **fields: Any) -> Optional[Job]:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return None
            for key, value in fields.items():
                setattr(job, key, value)
            return job


_store: JobStore | None = None


def get_job_store() -> JobStore:
    global _store
    if _store is None:
        _store = JobStore()
    return _store
