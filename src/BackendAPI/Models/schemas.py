"""API veri modelleri (Pydantic)."""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel


class JobStage(str, Enum):
    QUEUED = "queued"
    PROCESSING = "processing"        # VisionEngine pas tespiti çalışıyor
    COMMENTARY = "commentary"        # LLM spiker metni üretiliyor
    SYNTHESIZING = "synthesizing"    # TTS ile sese çevriliyor
    DONE = "done"
    FAILED = "failed"


class PassEventModel(BaseModel):
    """VisionEngine'in ürettiği tek bir pas event'i (özet alanlar)."""

    event_id: int
    event_type: str
    start_time_sec: Optional[float] = None
    end_time_sec: Optional[float] = None
    from_player_id: Optional[int] = None
    to_player_id: Optional[int] = None
    from_team_id: Optional[int] = None
    to_team_id: Optional[int] = None
    confidence: Optional[float] = None


class JobResponse(BaseModel):
    job_id: str
    filename: str
    stage: JobStage
    progress: int = 0           # 0..100
    message: str = ""
    mock: bool = False
    event_count: int = 0
    has_video: bool = False
    error: Optional[str] = None


class CommentaryResponse(BaseModel):
    job_id: str
    provider: str
    text: str
    audio_url: Optional[str] = None
    video_url: Optional[str] = None


class EventsResponse(BaseModel):
    job_id: str
    events: list[PassEventModel]
