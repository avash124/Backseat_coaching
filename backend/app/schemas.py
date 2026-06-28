import uuid

from pydantic import BaseModel


class CreateUploadRequest(BaseModel):
    filename: str
    content_type: str = "video/mp4"


class CreateUploadResponse(BaseModel):
    video_id: uuid.UUID
    upload_url: str
    storage_key: str


class ConfirmUploadResponse(BaseModel):
    job_id: uuid.UUID


class JobStatusResponse(BaseModel):
    id: uuid.UUID
    video_id: uuid.UUID
    status: str
    pipeline_version: str
    error: str | None = None


class PlayOut(BaseModel):
    start_s: float
    end_s: float
    team: int | None = None
    play_type: str | None = None
    outcome: str | None = None
    confidence: float | None = None


class ResultsResponse(BaseModel):
    video_id: uuid.UUID
    status: str
    video_url: str | None = None
    plays: list[PlayOut] = []
