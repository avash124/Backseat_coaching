import uuid
from datetime import datetime

from sqlalchemy import ForeignKey, Index, String, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.db import Base


def _uuid() -> uuid.UUID:
    return uuid.uuid4()


class User(Base):
    __tablename__ = "users"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=_uuid)
    # Supabase auth user id (the JWT `sub`). Lazily created on first authed request.
    external_auth_id: Mapped[str | None] = mapped_column(String, unique=True, index=True)
    email: Mapped[str | None] = mapped_column(String)
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())


class Video(Base):
    __tablename__ = "videos"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=_uuid)
    user_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("users.id"))
    storage_key: Mapped[str] = mapped_column(String)
    filename: Mapped[str | None] = mapped_column(String)
    status: Mapped[str] = mapped_column(String, default="uploaded")
    duration_s: Mapped[float | None] = mapped_column()
    fps: Mapped[float | None] = mapped_column()
    width: Mapped[int | None] = mapped_column()
    height: Mapped[int | None] = mapped_column()
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())


class Job(Base):
    __tablename__ = "jobs"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=_uuid)
    video_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("videos.id"), index=True)
    status: Mapped[str] = mapped_column(String, default="queued")
    pipeline_version: Mapped[str] = mapped_column(String)
    error: Mapped[str | None] = mapped_column(String)
    started_at: Mapped[datetime | None] = mapped_column()
    finished_at: Mapped[datetime | None] = mapped_column()
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())


class Artifact(Base):
    __tablename__ = "artifacts"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=_uuid)
    video_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("videos.id"))
    pipeline_version: Mapped[str] = mapped_column(String)
    type: Mapped[str] = mapped_column(String)
    storage_key: Mapped[str] = mapped_column(String)
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())

    __table_args__ = (Index("ix_artifacts_video_pipeline", "video_id", "pipeline_version"),)

class Track(Base):
    __tablename__ = "tracks"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=_uuid)
    video_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("videos.id"))
    pipeline_version: Mapped[str] = mapped_column(String)
    frame: Mapped[int] = mapped_column()
    track_id: Mapped[int] = mapped_column()
    bbox: Mapped[dict] = mapped_column(JSONB)  # {x, y, w, h}
    foot_x: Mapped[float | None] = mapped_column()
    foot_y: Mapped[float | None] = mapped_column()
    team: Mapped[int | None] = mapped_column()

    __table_args__ = (Index("ix_tracks_video_pipeline", "video_id", "pipeline_version"),)


class Play(Base):
    __tablename__ = "plays"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=_uuid)
    video_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("videos.id"))
    pipeline_version: Mapped[str] = mapped_column(String)
    start_s: Mapped[float] = mapped_column()
    end_s: Mapped[float] = mapped_column()
    team: Mapped[int | None] = mapped_column()
    play_type: Mapped[str | None] = mapped_column(String)
    outcome: Mapped[str | None] = mapped_column(String)
    confidence: Mapped[float | None] = mapped_column()

    __table_args__ = (Index("ix_plays_video_pipeline", "video_id", "pipeline_version"),)


class Stat(Base):
    __tablename__ = "stats"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=_uuid)
    video_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("videos.id"))
    pipeline_version: Mapped[str] = mapped_column(String)
    key: Mapped[str] = mapped_column(String)
    value: Mapped[dict] = mapped_column(JSONB)
