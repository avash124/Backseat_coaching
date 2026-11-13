import uuid
from datetime import datetime

from sqlalchemy import (
    BigInteger,
    DateTime,
    ForeignKey,
    Index,
    String,
    UniqueConstraint,
    func,
)
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
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class Video(Base):
    __tablename__ = "videos"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=_uuid)
    user_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("users.id", ondelete="SET NULL"), index=True
    )
    storage_key: Mapped[str] = mapped_column(String)
    # S3 multipart upload id; only meaningful while status is "uploading".
    upload_id: Mapped[str | None] = mapped_column(String)
    # Fixed when the upload is created, so a config change cannot shift the part
    # boundaries of an upload that is resumed later.
    part_size: Mapped[int | None] = mapped_column()
    filename: Mapped[str | None] = mapped_column(String)
    content_type: Mapped[str | None] = mapped_column(String)
    # uploading -> processing -> done | failed
    status: Mapped[str] = mapped_column(String, default="uploading", index=True)
    # Declared by the client at create; confirm checks the uploaded parts add up to it exactly.
    size_bytes: Mapped[int | None] = mapped_column(BigInteger)
    duration_s: Mapped[float | None] = mapped_column()
    fps: Mapped[float | None] = mapped_column()
    width: Mapped[int | None] = mapped_column()
    height: Mapped[int | None] = mapped_column()
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class Job(Base):
    __tablename__ = "jobs"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=_uuid)
    video_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("videos.id", ondelete="CASCADE"), index=True
    )
    status: Mapped[str] = mapped_column(String, default="queued", index=True)
    pipeline_version: Mapped[str] = mapped_column(String)
    error: Mapped[str | None] = mapped_column(String)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class Artifact(Base):
    """A file written to object storage by one pipeline stage.

    The unique constraint is what makes the pipeline a resumable DAG (roadmap 2.1):
    a stage looks for its own (video_id, stage, stage_version, type) row and skips
    itself if the row already exists.
    """

    __tablename__ = "artifacts"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=_uuid)
    video_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("videos.id", ondelete="CASCADE"))
    stage: Mapped[str] = mapped_column(String)
    stage_version: Mapped[str] = mapped_column(String)
    type: Mapped[str] = mapped_column(String)
    storage_key: Mapped[str] = mapped_column(String)
    meta: Mapped[dict | None] = mapped_column(JSONB)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        UniqueConstraint(
            "video_id", "stage", "stage_version", "type", name="uq_artifacts_video_stage_type"
        ),
        Index("ix_artifacts_video_stage", "video_id", "stage", "stage_version"),
    )


class Track(Base):
    """One row per tracked entity per video, NOT per frame.

    Per-frame boxes live in the `tracks.json` artifact (roadmap 2.4) — a row per
    frame per player would be ~1M rows for a single game hour at 30 fps. This table
    holds the track-level facts other tables need to join against: team label
    (Phase 4) and roster identity (Phase 7.0).
    """

    __tablename__ = "tracks"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=_uuid)
    video_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("videos.id", ondelete="CASCADE"))
    stage_version: Mapped[str] = mapped_column(String)
    # the tracker's integer id, unique within (video_id, stage_version)
    track_id: Mapped[int] = mapped_column()
    start_s: Mapped[float | None] = mapped_column()
    end_s: Mapped[float | None] = mapped_column()
    kind: Mapped[str | None] = mapped_column(String)
    team: Mapped[int | None] = mapped_column()
    jersey_number: Mapped[str | None] = mapped_column(String)
    player_name: Mapped[str | None] = mapped_column(String)

    __table_args__ = (
        UniqueConstraint(
            "video_id", "stage_version", "track_id", name="uq_tracks_video_version_track"
        ),
    )


class Play(Base):
    __tablename__ = "plays"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=_uuid)
    video_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("videos.id", ondelete="CASCADE"))
    stage_version: Mapped[str] = mapped_column(String)
    start_s: Mapped[float] = mapped_column()
    end_s: Mapped[float] = mapped_column()
    team: Mapped[int | None] = mapped_column()
    play_type: Mapped[str | None] = mapped_column(String)
    outcome: Mapped[str | None] = mapped_column(String)
    confidence: Mapped[float | None] = mapped_column()
    suggestions: Mapped[list | None] = mapped_column(JSONB)
    coach_verdict: Mapped[str] = mapped_column(String, default="unreviewed")
    coach_note: Mapped[str | None] = mapped_column(String)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (Index("ix_plays_video_version", "video_id", "stage_version"),)


class Observation(Base):
    """A diagnostic claim about one moment in one play (roadmap 7.5).

    Nothing writes to this until Phase 7.5, but it exists in schema v1 so diagnostics
    are first-class data rather than something bolted onto `stats` later.

    An observation is a falsifiable claim about a named player, so `evidence` is not
    optional in spirit: it holds the measured numbers behind the claim
    (e.g. {"separation_ft_at_catch": 1.4, "separation_ft_at_shot": 1.1}) which the UI
    shows instead of a bare verdict.
    """

    __tablename__ = "observations"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=_uuid)
    play_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("plays.id", ondelete="CASCADE"), index=True
    )
    stage_version: Mapped[str] = mapped_column(String)
    type: Mapped[str] = mapped_column(String, index=True)
    subject_track_id: Mapped[int | None] = mapped_column()
    object_track_id: Mapped[int | None] = mapped_column()
    t_s: Mapped[float] = mapped_column()
    severity: Mapped[float | None] = mapped_column()
    confidence: Mapped[float | None] = mapped_column()
    evidence: Mapped[dict | None] = mapped_column(JSONB)
    coach_verdict: Mapped[str] = mapped_column(String, default="unreviewed", index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (Index("ix_observations_play_type", "play_id", "type"),)


class Stat(Base):
    __tablename__ = "stats"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=_uuid)
    video_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("videos.id", ondelete="CASCADE"))
    stage_version: Mapped[str] = mapped_column(String)
    key: Mapped[str] = mapped_column(String)
    value: Mapped[dict] = mapped_column(JSONB)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        UniqueConstraint("video_id", "stage_version", "key", name="uq_stats_video_version_key"),
    )
