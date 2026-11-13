import uuid

from botocore.exceptions import ClientError
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select, update
from sqlalchemy.orm import Session

from app import storage
from app.config import get_settings
from app.db import get_db
from app.deps import get_current_user
from app.models import Job, User, Video
from app.schemas import (
    ConfirmUploadResponse,
    CreateUploadRequest,
    CreateUploadResponse,
    PartUrl,
    PresignPartsRequest,
    PresignPartsResponse,
    UploadedPart,
    UploadPartsResponse,
)
from app.worker.tasks import analyze_video

router = APIRouter(prefix="/uploads", tags=["uploads"])

ALLOWED_CONTENT_TYPES = {"video/mp4", "video/quicktime"}
S3_MAX_PARTS = 10_000


def _part_count(video: Video) -> int:
    return -(-video.size_bytes // video.part_size)


def _owned_video(db: Session, video_id: uuid.UUID, user: User, lock: bool = False) -> Video:
    video = db.get(Video, video_id, with_for_update=lock)
    if video is None or video.user_id != user.id:
        raise HTTPException(status_code=404, detail="Video not found")
    return video


def _uploading_video(db: Session, video_id: uuid.UUID, user: User, lock: bool = False) -> Video:
    video = _owned_video(db, video_id, user, lock)
    if video.status != "uploading":
        raise HTTPException(status_code=409, detail=f"Upload is {video.status}, not uploading")
    return video


def _is_no_such_upload(exc: ClientError) -> bool:
    return exc.response.get("Error", {}).get("Code") == "NoSuchUpload"


def _list_parts(db: Session, video: Video) -> list[dict]:
    try:
        return storage.list_parts(video.storage_key, video.upload_id)
    except ClientError as exc:
        if not _is_no_such_upload(exc):
            raise
        # Aborted out from under us (e.g. a bucket lifecycle rule): it cannot be resumed.
        # Conditional, so a GET racing a successful confirm cannot undo "processing".
        db.execute(
            update(Video)
            .where(Video.id == video.id, Video.status == "uploading")
            .values(status="failed")
        )
        db.commit()
        raise HTTPException(status_code=409, detail="Upload no longer exists") from exc


@router.post("", response_model=CreateUploadResponse)
def create_upload(
    body: CreateUploadRequest,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> CreateUploadResponse:
    settings = get_settings()
    if body.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(status_code=415, detail="Only MP4 and MOV uploads are supported")
    if body.size_bytes > settings.max_upload_bytes:
        raise HTTPException(status_code=413, detail="File exceeds the upload size cap")

    video = Video(
        user_id=user.id,
        filename=body.filename,
        content_type=body.content_type,
        size_bytes=body.size_bytes,
        # Grow the part only if the configured size would need more parts than S3 allows.
        part_size=max(settings.upload_part_size, -(-body.size_bytes // S3_MAX_PARTS)),
        storage_key="",
        status="uploading",
    )
    db.add(video)
    db.flush()  # get video.id

    video.storage_key = storage.raw_key(str(video.id))
    video.upload_id = storage.create_multipart_upload(video.storage_key, body.content_type)
    db.commit()

    return CreateUploadResponse(
        video_id=video.id,
        storage_key=video.storage_key,
        part_size=video.part_size,
        part_count=_part_count(video),
    )


@router.post("/{video_id}/parts", response_model=PresignPartsResponse)
def presign_parts(
    video_id: uuid.UUID,
    body: PresignPartsRequest,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> PresignPartsResponse:
    video = _uploading_video(db, video_id, user)
    part_count = _part_count(video)
    if any(not 1 <= n <= part_count for n in body.part_numbers):
        raise HTTPException(status_code=422, detail=f"Part numbers must be in 1..{part_count}")

    urls = storage.presigned_upload_parts(video.storage_key, video.upload_id, body.part_numbers)
    return PresignPartsResponse(
        parts=[PartUrl(part_number=n, url=u) for n, u in zip(body.part_numbers, urls, strict=True)]
    )


@router.get("/{video_id}/parts", response_model=UploadPartsResponse)
def list_uploaded_parts(
    video_id: uuid.UUID,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> UploadPartsResponse:
    video = _uploading_video(db, video_id, user)
    parts = _list_parts(db, video)
    return UploadPartsResponse(
        part_size=video.part_size,
        part_count=_part_count(video),
        uploaded=[
            UploadedPart(part_number=p["part_number"], size_bytes=p["size_bytes"]) for p in parts
        ],
    )


@router.post("/{video_id}/confirm", response_model=ConfirmUploadResponse)
def confirm_upload(
    video_id: uuid.UUID,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> ConfirmUploadResponse:
    settings = get_settings()
    # Row lock: a double-clicked confirm waits here until the first one has committed.
    video = _owned_video(db, video_id, user, lock=True)
    if video.status != "uploading":
        # Already confirmed: a retry after a lost response gets the same job back.
        job = db.scalar(
            select(Job).where(Job.video_id == video.id).order_by(Job.created_at.desc()).limit(1)
        )
        if job is None:
            raise HTTPException(status_code=409, detail=f"Upload is {video.status}, not uploading")
        return ConfirmUploadResponse(job_id=job.id)

    # Every part must have exactly the size the contract gives it, so the object is exactly
    # the declared size_bytes (already checked against the cap at create). A wrong-sized part
    # is treated like a missing one: the upload stays resumable.
    parts = _list_parts(db, video)
    sizes = {p["part_number"]: p["size_bytes"] for p in parts}
    part_count = _part_count(video)
    bad = [
        n
        for n in range(1, part_count + 1)
        if sizes.get(n) != min(video.part_size, video.size_bytes - (n - 1) * video.part_size)
    ]
    if bad:
        raise HTTPException(
            status_code=409, detail=f"{len(bad)} part(s) missing or the wrong size"
        )
    parts = [p for p in parts if p["part_number"] <= part_count]

    # The server, not the browser, supplies the ETags, so bucket CORS needs no ExposeHeaders.
    storage.complete_multipart_upload(video.storage_key, video.upload_id, parts)

    # Cheap container sniff: MP4 and MOV both open with an ISO-BMFF `ftyp` box.
    # Duration and codec checks need a decoder and belong to the Phase 2 decode stage.
    if storage.read_head(video.storage_key, 12)[4:8] != b"ftyp":
        storage.delete_object(video.storage_key)
        video.status = "failed"
        db.commit()
        raise HTTPException(status_code=422, detail="Not an MP4/MOV file")

    video.status = "processing"
    job = Job(video_id=video.id, status="queued", pipeline_version=settings.pipeline_version)
    db.add(job)
    db.commit()

    analyze_video.delay(str(job.id))
    return ConfirmUploadResponse(job_id=job.id)


@router.delete("/{video_id}", status_code=204)
def abort_upload(
    video_id: uuid.UUID,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> None:
    video = _uploading_video(db, video_id, user, lock=True)
    try:
        storage.abort_multipart_upload(video.storage_key, video.upload_id)
    except ClientError as exc:
        if not _is_no_such_upload(exc):
            raise
    video.status = "failed"
    db.commit()
