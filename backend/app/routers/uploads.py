from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app import storage
from app.config import get_settings
from app.db import get_db
from app.deps import get_current_user
from app.models import Job, User, Video
from app.schemas import ConfirmUploadResponse, CreateUploadRequest, CreateUploadResponse
from app.worker.tasks import analyze_video

router = APIRouter(prefix="/uploads", tags=["uploads"])


@router.post("", response_model=CreateUploadResponse)
def create_upload(
    body: CreateUploadRequest,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> CreateUploadResponse:
    video = Video(user_id=user.id, filename=body.filename, storage_key="", status="uploaded")
    db.add(video)
    db.flush()  # get video.id

    key = storage.raw_key(str(video.id))
    video.storage_key = key
    db.commit()

    return CreateUploadResponse(
        video_id=video.id,
        upload_url=storage.presigned_put(key, body.content_type),
        storage_key=key,
    )


@router.post("/{video_id}/confirm", response_model=ConfirmUploadResponse)
def confirm_upload(
    video_id: str,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> ConfirmUploadResponse:
    video = db.get(Video, video_id)
    if video is None or video.user_id != user.id:
        raise HTTPException(status_code=404, detail="Video not found")

    video.status = "processing"
    job = Job(video_id=video.id, status="queued", pipeline_version=get_settings().pipeline_version)
    db.add(job)
    db.commit()

    analyze_video.delay(str(job.id))
    return ConfirmUploadResponse(job_id=job.id)
