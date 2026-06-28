from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from app import storage
from app.db import get_db
from app.deps import get_current_user
from app.models import Play, User, Video
from app.schemas import PlayOut, ResultsResponse

router = APIRouter(prefix="/videos", tags=["results"])


@router.get("/{video_id}/results", response_model=ResultsResponse)
def get_results(
    video_id: str,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> ResultsResponse:
    video = db.get(Video, video_id)
    if video is None or video.user_id != user.id:
        raise HTTPException(status_code=404, detail="Video not found")

    plays = db.scalars(select(Play).where(Play.video_id == video.id).order_by(Play.start_s)).all()
    return ResultsResponse(
        video_id=video.id,
        status=video.status,
        video_url=storage.presigned_get(video.storage_key) if video.storage_key else None,
        plays=[PlayOut.model_validate(p, from_attributes=True) for p in plays],
    )
