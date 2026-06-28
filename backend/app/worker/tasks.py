import json
import time
from datetime import datetime

from app import storage
from app.config import get_settings
from app.db import SessionLocal
from app.models import Artifact, Job, Play, Video
from app.worker.celery_app import celery_app

FAKE_PLAYS = [
    {"start_s": 5.0, "end_s": 12.0, "team": 0, "play_type": "pick_and_roll", "outcome": "made_2", "confidence": 0.81},
    {"start_s": 18.0, "end_s": 24.0, "team": 1, "play_type": "isolation", "outcome": "miss", "confidence": 0.66},
    {"start_s": 30.0, "end_s": 37.0, "team": 0, "play_type": "transition", "outcome": "made_3", "confidence": 0.74},
]


@celery_app.task(name="analyze_video")
def analyze_video(job_id: str) -> None:
    db = SessionLocal()
    pv = get_settings().pipeline_version
    try:
        job = db.get(Job, job_id)
        if job is None:
            return
        job.status = "processing"
        job.started_at = datetime.utcnow()
        db.commit()

        time.sleep(10)  # pretend to do heavy GPU work

        video = db.get(Video, job.video_id)
        vid = str(video.id)
        tracks_key = storage.artifact_key(vid, pv, "tracks.json")
        storage.put_bytes(
            tracks_key,
            json.dumps({"frames": [], "note": "stub artifact"}).encode(),
        )
        db.add(Artifact(video_id=video.id, pipeline_version=pv, type="tracks", storage_key=tracks_key))

        for p in FAKE_PLAYS:
            db.add(Play(video_id=video.id, pipeline_version=pv, **p))

        video.status = "done"
        job.status = "done"
        job.finished_at = datetime.utcnow()
        db.commit()
    except Exception as exc:  # noqa: BLE001
        db.rollback()
        job = db.get(Job, job_id)
        if job:
            job.status = "failed"
            job.error = str(exc)
            db.commit()
        raise
    finally:
        db.close()
