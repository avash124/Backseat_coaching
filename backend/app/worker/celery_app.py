from celery import Celery

from app.config import get_settings

settings = get_settings()

celery_app = Celery(
    "nextup",
    broker=settings.redis_url,
    backend=settings.redis_url,
    include=["app.worker.tasks"],
)
celery_app.conf.task_track_started = True
