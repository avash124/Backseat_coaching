# One image for the API and the Celery worker: same code, different command.
#   API (default):  uvicorn on $PORT, or 8000 when PORT is unset
#   Worker:         celery -A app.worker.celery_app.celery_app worker --loglevel=info
#   Migrations:     alembic upgrade head
# Build from backend/:  docker build -t nextup-backend .
FROM python:3.13.15-slim-trixie

COPY --from=ghcr.io/astral-sh/uv:0.12.0 /uv /uvx /bin/

ENV PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1

WORKDIR /app

# Base [project] dependencies only: no dev, cv, eval or remote groups.
COPY pyproject.toml ./
RUN uv pip install --system --no-cache -r pyproject.toml

COPY alembic.ini ./
COPY alembic ./alembic
COPY app ./app

RUN useradd --create-home --uid 10001 app
USER app

EXPOSE 8000
CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
