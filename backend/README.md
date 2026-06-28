# Next-Up backend (Phase 1 skeleton)

Thin FastAPI + Celery slice that proves the async path end-to-end with a **stub**
worker (no ML yet). Data flow:

```
browser -> presigned PUT -> S3/MinIO (raw.mp4)
confirm -> create video + job -> Redis -> Celery stub (sleep, write fake plays)
browser polls /jobs/{id} -> /videos/{id}/results renders fake plays
```

## Run locally

1. Start infra (from repo root):
   ```
   docker compose -f infra/docker-compose.yml up -d
   ```
2. Configure env:
   ```
   cp .env.example .env     # defaults already point at the compose services
   ```
3. Install deps (uv):
   ```
   uv sync
   ```
4. Create the schema:
   ```
   uv run alembic revision --autogenerate -m "init"
   uv run alembic upgrade head
   ```
5. Run the API and the worker in two terminals:
   ```
   uv run uvicorn app.main:app --reload
   uv run celery -A app.worker.celery_app.celery_app worker --loglevel=info --pool=solo
   ```
   (`--pool=solo` is needed on Windows.)

## Smoke test (no auth, AUTH_ENABLED=false)

```
# 1. create upload
curl -X POST localhost:8000/uploads -H "content-type: application/json" \
  -d '{"filename":"game.mp4"}'
# -> {"video_id": "...", "upload_url": "...", "storage_key": "..."}

# 2. PUT a file to upload_url (any bytes work for the stub)
curl -X PUT "<upload_url>" --upload-file game.mp4 -H "content-type: video/mp4"

# 3. confirm -> enqueues the stub job
curl -X POST localhost:8000/uploads/<video_id>/confirm   # -> {"job_id": "..."}

# 4. poll
curl localhost:8000/jobs/<job_id>                         # status: queued->processing->done

# 5. results
curl localhost:8000/videos/<video_id>/results             # fake plays
```

## Turning on Supabase auth

Set `AUTH_ENABLED=true`, fill `SUPABASE_JWKS_URL`, and send
`Authorization: Bearer <supabase access token>` from the frontend.
