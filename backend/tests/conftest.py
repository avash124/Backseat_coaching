"""API test setup.

Tests run against a disposable Postgres database named by TEST_DATABASE_URL (it must end in
`_test`): its schema is rebuilt from the Alembic migrations and every table is truncated
before each test. Object storage and the Celery enqueue are replaced with in-memory fakes,
so neither S3/MinIO nor Redis is needed.
"""

import os
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import parse_qs, urlsplit

from sqlalchemy.engine import make_url

TEST_DATABASE_URL = os.environ.get(
    "TEST_DATABASE_URL", "postgresql+psycopg://nextup:nextup@localhost:5432/nextup_test"
)
if not (make_url(TEST_DATABASE_URL).database or "").endswith("_test"):
    raise RuntimeError("TEST_DATABASE_URL must name a database ending in _test")

# Settings are read once, on the first import of app.*, so this has to run before those
# imports. Real env vars beat backend/.env, so the developer's database is never used.
os.environ.update(
    DATABASE_URL=TEST_DATABASE_URL,
    REDIS_URL="redis://localhost:6379/15",
    S3_ENDPOINT_URL="http://s3.test",
    S3_ACCESS_KEY="test",
    S3_SECRET_KEY="test",
    S3_BUCKET="test-bucket",
)

import pytest
from alembic import command
from alembic.config import Config
from botocore.exceptions import ClientError
from fastapi import Header
from fastapi.testclient import TestClient
from sqlalchemy import text

from app import storage
from app.auth import current_user_id
from app.config import get_settings
from app.db import Base, engine
from app.main import app
from app.routers import uploads

BACKEND = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def alembic_cfg() -> Config:
    cfg = Config()
    cfg.set_main_option("script_location", str(BACKEND / "alembic"))
    return cfg


@pytest.fixture(scope="session", autouse=True)
def schema(alembic_cfg):
    with engine.begin() as conn:
        Base.metadata.drop_all(conn)
        conn.execute(text("DROP TABLE IF EXISTS alembic_version"))
    command.upgrade(alembic_cfg, "head")
    yield
    engine.dispose()


@pytest.fixture(autouse=True)
def clean_tables():
    tables = ", ".join(t.name for t in Base.metadata.sorted_tables)
    with engine.begin() as conn:
        conn.execute(text(f"TRUNCATE {tables} CASCADE"))


class FakeS3:
    """In-memory stand-in for the object-storage functions in app.storage."""

    def __init__(self):
        self.uploads: dict[str, dict[int, bytes]] = {}  # upload_id -> {part_number: bytes}
        self.objects: dict[str, bytes] = {}
        self.created = 0

    def create_multipart_upload(self, key, content_type):
        self.created += 1
        upload_id = f"upload-{self.created}"
        self.uploads[upload_id] = {}
        return upload_id

    def presigned_upload_parts(self, key, upload_id, part_numbers):
        return [f"https://s3.test/{key}?uploadId={upload_id}&partNumber={n}" for n in part_numbers]

    def put(self, url, data: bytes):
        """What the browser does with a presigned part URL."""
        q = parse_qs(urlsplit(url).query)
        self._parts(q["uploadId"][0])[int(q["partNumber"][0])] = data

    def _parts(self, upload_id):
        if upload_id not in self.uploads:
            raise ClientError({"Error": {"Code": "NoSuchUpload"}}, "ListParts")
        return self.uploads[upload_id]

    def list_parts(self, key, upload_id):
        parts = self._parts(upload_id)
        return [
            {"part_number": n, "size_bytes": len(d), "etag": f'"etag-{n}"'}
            for n, d in sorted(parts.items())
        ]

    def complete_multipart_upload(self, key, upload_id, parts):
        stored = self._parts(upload_id)
        assert [p["etag"] for p in parts] == [f'"etag-{n}"' for n in sorted(stored)]
        self.objects[key] = b"".join(stored[p["part_number"]] for p in parts)
        del self.uploads[upload_id]

    def abort_multipart_upload(self, key, upload_id):
        self._parts(upload_id)
        del self.uploads[upload_id]

    def read_head(self, key, n):
        return self.objects[key][:n]

    def delete_object(self, key):
        del self.objects[key]

    def presigned_get(self, key, expires=3600):
        return f"https://s3.test/{key}"

    def put_bytes(self, key, data, content_type="application/json"):
        self.objects[key] = data


@pytest.fixture
def s3(monkeypatch) -> FakeS3:
    fake = FakeS3()
    for name in (
        "create_multipart_upload",
        "presigned_upload_parts",
        "list_parts",
        "complete_multipart_upload",
        "abort_multipart_upload",
        "read_head",
        "delete_object",
        "presigned_get",
        "put_bytes",
    ):
        monkeypatch.setattr(storage, name, getattr(fake, name))
    return fake


@pytest.fixture
def enqueued(monkeypatch) -> list[str]:
    """Job ids the API handed to Celery."""
    jobs: list[str] = []
    monkeypatch.setattr(uploads, "analyze_video", SimpleNamespace(delay=jobs.append))
    return jobs


@pytest.fixture
def settings():
    return get_settings()


def _user_from_header(x_test_user: str = Header()) -> str:
    return x_test_user


@pytest.fixture
def client(s3, enqueued):
    """Requests as user "alice"; pass headers=BOB to act as someone else."""
    app.dependency_overrides[current_user_id] = _user_from_header
    yield TestClient(app, headers={"x-test-user": "alice"})
    app.dependency_overrides.clear()
