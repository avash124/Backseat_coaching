import time
import uuid

import pytest
from fastapi import HTTPException
from sqlalchemy.orm.attributes import set_committed_value

from app.db import SessionLocal
from app.models import Job, Video
from app.routers import uploads
from app.worker import tasks

BOB = {"x-test-user": "bob"}

# 25 bytes that open with an ISO-BMFF ftyp box; with 10-byte parts that is 3 parts.
MP4 = b"\x00\x00\x00\x14ftypisom\x00\x00\x02\x00isom" + b"\x00" * 5
PART = 10


@pytest.fixture
def small_parts(monkeypatch, settings):
    monkeypatch.setattr(settings, "upload_part_size", PART)


def _video(video_id) -> Video:
    with SessionLocal() as db:
        return db.get(Video, uuid.UUID(video_id))


def _create(client, data=MP4, **body):
    resp = client.post("/uploads", json={"filename": "game.mp4", "size_bytes": len(data), **body})
    assert resp.status_code == 200, resp.text
    return resp.json()


def _upload(client, s3, video_id, data, part_numbers, part_size=PART):
    resp = client.post(f"/uploads/{video_id}/parts", json={"part_numbers": part_numbers})
    assert resp.status_code == 200, resp.text
    for part in resp.json()["parts"]:
        n = part["part_number"]
        s3.put(part["url"], data[(n - 1) * part_size : n * part_size])


def _confirm(client, video_id):
    return client.post(f"/uploads/{video_id}/confirm")


def test_create_uses_configured_part_size(client, settings):
    size = 6 * 1024**3
    up = _create(client, size_bytes=size)

    assert up["part_size"] == settings.upload_part_size == 16 * 1024 * 1024
    assert up["part_count"] == 384
    assert up["storage_key"] == f"videos/{up['video_id']}/raw.mp4"
    video = _video(up["video_id"])
    assert video.status == "uploading"
    assert video.upload_id
    assert video.size_bytes == size


def test_part_size_grows_to_stay_within_s3_part_limit(client, small_parts):
    up = _create(client, size_bytes=200_001)
    assert up["part_size"] == 21
    assert up["part_count"] <= 10_000


def test_happy_path_create_parts_confirm_job_results(
    client, s3, enqueued, small_parts, monkeypatch
):
    up = _create(client)
    vid = up["video_id"]
    assert (up["part_size"], up["part_count"]) == (PART, 3)

    _upload(client, s3, vid, MP4, [1, 2, 3])
    resp = _confirm(client, vid)
    assert resp.status_code == 200, resp.text
    job_id = resp.json()["job_id"]

    assert enqueued == [job_id]
    assert s3.objects[up["storage_key"]] == MP4
    video = _video(vid)
    assert (video.status, video.size_bytes) == ("processing", len(MP4))

    job = client.get(f"/jobs/{job_id}").json()
    assert job == {
        "id": job_id,
        "video_id": vid,
        "status": "queued",
        "pipeline_version": "v0-stub",
        "error": None,
    }

    # Run the stub worker in-process, as Celery would.
    monkeypatch.setattr(time, "sleep", lambda _s: None)
    tasks.analyze_video(job_id)

    assert client.get(f"/jobs/{job_id}").json()["status"] == "done"
    results = client.get(f"/videos/{vid}/results").json()
    assert results["status"] == "done"
    assert results["video_url"] == f"https://s3.test/{up['storage_key']}"
    assert [p["start_s"] for p in results["plays"]] == [5.0, 18.0, 30.0]
    assert set(results["plays"][0]) == {
        "start_s", "end_s", "team", "play_type", "outcome", "confidence"
    }


def test_resume_reflects_uploaded_parts(client, s3, small_parts):
    vid = _create(client)["video_id"]
    _upload(client, s3, vid, MP4, [1, 3])

    resp = client.get(f"/uploads/{vid}/parts")
    assert resp.status_code == 200
    assert resp.json() == {
        "part_size": PART,
        "part_count": 3,
        "uploaded": [{"part_number": 1, "size_bytes": 10}, {"part_number": 3, "size_bytes": 5}],
    }

    assert _confirm(client, vid).status_code == 409  # part 2 missing
    assert _video(vid).status == "uploading"  # still resumable

    _upload(client, s3, vid, MP4, [2])
    assert _confirm(client, vid).status_code == 200


def test_quicktime_is_accepted(client):
    assert _create(client, content_type="video/quicktime")["part_count"] == 1


def test_create_rejects_unsupported_content_type(client):
    resp = client.post(
        "/uploads", json={"filename": "a.mkv", "content_type": "video/x-matroska", "size_bytes": 9}
    )
    assert resp.status_code == 415


def test_create_rejects_declared_size_over_cap(client, settings):
    body = {"filename": "big.mp4", "size_bytes": settings.max_upload_bytes + 1}
    assert client.post("/uploads", json=body).status_code == 413


def test_create_rejects_non_positive_size(client):
    assert client.post("/uploads", json={"filename": "a.mp4", "size_bytes": 0}).status_code == 422


def test_presign_rejects_bad_part_numbers(client, small_parts):
    vid = _create(client)["video_id"]
    for numbers in ([0], [4], list(range(1, 102))):
        resp = client.post(f"/uploads/{vid}/parts", json={"part_numbers": numbers})
        assert resp.status_code == 422, numbers


def test_confirm_rejects_non_mp4(client, s3, small_parts):
    junk = b"this is not a video at all"
    up = _create(client, data=junk)
    _upload(client, s3, up["video_id"], junk, [1, 2, 3])

    assert _confirm(client, up["video_id"]).status_code == 422
    assert up["storage_key"] not in s3.objects
    assert _video(up["video_id"]).status == "failed"
    assert _confirm(client, up["video_id"]).status_code == 409


def test_confirm_rejects_parts_that_do_not_match_declared_size(client, s3, small_parts):
    up = _create(client)  # declares 25 bytes: parts of 10, 10, 5
    vid = up["video_id"]
    _upload(client, s3, vid, MP4 + MP4, [1, 2, 3], part_size=20)  # 20, 20, 10 arrive

    resp = _confirm(client, vid)
    assert resp.status_code == 409
    assert "wrong size" in resp.json()["detail"]
    assert up["storage_key"] not in s3.objects  # nothing completed
    assert _video(vid).status == "uploading"  # still resumable

    _upload(client, s3, vid, MP4, [1, 2, 3])  # re-send at the right size
    assert _confirm(client, vid).status_code == 200
    assert s3.objects[up["storage_key"]] == MP4


def test_confirm_again_returns_the_same_job(client, s3, enqueued, small_parts):
    """A client whose confirm response was lost can retry without starting over."""
    vid = _create(client)["video_id"]
    _upload(client, s3, vid, MP4, [1, 2, 3])
    first = _confirm(client, vid)
    again = _confirm(client, vid)
    assert first.status_code == again.status_code == 200
    assert again.json() == first.json()
    assert len(enqueued) == 1
    with SessionLocal() as db:
        assert db.query(Job).count() == 1


def test_malformed_ids_are_422_not_500(client):
    assert client.get("/jobs/not-a-uuid").status_code == 422
    assert client.get("/videos/not-a-uuid/results").status_code == 422


def test_abort(client, s3, small_parts):
    vid = _create(client)["video_id"]
    _upload(client, s3, vid, MP4, [1])

    resp = client.delete(f"/uploads/{vid}")
    assert resp.status_code == 204
    assert resp.content == b""
    assert s3.uploads == {}
    assert _video(vid).status == "failed"

    assert client.delete(f"/uploads/{vid}").status_code == 409
    assert client.get(f"/uploads/{vid}/parts").status_code == 409
    assert client.post(f"/uploads/{vid}/parts", json={"part_numbers": [1]}).status_code == 409
    assert _confirm(client, vid).status_code == 409  # failed with no job: nothing to return


def test_delete_after_confirm_is_409(client, s3, small_parts):
    vid = _create(client)["video_id"]
    _upload(client, s3, vid, MP4, [1, 2, 3])
    assert _confirm(client, vid).status_code == 200
    assert client.delete(f"/uploads/{vid}").status_code == 409


def test_upload_missing_from_storage_is_409(client, s3, small_parts):
    vid = _create(client)["video_id"]
    s3.uploads.clear()  # e.g. a bucket lifecycle rule aborted it

    assert client.get(f"/uploads/{vid}/parts").status_code == 409
    assert _video(vid).status == "failed"


def test_other_users_video_and_job_are_404(client, s3, small_parts):
    up = _create(client)
    vid = up["video_id"]
    _upload(client, s3, vid, MP4, [1, 2])

    resp = client.post(f"/uploads/{vid}/parts", json={"part_numbers": [3]}, headers=BOB)
    assert resp.status_code == 404
    assert client.get(f"/uploads/{vid}/parts", headers=BOB).status_code == 404
    assert client.post(f"/uploads/{vid}/confirm", headers=BOB).status_code == 404
    assert client.delete(f"/uploads/{vid}", headers=BOB).status_code == 404
    assert _video(vid).status == "uploading"  # bob changed nothing

    _upload(client, s3, vid, MP4, [3])
    job_id = _confirm(client, vid).json()["job_id"]
    assert client.get(f"/jobs/{job_id}", headers=BOB).status_code == 404
    assert client.get(f"/videos/{vid}/results", headers=BOB).status_code == 404
    assert client.get(f"/jobs/{job_id}").status_code == 200
    assert client.get(f"/videos/{vid}/results").status_code == 200


def test_unknown_video_is_404(client):
    vid = uuid.uuid4()
    assert client.get(f"/uploads/{vid}/parts").status_code == 404
    assert client.post(f"/uploads/{vid}/confirm").status_code == 404
    assert client.delete(f"/uploads/{vid}").status_code == 404


def test_storage_gone_after_confirm_does_not_mark_failed(client, s3, small_parts):
    """A GET /parts that read "uploading" just before a confirm committed must not undo it."""
    vid = _create(client)["video_id"]
    _upload(client, s3, vid, MP4, [1, 2, 3])
    assert _confirm(client, vid).status_code == 200  # completes, so the upload id is gone

    with SessionLocal() as db:
        stale = db.get(Video, uuid.UUID(vid))
        set_committed_value(stale, "status", "uploading")  # what the racing request read
        with pytest.raises(HTTPException) as exc:
            uploads._list_parts(db, stale)
    assert exc.value.status_code == 409
    assert _video(vid).status == "processing"
