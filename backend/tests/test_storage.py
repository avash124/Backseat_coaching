"""app.storage against a stubbed boto3 client: the S3 calls the API fakes elsewhere."""

from io import BytesIO
from urllib.parse import parse_qs, urlsplit

import boto3
import pytest
from botocore.client import Config
from botocore.response import StreamingBody
from botocore.stub import Stubber

from app import storage

KEY, UPLOAD = "videos/abc/raw.mp4", "upload-1"
BASE = {"Bucket": "test-bucket", "Key": KEY, "UploadId": UPLOAD}


@pytest.fixture
def stubbed(monkeypatch):
    client = boto3.client(
        "s3",
        endpoint_url="http://s3.test",
        aws_access_key_id="test",
        aws_secret_access_key="test",
        region_name="us-east-1",
        config=Config(signature_version="s3v4"),
    )
    monkeypatch.setattr(storage, "_client", lambda: client)
    with Stubber(client) as stub:
        yield stub
        stub.assert_no_pending_responses()


def _parts(numbers):
    return [{"PartNumber": n, "Size": 16 * 1024 * 1024, "ETag": f'"e{n}"'} for n in numbers]


def test_list_parts_follows_pagination(stubbed):
    stubbed.add_response(
        "list_parts",
        {"Parts": _parts(range(1, 1001)), "IsTruncated": True, "NextPartNumberMarker": 1000},
        BASE,
    )
    stubbed.add_response(
        "list_parts",
        {"Parts": _parts(range(1001, 1201)), "IsTruncated": False},
        {**BASE, "PartNumberMarker": 1000},
    )

    parts = storage.list_parts(KEY, UPLOAD)

    assert [p["part_number"] for p in parts] == list(range(1, 1201))
    assert parts[1199] == {"part_number": 1200, "size_bytes": 16 * 1024 * 1024, "etag": '"e1200"'}


def test_complete_sends_the_listed_etags(stubbed):
    stubbed.add_response(
        "complete_multipart_upload",
        {},
        {
            **BASE,
            "MultipartUpload": {
                "Parts": [{"PartNumber": 1, "ETag": '"e1"'}, {"PartNumber": 2, "ETag": '"e2"'}]
            },
        },
    )
    parts = [
        {"part_number": 1, "size_bytes": 10, "etag": '"e1"'},
        {"part_number": 2, "size_bytes": 5, "etag": '"e2"'},
    ]
    storage.complete_multipart_upload(KEY, UPLOAD, parts)


def test_read_head_is_a_ranged_get(stubbed):
    head = b"\x00\x00\x00\x14ftypisom"
    stubbed.add_response(
        "get_object",
        {"Body": StreamingBody(BytesIO(head), len(head))},
        {"Bucket": "test-bucket", "Key": KEY, "Range": "bytes=0-11"},
    )
    assert storage.read_head(KEY, 12) == head


def test_part_urls_need_no_extra_headers():
    """The browser PUTs raw bytes, so the signature must cover only the host header."""
    urls = storage.presigned_upload_parts(KEY, UPLOAD, [1, 2])

    for n, url in enumerate(urls, start=1):
        q = parse_qs(urlsplit(url).query)
        assert urlsplit(url).path == f"/test-bucket/{KEY}"
        assert q["partNumber"] == [str(n)]
        assert q["uploadId"] == [UPLOAD]
        assert q["X-Amz-SignedHeaders"] == ["host"]
