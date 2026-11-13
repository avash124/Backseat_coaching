import boto3
from botocore.client import Config

from app.config import get_settings


def _client():
    s = get_settings()
    return boto3.client(
        "s3",
        endpoint_url=s.s3_endpoint_url,
        aws_access_key_id=s.s3_access_key,
        aws_secret_access_key=s.s3_secret_key,
        region_name=s.s3_region,
        config=Config(signature_version="s3v4"),
    )


def raw_key(video_id: str) -> str:
    return f"videos/{video_id}/raw.mp4"


def artifact_key(video_id: str, stage: str, stage_version: str, name: str) -> str:
    return f"videos/{video_id}/{stage}/{stage_version}/{name}"


def presigned_get(key: str, expires: int = 3600) -> str:
    s = get_settings()
    if s.s3_public_url:
        return f"{s.s3_public_url}/{key}"
    return _client().generate_presigned_url(
        "get_object",
        Params={"Bucket": s.s3_bucket, "Key": key},
        ExpiresIn=expires,
    )


def put_bytes(key: str, data: bytes, content_type: str = "application/json") -> None:
    _client().put_object(
        Bucket=get_settings().s3_bucket, Key=key, Body=data, ContentType=content_type
    )


def create_multipart_upload(key: str, content_type: str) -> str:
    resp = _client().create_multipart_upload(
        Bucket=get_settings().s3_bucket, Key=key, ContentType=content_type
    )
    return resp["UploadId"]


def presigned_upload_parts(
    key: str, upload_id: str, part_numbers: list[int], expires: int = 3600
) -> list[str]:
    client = _client()
    bucket = get_settings().s3_bucket
    return [
        client.generate_presigned_url(
            "upload_part",
            Params={"Bucket": bucket, "Key": key, "UploadId": upload_id, "PartNumber": n},
            ExpiresIn=expires,
        )
        for n in part_numbers
    ]


def list_parts(key: str, upload_id: str) -> list[dict]:
    """Parts S3 has received so far, in ascending part-number order."""
    pages = (
        _client()
        .get_paginator("list_parts")
        .paginate(Bucket=get_settings().s3_bucket, Key=key, UploadId=upload_id)
    )
    return [
        {"part_number": p["PartNumber"], "size_bytes": p["Size"], "etag": p["ETag"]}
        for page in pages
        for p in page.get("Parts", [])
    ]


def complete_multipart_upload(key: str, upload_id: str, parts: list[dict]) -> None:
    """`parts` is what `list_parts` returned."""
    _client().complete_multipart_upload(
        Bucket=get_settings().s3_bucket,
        Key=key,
        UploadId=upload_id,
        MultipartUpload={
            "Parts": [{"PartNumber": p["part_number"], "ETag": p["etag"]} for p in parts]
        },
    )


def abort_multipart_upload(key: str, upload_id: str) -> None:
    _client().abort_multipart_upload(
        Bucket=get_settings().s3_bucket, Key=key, UploadId=upload_id
    )


def read_head(key: str, n: int) -> bytes:
    """The first `n` bytes of an object (fewer if it is shorter)."""
    resp = _client().get_object(
        Bucket=get_settings().s3_bucket, Key=key, Range=f"bytes=0-{n - 1}"
    )
    return resp["Body"].read()


def delete_object(key: str) -> None:
    _client().delete_object(Bucket=get_settings().s3_bucket, Key=key)
