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


def artifact_key(video_id: str, pipeline_version: str, name: str) -> str:
    return f"videos/{video_id}/{pipeline_version}/{name}"


def presigned_put(key: str, content_type: str, expires: int = 3600) -> str:
    return _client().generate_presigned_url(
        "put_object",
        Params={"Bucket": get_settings().s3_bucket, "Key": key, "ContentType": content_type},
        ExpiresIn=expires,
    )


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
