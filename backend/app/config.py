from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str
    redis_url: str

    s3_endpoint_url: str
    s3_region: str = "us-east-1"
    s3_access_key: str
    s3_secret_key: str
    s3_bucket: str
    s3_public_url: str = ""

    auth_enabled: bool = False
    supabase_jwks_url: str = ""
    supabase_jwt_aud: str = "authenticated"

    frontend_origin: str = "http://localhost:3000"
    pipeline_version: str = "v0-stub"


@lru_cache
def get_settings() -> Settings:
    return Settings()
