from functools import lru_cache

import httpx
from fastapi import Depends, Header, HTTPException
from jose import jwt

from app.config import Settings, get_settings

DEV_USER = "dev-local-user"


@lru_cache
def _jwks(url: str) -> dict:
    return httpx.get(url, timeout=10).json()


def current_user_id(
    authorization: str | None = Header(default=None),
    settings: Settings = Depends(get_settings),
) -> str:
    if not settings.auth_enabled:
        return DEV_USER

    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Missing bearer token")
    token = authorization.split(" ", 1)[1]

    try:
        claims = jwt.decode(
            token,
            _jwks(settings.supabase_jwks_url),
            audience=settings.supabase_jwt_aud,
            options={"verify_at_hash": False},
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=401, detail="Invalid token") from exc

    sub = claims.get("sub")
    if not sub:
        raise HTTPException(status_code=401, detail="Token missing sub")
    return sub
