from fastapi import Depends
from sqlalchemy import select
from sqlalchemy.orm import Session
from app.auth import current_user_id
from app.db import get_db
from app.models import User


def get_current_user(
    db: Session = Depends(get_db),
    auth_id: str = Depends(current_user_id),
) -> User:
    """Lazily create the local user row keyed by the Supabase auth id."""
    user = db.scalar(select(User).where(User.external_auth_id == auth_id))
    if user is None:
        user = User(external_auth_id=auth_id)
        db.add(user)
        db.commit()
        db.refresh(user)
    return user
