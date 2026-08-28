from __future__ import annotations

"""Зависимости FastAPI.
require_api_key и current_user — точки расширения под Keycloak: при переезде
меняются тела функций, не сигнатуры ручек.
"""

import secrets
import uuid

from fastapi import Header, HTTPException

from app.config import settings


async def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    if not settings.api_key:
        return
    if not x_api_key or not secrets.compare_digest(x_api_key, settings.api_key):
        raise HTTPException(
            status_code=401, detail="Неверный или отсутствующий ключ")


async def current_user(x_user_id: str | None = Header(default=None)) -> uuid.UUID:
    """Пока — доверенный заголовок от мастера."""
    if not x_user_id:
        raise HTTPException(status_code=401, detail="Требуется X-User-Id")
    try:
        return uuid.UUID(x_user_id)
    except ValueError:
        raise HTTPException(
            status_code=401, detail="X-User-Id должен быть UUID")
