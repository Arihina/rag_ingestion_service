from __future__ import annotations


"""Зависимости FastAPI.

require_api_key — точка расширения под Keycloak: при переезде меняется
тело функции, не сигнатуры ручек.
"""

import secrets

from fastapi import Header, HTTPException

from app.config import settings


async def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    if not settings.api_key:
        return

    if not x_api_key or not secrets.compare_digest(x_api_key, settings.api_key):
        raise HTTPException(
            status_code=401, detail="Неверный или отсутствующий ключ")
