from __future__ import annotations

"""Зависимости FastAPI.
current_user — точка расширения под Keycloak: при переезде меняется тело
функции, не сигнатуры ручек.
"""

import uuid

from fastapi import Header, HTTPException


async def current_user(x_user_id: str | None = Header(default=None)) -> uuid.UUID:
    """Пока — доверенный заголовок от мастера.

    Осознанный временный компромисс: запрос с произвольным X-User-Id
    обходит скоупинг целиком, поэтому платформенный порт доступен только
    мастеру, а внутренний — только сервисам платформы.
    """
    if not x_user_id:
        raise HTTPException(status_code=401, detail="Требуется X-User-Id")
    try:
        return uuid.UUID(x_user_id)
    except ValueError:
        raise HTTPException(
            status_code=401, detail="X-User-Id должен быть UUID")
