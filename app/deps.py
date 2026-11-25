from __future__ import annotations

"""Зависимости FastAPI.

current_user — единая точка получения пользователя для бизнес-кода. Он читает
``request.state.user_id``, который положила middleware ``platform_security``
после проверки ``X-Internal-Token`` (master) и валидации ``X-User-Id`` как
UUID. Бизнес-код не должен читать заголовки напрямую — иначе на новой ручке
легко забыть проверку.

Внутренний порт 8012 закрыт двумя раздельными токенами:
- X-Internal-Admin-Token   → /admin/* (оператор);
- X-Internal-Service-Token → /embed, /v1/internal/* (сервисы/воркер).
"""

import secrets
import uuid

from fastapi import Header, HTTPException, Request

from app.config import settings


async def current_user(request: Request) -> uuid.UUID:
    """UUID пользователя, установленный middleware после проверки master.

    Прямое доверие заголовку ``X-User-Id`` убрано: теперь это значение
    появляется в ``request.state`` только после валидного ``X-Internal-Token``.
    """
    user_id = getattr(request.state, "user_id", None)
    if user_id is None:
        raise HTTPException(status_code=401, detail="Требуется X-User-Id")
    return user_id


def _require_token(present: str | None, expected: str | None, name: str) -> None:
    """Fail-closed: токен не настроен — 503, отсутствует/неверен — 401."""
    if not expected:
        raise HTTPException(503, f"{name} не настроен на сервере")
    if not present or not secrets.compare_digest(present, expected):
        raise HTTPException(401, f"Требуется {name}")


async def require_internal_admin(
    x_internal_admin_token: str | None = Header(default=None),
) -> None:
    """Операторский доступ к /admin/*."""
    _require_token(
        x_internal_admin_token, settings.internal_admin_token,
        "X-Internal-Admin-Token",
    )


async def require_internal_service(
    x_internal_service_token: str | None = Header(default=None),
) -> None:
    """Сервисный доступ к /embed и /v1/internal/*."""
    _require_token(
        x_internal_service_token, settings.internal_service_token,
        "X-Internal-Service-Token",
    )
