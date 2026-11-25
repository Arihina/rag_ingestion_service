from __future__ import annotations

"""Внутренняя аутентификация master -> агент (этап 1, статический токен).

Проверка выполняется middleware, а не dependency на каждой ручке: так новый
эндпоинт закрыт по умолчанию, а не «пока не вспомнили проверить».

Доверие — к credential, а не к адресу: ``X-Internal-Token`` доказывает, что
запрос пришёл от master. ``X-User-Id`` после этого считается UUID пользователя
(``sub`` проверенного JWT), и скоупинг данных по нему остаётся единственной
авторизационной проверкой на стороне агента. Роли здесь не проверяются — это
политика master.

``X-Request-Id`` — служебный заголовок корреляции логов, а не аутентификация:
принимается от master/nginx только в допустимом виде, иначе генерируется.
"""

import json
import logging
import re
import secrets
import time
import uuid

from starlette.datastructures import Headers, MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from app.config import settings

logger = logging.getLogger(__name__)

INTERNAL_TOKEN_HEADER = "X-Internal-Token"
USER_ID_HEADER = "X-User-Id"
REQUEST_ID_HEADER = "X-Request-Id"

# Тот же контракт, что у master: только печатные ASCII без пробелов/CRLF.
_REQUEST_ID_RE = re.compile(r"^[A-Za-z0-9._:-]{1,128}$")

AUTHENTICATION_ERROR = "authentication_error"


def normalize_request_id(value: str | None) -> str:
    """Принять валидный X-Request-Id или сгенерировать новый.

    Некорректный id не повод отдавать 400: это диагностический заголовок.
    Но пропускать его в логи как есть нельзя (защита от log-injection).
    """
    if value and _REQUEST_ID_RE.match(value):
        return value
    return uuid.uuid4().hex


def _auth_error(message: str) -> dict:
    return {
        "error": {
            "message": message,
            "type": AUTHENTICATION_ERROR,
            "param": None,
            "code": None,
        }
    }


def _token_matches(present: str | None, expected: str | None) -> bool:
    """Постоянное по времени сравнение; пустой ожидаемый секрет — всегда нет."""
    if not present or not expected:
        return False
    return secrets.compare_digest(present, expected)


def _safe_log_value(value: object, limit: int = 512) -> str:
    text = str(value)[:limit]
    return "".join(ch if ch.isprintable() else "?" for ch in text)


class InternalAuthMiddleware:
    """Проверка X-Internal-Token и X-User-Id на платформенном порту.

    Чистый ASGI-middleware (не BaseHTTPMiddleware): тело не буферизуется,
    multipart-загрузки идут потоком. Кладёт проверенный ``user_id`` в
    ``request.state`` — бизнес-код читает только его.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        path = scope.get("path", "/")
        if path in settings.public_paths:
            await self.app(scope, receive, send)
            return

        mode = settings.internal_auth_mode
        headers = Headers(scope=scope)

        if mode == "token":
            token = headers.get(INTERNAL_TOKEN_HEADER)
            if not _token_matches(token, settings.master_internal_token):
                await self._reject(send, "Требуется внутренний токен master")
                return

        # user_id валидируем во всех режимах, кроме пропущенного пути выше.
        # В off-режиме (dev) отсутствие заголовка не блокирует запрос здесь —
        # его дожмёт current_user на бизнес-ручке, а /docs останется доступен.
        raw = headers.get(USER_ID_HEADER)
        parsed: uuid.UUID | None = None
        if raw:
            try:
                parsed = uuid.UUID(raw)
            except ValueError:
                parsed = None

        if mode == "token" and parsed is None:
            await self._reject(send, "X-User-Id должен быть UUID")
            return

        if parsed is not None:
            scope.setdefault("state", {})["user_id"] = parsed

        await self.app(scope, receive, send)

    async def _reject(self, send: Send, message: str) -> None:
        body = json.dumps(_auth_error(message), ensure_ascii=False).encode("utf-8")
        await send({
            "type": "http.response.start",
            "status": 401,
            "headers": [(b"content-type", b"application/json")],
        })
        await send({"type": "http.response.body", "body": body})


class RequestContextMiddleware:
    """request_id + access-лог. Должен быть самым внешним middleware.

    Возвращает X-Request-Id в заголовке ответа и пишет структурированную
    access-строку (request_id, agent_id, method, path, status, user_id,
    duration_ms). Токены и Authorization в лог не пишутся.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        incoming = Headers(scope=scope).get(REQUEST_ID_HEADER)
        request_id = normalize_request_id(incoming)
        scope.setdefault("state", {})["request_id"] = request_id

        status = 500
        started = time.perf_counter()

        async def send_with_request_id(message: Message) -> None:
            nonlocal status
            if message["type"] == "http.response.start":
                status = int(message["status"])
                MutableHeaders(scope=message)[REQUEST_ID_HEADER] = request_id
            await send(message)

        try:
            await self.app(scope, receive, send_with_request_id)
        finally:
            duration_ms = (time.perf_counter() - started) * 1000
            user_id = scope.get("state", {}).get("user_id")
            logger.info(
                "request_id=%s agent_id=%s method=%s path=%s status=%d "
                "user_id=%s duration_ms=%.1f",
                request_id,
                settings.agent_id,
                _safe_log_value(scope.get("method", "")),
                _safe_log_value(scope.get("path", "")),
                status,
                user_id or "-",
                duration_ms,
            )
