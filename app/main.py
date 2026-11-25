from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from opensearchpy import AsyncOpenSearch
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.api import admin, embed, health, internal, rags
from app.config import settings
from app.db import dispose as dispose_db
from app.embeddings import BgeM3Embedder
from app.platform_security import InternalAuthMiddleware, RequestContextMiddleware
from app.search import OpenSearchLoader
from app.state import state
from app.storage import ObjectStorage

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    logger.info("Загрузка %s (индексация)...", settings.embed_model)
    state.embedder = BgeM3Embedder(
        settings.embed_model,
        batch_size=settings.embed_batch_size,
        max_length=settings.embed_max_length,
        device=settings.embed_device,
    )

    if settings.separate_query_embedder:
        logger.info("Загрузка %s (запросы)...", settings.embed_model)
        state.query_embedder = BgeM3Embedder(
            settings.embed_model,
            batch_size=settings.embed_query_batch_size,
            max_length=settings.embed_max_length,
            device=settings.embed_device,
        )
    else:
        state.query_embedder = state.embedder

    state.embed_sem = asyncio.Semaphore(settings.embed_concurrency)
    state.query_sem = asyncio.Semaphore(settings.embed_query_concurrency)

    auth = (
        (settings.opensearch_user, settings.opensearch_password)
        if settings.opensearch_user
        else None
    )
    state.os_client = AsyncOpenSearch(
        hosts=[settings.opensearch_url],
        http_auth=auth,
        verify_certs=bool(auth),
        timeout=settings.opensearch_timeout,
    )
    state.loader = OpenSearchLoader(state.os_client, settings.index_name)
    await state.loader.ensure_index()
    await state.loader.verify_cosine_calibration()

    state.storage = ObjectStorage()
    state.storage.ensure_buckets()

    state.ready.set()
    logger.info("Готов")
    yield

    await state.os_client.close()
    await dispose_db()


@asynccontextmanager
async def internal_lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Внутреннее приложение своих ресурсов не создаёт — только ждёт чужих.

    Оба сервера стартуют одновременно, и без этого ожидания /embed мог бы
    принять запрос до загрузки модели.
    """
    await state.ready.wait()
    yield


_docs_enabled = settings.env != "prod"

app = FastAPI(
    title="RAG Ingest",
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/docs" if _docs_enabled else None,
    redoc_url="/redoc" if _docs_enabled else None,
    openapi_url="/openapi.json" if _docs_enabled else None,
)
app.include_router(health.router)
app.include_router(rags.router)

# Внутренние ручки (/embed, /v1/internal, /admin) НЕ входят в платформенное
# приложение безусловно: платформенный порт проксирует master, и эти пути там
# не должны существовать вовсе (а не «закрыты сетью»). Документация на
# внутреннем порту отключена всегда — это карта служебных ручек.
internal_app = FastAPI(
    title="RAG Ingest (internal)",
    version="2.0.0",
    lifespan=internal_lifespan,
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
)
internal_app.include_router(health.router)
internal_app.include_router(health.ready_router)
internal_app.include_router(embed.router)
internal_app.include_router(internal.router)
internal_app.include_router(admin.router)

app.add_middleware(InternalAuthMiddleware)
app.add_middleware(RequestContextMiddleware)

internal_app.add_middleware(RequestContextMiddleware)


def _error_body(status_code: int, message: str, param: str | None = None) -> dict:
    types = {
        400: "invalid_request_error",
        401: "authentication_error",
        404: "not_found_error",
        409: "invalid_request_error",
        413: "invalid_request_error",
        415: "invalid_request_error",
    }
    return {
        "error": {
            "message": message,
            "type": types.get(status_code, "server_error"),
            "param": param,
            "code": None,
        }
    }


@app.exception_handler(StarletteHTTPException)
async def _http_error(request: Request, exc: StarletteHTTPException) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code, content=_error_body(
            exc.status_code, str(exc.detail))
    )


def _install_error_handlers(target: FastAPI) -> None:
    @target.exception_handler(StarletteHTTPException)
    async def _http_error(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content=_error_body(exc.status_code, str(exc.detail)),
        )

    @target.exception_handler(RequestValidationError)
    async def _validation_error(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        first = exc.errors()[0] if exc.errors() else {}
        loc = [
            str(part)
            for part in first.get("loc", ())
            if part not in ("body", "query", "path", "header")
        ]
        return JSONResponse(
            status_code=400,
            content=_error_body(
                400, first.get("msg", "Некорректный запрос"), ".".join(
                    loc) or None
            ),
        )

    @target.exception_handler(Exception)
    async def _unhandled_error(request: Request, exc: Exception) -> JSONResponse:
        logger.exception(
            "Необработанная ошибка на %s %s", request.method, request.url.path
        )
        return JSONResponse(
            status_code=500,
            content=_error_body(
                500, f"Внутренняя ошибка: {type(exc).__name__}"),
        )


_install_error_handlers(app)
_install_error_handlers(internal_app)


def _assert_internal_tokens() -> None:
    """Внутренний порт без токенов не поднимаем.

    Иначе /admin/* и /embed окажутся закрыты только сетью, а не проверкой.
    """
    missing = []
    if not settings.internal_admin_token:
        missing.append("INGEST_INTERNAL_ADMIN_TOKEN")
    if not settings.internal_service_token:
        missing.append("INGEST_INTERNAL_SERVICE_TOKEN")
    if missing:
        raise RuntimeError(
            "Внутренний порт включён, но не заданы токены: "
            + ", ".join(missing)
        )


def _assert_platform_auth() -> None:
    """Fail-closed для аутентификации платформенного порта.

    * mode=token и пустой master-токен — не стартуем (иначе все запросы
      молча отбивались бы 401, а «как раньше» работать не должны);
    * mode=off — допустим только вне prod, с явным предупреждением;
    * mode=assertion — этап 2 ещё не реализован, не даём запустить молча;
    * internal_port=0 в prod — не стартуем без явного флага.
    """
    mode = settings.internal_auth_mode

    if mode == "assertion":
        raise RuntimeError(
            "INGEST_INTERNAL_AUTH_MODE=assertion ещё не реализован (этап 2). "
            "Используйте 'token'."
        )

    if mode == "token" and not settings.master_internal_token:
        raise RuntimeError(
            "INGEST_INTERNAL_AUTH_MODE=token, но INGEST_MASTER_INTERNAL_TOKEN "
            "не задан. Платформенный порт без master-токена не стартует "
            "(fail-closed)."
        )

    if mode == "off":
        if settings.env == "prod":
            raise RuntimeError(
                "INGEST_INTERNAL_AUTH_MODE=off запрещён при INGEST_ENV=prod."
            )
        logger.warning(
            "Внутренняя аутентификация ВЫКЛЮЧЕНА (internal_auth_mode=off): "
            "платформенный порт доверяет X-User-Id без master-токена. "
            "Только для локальной разработки/тестов."
        )

    if settings.env == "prod" and settings.internal_port == 0:
        if not settings.allow_internal_on_public:
            raise RuntimeError(
                "INGEST_INTERNAL_PORT=0 запрещён при INGEST_ENV=prod: "
                "/embed и /v1/internal/* будет некому обслуживать. "
                "Задайте порт или, при явной необходимости, "
                "INGEST_ALLOW_INTERNAL_ON_PUBLIC=true."
            )
        logger.warning(
            "INGEST_INTERNAL_PORT=0 в prod разрешён флагом "
            "INGEST_ALLOW_INTERNAL_ON_PUBLIC=true — внутренние ручки "
            "недоступны."
        )


async def _serve() -> None:
    """Два сервера в одном процессе."""
    import uvicorn

    def server(target: FastAPI, host: str, port: int) -> "uvicorn.Server":
        return uvicorn.Server(
            uvicorn.Config(
                target,
                host=host,
                port=port,
                log_level=settings.log_level,
                timeout_keep_alive=settings.timeout_keep_alive,
            )
        )

    public = server(app, settings.host, settings.port)
    logger.info("Платформенный порт: %s:%s", settings.host, settings.port)

    if settings.internal_port == 0:
        logger.warning("internal_port=0 — внутренний порт не поднимается")
        await public.serve()
        return

    _assert_internal_tokens()

    private = server(internal_app, settings.internal_host,
                     settings.internal_port)
    logger.info("Внутренний порт: %s:%s",
                settings.internal_host, settings.internal_port)
    await asyncio.gather(public.serve(), private.serve())


def run() -> None:
    """Запуск сервиса: python -m app.main"""
    import uvicorn

    logging.basicConfig(
        level=settings.log_level.upper(),
        format="%(asctime)s %(levelname)-5s [%(name)s] %(message)s",
    )

    _assert_platform_auth()

    if settings.reload:
        logger.warning(
            "reload включён: порты НЕ разводятся, модели перезагружаются "
            "на каждую правку"
        )
        uvicorn.run(
            "app.main:app",
            host=settings.host,
            port=settings.port,
            reload=True,
            log_level=settings.log_level,
            timeout_keep_alive=settings.timeout_keep_alive,
        )
        return

    asyncio.run(_serve())


if __name__ == "__main__":
    run()
