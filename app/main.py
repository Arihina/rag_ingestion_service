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


app = FastAPI(title="RAG Ingest", version="2.0.0", lifespan=lifespan)
app.include_router(health.router)
app.include_router(rags.router)

internal_app = FastAPI(
    title="RAG Ingest (internal)", version="2.0.0", lifespan=internal_lifespan
)
internal_app.include_router(health.router)
internal_app.include_router(embed.router)
internal_app.include_router(internal.router)
internal_app.include_router(admin.router)

if settings.internal_port == 0:
    app.include_router(embed.router)
    app.include_router(internal.router)
    app.include_router(admin.router)


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
        logger.warning(
            "internal_port=0 — служебные ручки открыты на платформенном порту"
        )
        await public.serve()
        return

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
