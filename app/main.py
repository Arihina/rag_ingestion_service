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

    logger.info("Готов")
    yield

    await state.os_client.close()
    await dispose_db()


app = FastAPI(title="RAG Ingest API", version="2.0.0", lifespan=lifespan)

app.include_router(health.router)
app.include_router(embed.router)
app.include_router(rags.router)
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


@app.exception_handler(Exception)
async def _unhandled_error(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Необработанная ошибка на %s %s",
                     request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content=_error_body(500, f"Внутренняя ошибка: {type(exc).__name__}"),
    )


@app.exception_handler(RequestValidationError)
async def _validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
    first = exc.errors()[0] if exc.errors() else {}
    loc = [str(p) for p in first.get("loc", ())
           if p not in ("body", "query", "path", "header")]
    return JSONResponse(
        status_code=400,
        content=_error_body(400, first.get(
            "msg", "Некорректный запрос"), ".".join(loc) or None),
    )


def run() -> None:
    """Запуск сервиса: python -m app.main"""
    import uvicorn

    logging.basicConfig(
        level=settings.log_level.upper(),
        format="%(asctime)s %(levelname)-5s [%(name)s] %(message)s",
    )

    if settings.reload:
        logger.warning(
            "reload включён: модели будут перезагружаться на каждую правку")

    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.reload,
        log_level=settings.log_level,
        timeout_keep_alive=settings.timeout_keep_alive,
    )


if __name__ == "__main__":
    run()
