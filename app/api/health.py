from __future__ import annotations

"""Проверка доступности зависимостей."""

import logging

from fastapi import APIRouter
from pydantic import BaseModel
from sqlalchemy import text

from app.config import settings
from app.db import sessionmaker
from app.state import state
from app.tasks.conn import redis_conn

logger = logging.getLogger(__name__)

router = APIRouter(tags=["service"])


class HealthResponse(BaseModel):
    status: str
    docling: bool
    opensearch: bool
    postgres: bool
    redis: bool
    storage: bool
    model: str


async def _docling_ok() -> bool:
    import httpx

    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{settings.docling_url}/health")
            return response.status_code == 200
    except Exception:
        return False


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    docling_ok = await _docling_ok()

    try:
        opensearch_ok = await state.os_client.ping()
    except Exception:
        logger.warning("opensearch недоступен", exc_info=True)
        opensearch_ok = False

    try:
        async with sessionmaker()() as session:
            await session.execute(text("SELECT 1"))
        postgres_ok = True
    except Exception:
        logger.warning("postgres недоступен", exc_info=True)
        postgres_ok = False

    try:
        redis_ok = bool(redis_conn().ping())
    except Exception:
        logger.warning("redis недоступен", exc_info=True)
        redis_ok = False

    storage_ok = state.storage.healthy()

    checks = (docling_ok, opensearch_ok, postgres_ok, redis_ok, storage_ok)
    return HealthResponse(
        status="ok" if all(checks) else "degraded",
        docling=docling_ok,
        opensearch=opensearch_ok,
        postgres=postgres_ok,
        redis=redis_ok,
        storage=storage_ok,
        model=settings.embed_model,
    )
