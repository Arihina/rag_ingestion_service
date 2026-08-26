from __future__ import annotations


import logging

from fastapi import APIRouter
from pydantic import BaseModel

from app.config import settings
from app.state import state

logger = logging.getLogger(__name__)

router = APIRouter(tags=["service"])


class HealthResponse(BaseModel):
    status: str
    docling: bool
    opensearch: bool
    model: str
    pending_jobs: int


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    docling_ok = await state.chunker.health()
    try:
        opensearch_ok = await state.os_client.ping()
    except Exception:
        logger.warning("opensearch недоступен", exc_info=True)
        opensearch_ok = False

    return HealthResponse(
        status="ok" if (docling_ok and opensearch_ok) else "degraded",
        docling=docling_ok,
        opensearch=opensearch_ok,
        model=settings.embed_model,
        pending_jobs=state.jobs.pending_count(),
    )
