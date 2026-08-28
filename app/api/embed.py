from __future__ import annotations

"""Векторизация тем же семейством весов, что обрабатывало документы.

Экземпляра модели два, и выбирается он полем pool. Веса и версия совпадают,
поэтому вектор запроса физически не может разойтись с вектором документа —
ради этого /embed и существует. Разделение нужно для другого: заливка
корпуса это сотни батчей подряд, и на общем семафоре чат ждал бы не свои
30 мс, а глубину очереди, умноженную на время батча.
"""

import asyncio
from typing import Literal

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from app.config import settings
from app.deps import require_api_key
from app.state import state

router = APIRouter(tags=["embed"])


class EmbedRequest(BaseModel):
    """texts — список: все варианты multi-query уходят ОДНИМ вызовом."""

    texts: list[str] = Field(min_length=1, max_length=256)
    pool: Literal["query", "ingest"] = "query"


class EmbedItem(BaseModel):
    dense: list[float]
    sparse: dict[str, float]


class EmbedResponse(BaseModel):
    model: str
    embeddings: list[EmbedItem]


@router.post("/embed", response_model=EmbedResponse, dependencies=[Depends(require_api_key)])
async def embed(request: EmbedRequest) -> EmbedResponse:
    if request.pool == "ingest":
        embedder, semaphore = state.embedder, state.embed_sem
    else:
        embedder, semaphore = state.query_embedder, state.query_sem

    async with semaphore:
        results = await asyncio.to_thread(embedder.embed, request.texts)

    return EmbedResponse(
        model=settings.embed_model,
        embeddings=[EmbedItem(dense=r.dense, sparse=r.sparse)
                    for r in results],
    )
