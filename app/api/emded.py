from __future__ import annotations


import asyncio

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from app.config import settings
from app.deps import require_api_key
from app.state import state

router = APIRouter(tags=["embed"])


class EmbedRequest(BaseModel):
    """texts — список: все варианты multi-query уходят ОДНИМ вызовом.

    Поштучные обращения на каждый вариант превратились бы в N сетевых
    round-trip'ов, каждый за семафором GPU.
    """

    texts: list[str] = Field(min_length=1, max_length=256)


class EmbedItem(BaseModel):
    dense: list[float]
    sparse: dict[str, float]


class EmbedResponse(BaseModel):
    model: str
    embeddings: list[EmbedItem]


@router.post(
    "/embed", response_model=EmbedResponse, dependencies=[Depends(require_api_key)]
)
async def embed(request: EmbedRequest) -> EmbedResponse:
    async with state.query_sem:
        results = await asyncio.to_thread(
            state.query_embedder.embed, request.texts)

    return EmbedResponse(
        model=settings.embed_model,
        embeddings=[EmbedItem(dense=r.dense, sparse=r.sparse)
                    for r in results],
    )
