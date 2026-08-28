from __future__ import annotations

"""Внутренняя ручка для agentic_rag. Через мастер НЕ проксируется."""

import uuid

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import get_session, repo
from app.deps import require_api_key
from app.schemas.rags import InternalRagOut

router = APIRouter(
    prefix="/v1/internal", tags=["internal"], dependencies=[Depends(require_api_key)]
)


@router.get("/rags/{rag_id}", response_model=InternalRagOut)
async def internal_rag(
    rag_id: uuid.UUID,
    user_id: uuid.UUID = Query(...),
    session: AsyncSession = Depends(get_session),
) -> InternalRagOut:
    """Конфиг набора плюс проверка владения одним запросом."""
    rag = await repo.get_rag(session, rag_id, user_id)
    if rag is None:
        raise HTTPException(status_code=404, detail="Набор не найден")

    counts = await repo.counts_for(session, rag_id)
    return InternalRagOut(
        id=rag.id,
        name=rag.name,
        status=counts.status,
        prompt=rag.prompt,
        temperature=rag.temperature,
        top_k=rag.top_k,
        score_threshold=rag.score_threshold,
    )
