from __future__ import annotations

"""Внутренние ручки для agentic_rag. Через мастер НЕ проксируется.

Путь намеренно под /v1/internal, а не /v1/platform: catch-all мастера
форвардит любой путь под v1/, поэтому наружу её закрывает сеть, а не
маршрутизация. Отдельный порт или сетевая политика обязательны.
"""

import uuid

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import get_session, repo
from app.schemas.rags import (
    DocumentEntry,
    DocumentLookupIn,
    DocumentLookupOut,
    InternalRagOut,
)

router = APIRouter(
    prefix="/v1/internal", tags=["internal"]
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


@router.post("/documents/lookup", response_model=DocumentLookupOut)
async def lookup_documents(
    payload: DocumentLookupIn,
    user_id: uuid.UUID = Query(...),
    session: AsyncSession = Depends(get_session),
) -> DocumentLookupOut:
    """Имена документов по идентификаторам — для подписей под цитатами."""
    documents = await repo.documents_by_ids(session, payload.document_ids, user_id)
    return DocumentLookupOut(
        documents=[
            DocumentEntry(
                document_id=document.id,
                filename=document.filename,
                rag_id=document.rag_id,
            )
            for document in documents
        ]
    )
