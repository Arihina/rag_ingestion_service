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
from app.deps import require_internal_service
from app.schemas.rags import (
    DocumentEntry,
    DocumentLookupIn,
    DocumentLookupOut,
    InternalRagOut,
)

router = APIRouter(
    prefix="/v1/internal", tags=["internal"],
    dependencies=[Depends(require_internal_service)],
)


@router.get("/rags/{rag_id}", response_model=InternalRagOut)
async def internal_rag(
    rag_id: uuid.UUID,
    user_id: uuid.UUID = Query(...),
    session: AsyncSession = Depends(get_session),
) -> InternalRagOut:
    """Конфиг набора плюс проверка доступа одним запросом.

    Чтение допускается для владельца и для shared/public наборов (общий
    технический корпус). Запись этим вызовом не даётся.
    """
    rag = await repo.get_rag_for_read(session, rag_id, user_id)
    if rag is None:
        raise HTTPException(status_code=404, detail="Набор не найден")

    counts = await repo.counts_for(session, rag_id)
    return InternalRagOut(
        id=rag.id,
        name=rag.name,
        status=counts.status,
        kind=rag.kind,
        visibility=rag.visibility,
        index_name=rag.index_name,
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
    documents = await repo.documents_by_ids_for_read(
        session, payload.document_ids, user_id
    )
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
