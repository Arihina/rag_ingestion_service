from __future__ import annotations

"""Операторские ручки. Под /admin, в пользовательский контракт не входят."""

import uuid

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.db import ImportBatch, RagSet, get_session
from app.schemas.rags import BatchOut
from app.storage import ObjectStorage
from app.tasks.conn import maintenance_queue
from app.tasks.jobs import import_s3_prefix

router = APIRouter(prefix="/admin", tags=["admin"])


class S3ImportRequest(BaseModel):
    """Префикс в staging-бакете, не во внешнем хранилище."""

    prefix: str = Field(min_length=1, max_length=512)


@router.post(
    "/rags/{rag_id}/imports/s3",
    response_model=BatchOut,
    status_code=status.HTTP_202_ACCEPTED,
)
async def import_from_staging(
    rag_id: uuid.UUID,
    payload: S3ImportRequest,
    session: AsyncSession = Depends(get_session),
) -> ImportBatch:
    rag = await session.get(RagSet, rag_id)
    if rag is None or rag.deleted_at is not None:
        raise HTTPException(status_code=404, detail="Набор не найден")

    batch = ImportBatch(rag_id=rag_id, kind="s3")
    session.add(batch)
    await session.commit()
    await session.refresh(batch)

    maintenance_queue().enqueue(import_s3_prefix, str(batch.id), payload.prefix)
    return batch


@router.get("/staging")
async def list_staging(prefix: str = "") -> dict:
    objects = ObjectStorage().list_prefix(settings.s3_bucket_staging, prefix)
    return {
        "count": len(objects),
        "objects": [{"key": o["Key"], "size": o["Size"]} for o in objects[:500]],
    }
