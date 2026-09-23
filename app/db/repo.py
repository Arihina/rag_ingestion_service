from __future__ import annotations

"""Запросы к control plane."""

import uuid
from dataclasses import dataclass
from datetime import datetime, timezone

from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Document, ImportBatch, IngestJob, RagSet


@dataclass(frozen=True)
class DocumentCounts:
    total: int = 0
    ready: int = 0
    failed: int = 0
    pending: int = 0
    chunks: int = 0

    @property
    def has_pending(self) -> bool:
        return self.pending > 0

    @property
    def status(self) -> str:
        if self.total == 0:
            return "empty"
        if self.ready > 0:
            return "ready"
        if self.pending > 0:
            return "ingesting"
        return "failed"


async def counts_for(session: AsyncSession, rag_id: uuid.UUID) -> DocumentCounts:
    row = (
        await session.execute(
            select(
                func.count().label("total"),
                func.count().filter(Document.status == "success").label("ready"),
                func.count().filter(Document.status == "failed").label("failed"),
                func.count()
                .filter(Document.status.in_(("pending", "processing")))
                .label("pending"),
                func.coalesce(func.sum(Document.chunks_count),
                              0).label("chunks"),
            ).where(Document.rag_id == rag_id)
        )
    ).one()
    return DocumentCounts(*row)


async def counts_for_many(
    session: AsyncSession, rag_ids: list[uuid.UUID]
) -> dict[uuid.UUID, DocumentCounts]:
    """Один запрос на список наборов — иначе N+1 на каждой отрисовке списка."""
    if not rag_ids:
        return {}
    rows = await session.execute(
        select(
            Document.rag_id,
            func.count().label("total"),
            func.count().filter(Document.status == "success").label("ready"),
            func.count().filter(Document.status == "failed").label("failed"),
            func.count().filter(Document.status.in_(("pending", "processing"))).label("pending"),
            func.coalesce(func.sum(Document.chunks_count), 0).label("chunks"),
        )
        .where(Document.rag_id.in_(rag_ids))
        .group_by(Document.rag_id)
    )
    found = {r[0]: DocumentCounts(*r[1:]) for r in rows}
    return {rid: found.get(rid, DocumentCounts()) for rid in rag_ids}


async def get_rag(
    session: AsyncSession, rag_id: uuid.UUID, owner_id: uuid.UUID
) -> RagSet | None:
    return (
        await session.execute(
            select(RagSet).where(
                RagSet.id == rag_id,
                RagSet.owner_id == owner_id,
                RagSet.deleted_at.is_(None),
            )
        )
    ).scalar_one_or_none()


async def list_rags(session: AsyncSession, owner_id: uuid.UUID) -> list[RagSet]:
    return list(
        (
            await session.execute(
                select(RagSet)
                .where(RagSet.owner_id == owner_id, RagSet.deleted_at.is_(None))
                .order_by(RagSet.updated_at.desc())
            )
        ).scalars()
    )


async def count_rags(session: AsyncSession, owner_id: uuid.UUID) -> int:
    return (
        await session.execute(
            select(func.count())
            .select_from(RagSet)
            .where(RagSet.owner_id == owner_id, RagSet.deleted_at.is_(None))
        )
    ).scalar_one()


async def soft_delete_rag(session: AsyncSession, rag_id: uuid.UUID) -> None:
    await session.execute(
        update(RagSet)
        .where(RagSet.id == rag_id)
        .values(deleted_at=datetime.now(timezone.utc))
    )


async def get_document(
    session: AsyncSession, rag_id: uuid.UUID, document_id: uuid.UUID
) -> Document | None:
    return (
        await session.execute(
            select(Document).where(Document.id ==
                                   document_id, Document.rag_id == rag_id)
        )
    ).scalar_one_or_none()


async def find_by_hash(
    session: AsyncSession, rag_id: uuid.UUID, content_hash: str
) -> Document | None:
    return (
        await session.execute(
            select(Document).where(
                Document.rag_id == rag_id, Document.content_hash == content_hash
            )
        )
    ).scalar_one_or_none()


async def list_documents(session: AsyncSession, rag_id: uuid.UUID) -> list[Document]:
    return list(
        (
            await session.execute(
                select(Document)
                .where(Document.rag_id == rag_id)
                .order_by(Document.created_at.desc())
            )
        ).scalars()
    )


async def total_bytes(session: AsyncSession, rag_id: uuid.UUID) -> int:
    return (
        await session.execute(
            select(func.coalesce(func.sum(Document.size_bytes), 0)).where(
                Document.rag_id == rag_id
            )
        )
    ).scalar_one()


async def mark_document(
    session: AsyncSession,
    document_id: uuid.UUID,
    status: str,
    *,
    error: str | None = None,
    chunks_count: int | None = None,
) -> None:
    values: dict = {"status": status, "error": error}
    if chunks_count is not None:
        values["chunks_count"] = chunks_count
    await session.execute(
        update(Document).where(Document.id == document_id).values(**values)
    )


async def get_batch(
    session: AsyncSession, batch_id: uuid.UUID, rag_id: uuid.UUID
) -> ImportBatch | None:
    return (
        await session.execute(
            select(ImportBatch).where(
                ImportBatch.id == batch_id, ImportBatch.rag_id == rag_id
            )
        )
    ).scalar_one_or_none()


async def finish_batch(
    session: AsyncSession,
    batch_id: uuid.UUID,
    *,
    status: str,
    created: int = 0,
    skipped: int = 0,
    total: int = 0,
    error: str | None = None,
) -> None:
    await session.execute(
        update(ImportBatch)
        .where(ImportBatch.id == batch_id)
        .values(
            status=status,
            documents_created=created,
            documents_skipped=skipped,
            documents_total=total,
            error=error,
            finished_at=datetime.now(timezone.utc),
        )
    )


async def record_job(
    session: AsyncSession, document_id: uuid.UUID, rq_job_id: str
) -> IngestJob:
    job = IngestJob(document_id=document_id, rq_job_id=rq_job_id)
    session.add(job)
    return job


async def documents_by_ids(
    session: AsyncSession, document_ids: list[uuid.UUID], owner_id: uuid.UUID
) -> list[Document]:
    """Документы владельца по списку идентификаторов."""
    if not document_ids:
        return []
    return list(
        (
            await session.execute(
                select(Document)
                .join(RagSet, RagSet.id == Document.rag_id)
                .where(
                    Document.id.in_(set(document_ids)),
                    RagSet.owner_id == owner_id,
                    RagSet.deleted_at.is_(None),
                )
            )
        ).scalars()
    )


async def existing_hashes(session: AsyncSession, rag_id: uuid.UUID) -> set[str]:
    """Хэши документов, уже лежащих в наборе."""
    return set(
        (
            await session.execute(
                select(Document.content_hash).where(Document.rag_id == rag_id)
            )
        ).scalars()
    )
