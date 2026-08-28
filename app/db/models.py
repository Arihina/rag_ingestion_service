from __future__ import annotations

"""Схема control plane.
Связь с SeaweedFS —
только через storage_key и icon_key.
"""

import uuid
from datetime import datetime

from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    Float,
    String,
    Text,
    UniqueConstraint,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import UUID as PgUUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base


def _uuid_pk() -> Mapped[uuid.UUID]:
    return mapped_column(PgUUID(as_uuid=True), primary_key=True, default=uuid.uuid4)


class RagSet(Base):
    __tablename__ = "rag_sets"

    id: Mapped[uuid.UUID] = _uuid_pk()
    owner_id: Mapped[uuid.UUID] = mapped_column(
        PgUUID(as_uuid=True), nullable=False)

    name: Mapped[str] = mapped_column(String(200), nullable=False)
    description: Mapped[str | None] = mapped_column(Text)

    prompt: Mapped[str | None] = mapped_column(Text)
    temperature: Mapped[float] = mapped_column(
        Float, nullable=False, server_default=text("0.3"))
    top_k: Mapped[int] = mapped_column(
        Integer, nullable=False, server_default=text("5"))
    score_threshold: Mapped[float] = mapped_column(
        Float, nullable=False, server_default=text("0.0"))

    icon_key: Mapped[str | None] = mapped_column(String(512))

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )
    deleted_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True))

    documents: Mapped[list["Document"]] = relationship(
        back_populates="rag_set", cascade="all, delete-orphan"
    )

    __table_args__ = (
        Index("ix_rag_sets_owner_alive", "owner_id",
              postgresql_where=text("deleted_at IS NULL")),
        Index(
            "uq_rag_sets_owner_name",
            "owner_id",
            func.lower(name),
            unique=True,
            postgresql_where=text("deleted_at IS NULL"),
        ),
        CheckConstraint("temperature >= 0 AND temperature <= 1",
                        name="ck_rag_temperature"),
        CheckConstraint("top_k >= 1 AND top_k <= 10", name="ck_rag_top_k"),
        CheckConstraint(
            "score_threshold >= 0 AND score_threshold <= 1", name="ck_rag_score_threshold"
        ),
    )


class ImportBatch(Base):
    """Одна операция пользователя, разворачивающаяся в N документов.
    Нужна и для архива, и для импорта.
    """

    __tablename__ = "import_batches"

    id: Mapped[uuid.UUID] = _uuid_pk()
    rag_id: Mapped[uuid.UUID] = mapped_column(
        PgUUID(as_uuid=True), ForeignKey("rag_sets.id", ondelete="CASCADE"), nullable=False
    )
    kind: Mapped[str] = mapped_column(
        String(16), nullable=False)  # archive | s3
    status: Mapped[str] = mapped_column(
        String(16), nullable=False, server_default=text("'pending'"))

    documents_total: Mapped[int] = mapped_column(
        Integer, nullable=False, server_default=text("0"))
    documents_created: Mapped[int] = mapped_column(
        Integer, nullable=False, server_default=text("0"))
    documents_skipped: Mapped[int] = mapped_column(
        Integer, nullable=False, server_default=text("0"))
    error: Mapped[str | None] = mapped_column(Text)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now())
    finished_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True))

    __table_args__ = (
        Index("ix_import_batches_rag", "rag_id"),
        CheckConstraint("kind IN ('archive','s3')", name="ck_batch_kind"),
    )


class Document(Base):
    __tablename__ = "documents"

    id: Mapped[uuid.UUID] = _uuid_pk()
    rag_id: Mapped[uuid.UUID] = mapped_column(
        PgUUID(as_uuid=True), ForeignKey("rag_sets.id", ondelete="CASCADE"), nullable=False
    )

    filename: Mapped[str] = mapped_column(String(512), nullable=False)
    size_bytes: Mapped[int] = mapped_column(BigInteger, nullable=False)
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    storage_key: Mapped[str] = mapped_column(String(1024), nullable=False)

    origin: Mapped[str] = mapped_column(
        String(16), nullable=False, server_default=text("'upload'"))
    origin_ref: Mapped[uuid.UUID | None] = mapped_column(
        PgUUID(as_uuid=True), ForeignKey(
            "import_batches.id", ondelete="SET NULL")
    )

    status: Mapped[str] = mapped_column(
        String(16), nullable=False, server_default=text("'pending'"))
    error: Mapped[str | None] = mapped_column(Text)
    chunks_count: Mapped[int] = mapped_column(
        Integer, nullable=False, server_default=text("0"))

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    rag_set: Mapped[RagSet] = relationship(back_populates="documents")

    __table_args__ = (
        # Дедупликация повторной загрузки
        UniqueConstraint("rag_id", "content_hash",
                         name="uq_documents_rag_hash"),
        Index("ix_documents_rag_status", "rag_id", "status"),
        CheckConstraint(
            "status IN ('pending','processing','success','failed')", name="ck_document_status"
        ),
        CheckConstraint("origin IN ('upload','archive','s3')",
                        name="ck_document_origin"),
    )


class IngestJob(Base):
    __tablename__ = "ingest_jobs"

    id: Mapped[uuid.UUID] = _uuid_pk()
    document_id: Mapped[uuid.UUID] = mapped_column(
        PgUUID(as_uuid=True), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False
    )
    rq_job_id: Mapped[str] = mapped_column(String(64), nullable=False)
    attempts: Mapped[int] = mapped_column(
        Integer, nullable=False, server_default=text("0"))

    started_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True))
    error: Mapped[str | None] = mapped_column(Text)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now())

    __table_args__ = (Index("ix_ingest_jobs_document", "document_id"),)
