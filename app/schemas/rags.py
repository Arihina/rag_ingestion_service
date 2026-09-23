from __future__ import annotations

"""Контракты платформенного API.

Потолок top_k — не вкусовщина: пул растёт как
top_k x 3 ветки x N вариантов multi-query x итерации, и он же ограничивает
размер контекста, уходящего в eval агентного цикла.
"""

import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.config import settings


class RagConfig(BaseModel):
    prompt: str | None = Field(
        default=None, max_length=settings.rag_prompt_max_chars)
    temperature: float = Field(
        default=settings.rag_default_temperature, ge=0.0, le=1.0)
    top_k: int = Field(default=settings.rag_default_top_k,
                       ge=1, le=settings.rag_top_k_max)
    score_threshold: float = Field(
        default=settings.rag_default_score_threshold, ge=0.0, le=1.0
    )


class RagCreate(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    description: str | None = Field(default=None, max_length=2000)
    prompt: str | None = Field(
        default=None, max_length=settings.rag_prompt_max_chars)
    temperature: float = Field(
        default=settings.rag_default_temperature, ge=0.0, le=1.0)
    top_k: int = Field(default=settings.rag_default_top_k,
                       ge=1, le=settings.rag_top_k_max)
    score_threshold: float = Field(
        default=settings.rag_default_score_threshold, ge=0.0, le=1.0
    )


class RagUpdate(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=200)
    description: str | None = Field(default=None, max_length=2000)
    prompt: str | None = Field(
        default=None, max_length=settings.rag_prompt_max_chars)
    temperature: float | None = Field(default=None, ge=0.0, le=1.0)
    top_k: int | None = Field(default=None, ge=1, le=settings.rag_top_k_max)
    score_threshold: float | None = Field(default=None, ge=0.0, le=1.0)


class DocumentCountsOut(BaseModel):
    total: int
    ready: int
    failed: int
    pending: int


class RagOut(BaseModel):
    id: uuid.UUID
    name: str
    description: str | None
    status: str
    has_icon: bool
    documents: DocumentCountsOut
    chunks_total: int
    has_pending: bool
    config: RagConfig
    created_at: datetime
    updated_at: datetime


class DocumentOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    filename: str
    size_bytes: int
    status: str
    error: str | None
    chunks_count: int
    origin: str
    created_at: datetime


class DocumentAccepted(BaseModel):
    id: uuid.UUID
    filename: str
    status: str
    duplicate_of: uuid.UUID | None = None


class UploadAccepted(BaseModel):
    documents: list[DocumentAccepted]


class BatchOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    kind: str
    status: str
    documents_total: int
    documents_created: int
    documents_skipped: int
    error: str | None
    created_at: datetime
    finished_at: datetime | None


class InternalRagOut(BaseModel):
    """Ответ agentic_rag: конфиг набора плюс подтверждение владения."""

    id: uuid.UUID
    name: str
    status: str
    prompt: str | None
    temperature: float
    top_k: int
    score_threshold: float


class DocumentLookupIn(BaseModel):
    """Batch-подстановка имён по идентификаторам документов."""

    model_config = ConfigDict(extra="forbid")

    document_ids: list[uuid.UUID] = Field(min_length=1, max_length=100)


class DocumentEntry(BaseModel):
    document_id: uuid.UUID
    filename: str
    rag_id: uuid.UUID


class DocumentLookupOut(BaseModel):
    documents: list[DocumentEntry]


class RagFromBucketIn(RagCreate):
    """Создание набора из уже лежащих в хранилище файлов."""

    model_config = ConfigDict(extra="forbid")

    bucket: str = Field(min_length=1, max_length=255)
    prefix: str = Field(default="", max_length=1024)
    recursive: bool = True
    extensions: list[str] = Field(default_factory=list, max_length=32)


class AcceptedDocument(BaseModel):
    document_id: uuid.UUID
    source_key: str


class RejectedDocument(BaseModel):
    source_key: str
    reason: str


class FromBucketOut(BaseModel):
    rag_id: uuid.UUID
    name: str
    status: str
    accepted_documents: list[AcceptedDocument]
    rejected_documents: list[RejectedDocument]


class DocumentsFromBucketIn(BaseModel):
    """Добавление файлов из хранилища в существующий набор.

    Два взаимоисключающих режима:

      * keys — явный список объектов. Метаданные берутся HEAD-запросом на
        каждый, потому что листинг для произвольных ключей бесполезен.
      * prefix — пересканировать префикс. Осмысленно именно здесь:
        дедупликация идёт против уже лежащих в наборе документов, так что
        повторный вызов заберёт только новое.

    Ровно один из них, иначе непонятно, что делать с пересечением.
    """

    model_config = ConfigDict(extra="forbid")

    bucket: str = Field(min_length=1, max_length=255)
    keys: list[str] = Field(default_factory=list, max_length=500)
    prefix: str | None = Field(default=None, max_length=1024)
    recursive: bool = True
    extensions: list[str] = Field(default_factory=list, max_length=32)

    @model_validator(mode="after")
    def _exactly_one_mode(self) -> "DocumentsFromBucketIn":
        if bool(self.keys) == (self.prefix is not None):
            raise ValueError("нужен ровно один из keys или prefix")
        return self


class DocumentsFromBucketOut(BaseModel):
    rag_id: uuid.UUID
    status: str
    accepted_documents: list[AcceptedDocument]
    rejected_documents: list[RejectedDocument]
