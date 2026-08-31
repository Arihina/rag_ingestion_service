from __future__ import annotations

"""Задачи RQ. ВОРКЕР НЕ ДЕРЖИТ МОДЕЛЬ."""

import asyncio
import hashlib
import logging
import shutil
import tempfile
import uuid
import zipfile
from pathlib import Path
from datetime import datetime, timezone
from io import BytesIO

from opensearchpy import AsyncOpenSearch
from sqlalchemy import update

from app.archive import ArchiveRejected, plan
from app.config import settings
from app.db import Document, ImportBatch, IngestJob, RagSet, session_scope
from app.db import repo
from app.docling import (
    ChunkingOptions,
    ConversionOptions,
    DoclingFileClient,
    DoclingUrlClient,
)
from app.embeddings import HttpEmbedder
from app.pipeline import IngestPipeline
from app.search import OpenSearchLoader
from app.storage import ObjectStorage, document_key

logger = logging.getLogger(__name__)


def _opensearch() -> AsyncOpenSearch:
    auth = (
        (settings.opensearch_user, settings.opensearch_password)
        if settings.opensearch_user
        else None
    )
    return AsyncOpenSearch(
        hosts=[settings.opensearch_url],
        http_auth=auth,
        verify_certs=bool(auth),
        timeout=settings.opensearch_timeout,
    )


def ingest_document(document_id: str) -> int:
    """Точка входа RQ (синхронная). Внутри — обычный async-пайплайн."""
    return asyncio.run(_ingest_document(uuid.UUID(document_id)))


async def _ingest_document(document_id: uuid.UUID) -> int:
    storage = ObjectStorage()
    os_client = _opensearch()
    loader = OpenSearchLoader(os_client, settings.index_name)
    embedder = HttpEmbedder(
        settings.self_url,
        batch_size=settings.embed_batch_size,
        pool="ingest",
    )

    try:
        async with session_scope() as session:
            document = await session.get(Document, document_id)
            if document is None:
                logger.warning("Документ %s исчез до обработки", document_id)
                return 0
            rag_id = str(document.rag_id)
            storage_key = document.storage_key
            document.status = "processing"
            await session.execute(
                update(IngestJob)
                .where(IngestJob.document_id == document_id)
                .values(started_at=datetime.now(timezone.utc))
            )

        use_upload = settings.docling_transfer == "upload"
        if use_upload:
            payload, _ = storage.get_bytes(
                settings.s3_bucket_docs, storage_key)
            tmp = Path(tempfile.mkdtemp()) / Path(storage_key).name
            tmp.write_bytes(payload)
            source: object = tmp
            client_cls: type = DoclingFileClient
        else:
            source = storage.presigned_get(
                settings.s3_bucket_docs, storage_key)
            client_cls = DoclingUrlClient

        async with client_cls(
            settings.docling_url, api_key=settings.docling_api_key
        ) as chunker:
            pipeline = IngestPipeline(
                chunker,
                embedder,
                loader,
                conversion=ConversionOptions(
                    do_ocr=True,
                    force_ocr=settings.force_ocr,
                    ocr_engine=settings.ocr_engine,
                    ocr_lang=tuple(
                        lang.strip()
                        for lang in settings.ocr_lang.split(",")
                        if lang.strip()
                    ),
                ),
                chunking=ChunkingOptions(
                    tokenizer=settings.embed_model,
                    max_tokens=settings.chunk_max_tokens,
                ),
            )
            try:
                loaded = await pipeline.ingest_source(
                    source, rag_id=rag_id, document_id=str(document_id)
                )
            finally:
                if use_upload:
                    shutil.rmtree(tmp.parent, ignore_errors=True)

        async with session_scope() as session:
            await repo.mark_document(
                session, document_id, "success", chunks_count=loaded
            )
            await session.execute(
                update(IngestJob)
                .where(IngestJob.document_id == document_id)
                .values(finished_at=datetime.now(timezone.utc))
            )
        return loaded

    except Exception as exc:
        logger.exception("Индексация документа %s провалилась", document_id)
        async with session_scope() as session:
            await repo.mark_document(session, document_id, "failed", error=str(exc)[:2000])
            await session.execute(
                update(IngestJob)
                .where(IngestJob.document_id == document_id)
                .values(finished_at=datetime.now(timezone.utc), error=str(exc)[:2000])
            )
        raise
    finally:
        embedder.close()
        await os_client.close()
        from app.db import dispose

        await dispose()


def expand_archive(batch_id: str, staging_key: str) -> int:
    return asyncio.run(_expand_archive(uuid.UUID(batch_id), staging_key))


async def _register_document(
    *,
    rag_id: uuid.UUID,
    filename: str,
    payload: bytes,
    origin: str,
    batch_id: uuid.UUID,
) -> uuid.UUID | None:
    """Объект -> строка -> задача. Порядок фиксирован.

    Упали после записи объекта — остался осиротевший объект без строки:
    невидим, не мешает, подчищается фоновой сверкой. Обратный порядок хуже:
    воркер получил бы задачу на несуществующий файл, и документ навсегда
    завис бы в failed — а это видно пользователю.

    Возвращает None, если такой контент в наборе уже есть.
    """
    from app.tasks.conn import ingest_queue

    storage = ObjectStorage()
    content_hash = hashlib.blake2b(payload, digest_size=16).hexdigest()

    async with session_scope() as session:
        if await repo.find_by_hash(session, rag_id, content_hash) is not None:
            return None

    document_id = uuid.uuid4()
    key = document_key(str(rag_id), str(document_id), filename)
    storage.put_bytes(settings.s3_bucket_docs, key,
                      payload, "application/octet-stream")

    async with session_scope() as session:
        session.add(
            Document(
                id=document_id,
                rag_id=rag_id,
                filename=filename,
                size_bytes=len(payload),
                content_hash=content_hash,
                storage_key=key,
                origin=origin,
                origin_ref=batch_id,
            )
        )

    job = ingest_queue().enqueue(ingest_document, str(document_id))
    async with session_scope() as session:
        await repo.record_job(session, document_id, job.id)
    return document_id


async def _expand_archive(batch_id: uuid.UUID, staging_key: str) -> int:
    storage = ObjectStorage()
    created = skipped = 0

    try:
        async with session_scope() as session:
            batch = await session.get(ImportBatch, batch_id)
            if batch is None:
                logger.warning("Батч %s исчез до обработки", batch_id)
                return 0
            rag_id = batch.rag_id
            batch.status = "processing"

        raw, _ = storage.get_bytes(settings.s3_bucket_staging, staging_key)
        with zipfile.ZipFile(BytesIO(raw)) as archive:
            archive_plan = plan(archive)
            skipped = len(archive_plan.skipped)
            for name, reason in archive_plan.skipped:
                logger.info("Пропущено %s: %s", name, reason)

            for item in archive_plan.entries:
                document_id = await _register_document(
                    rag_id=rag_id,
                    filename=item.name,
                    payload=archive.read(item.raw_name),
                    origin="archive",
                    batch_id=batch_id,
                )
                if document_id is None:
                    skipped += 1
                else:
                    created += 1

        async with session_scope() as session:
            await repo.finish_batch(
                session,
                batch_id,
                status="success",
                created=created,
                skipped=skipped,
                total=created + skipped,
            )

    except ArchiveRejected as exc:
        async with session_scope() as session:
            await repo.finish_batch(session, batch_id, status="rejected", error=str(exc))
        logger.warning("Архив отклонён (батч %s): %s", batch_id, exc)
    except Exception as exc:
        logger.exception("Разворачивание архива %s провалилось", batch_id)
        async with session_scope() as session:
            await repo.finish_batch(
                session, batch_id, status="failed", error=str(exc)[:2000]
            )
        raise
    finally:
        storage.delete(settings.s3_bucket_staging, staging_key)
        from app.db import dispose

        await dispose()

    return created


def import_s3_prefix(batch_id: str, prefix: str) -> int:
    return asyncio.run(_import_s3_prefix(uuid.UUID(batch_id), prefix))


async def _import_s3_prefix(batch_id: uuid.UUID, prefix: str) -> int:
    """Импорт КОПИРУЕТ объект в rag-docs, а не ссылается на staging.

    Иначе уборка staging выбила бы документы из-под набора, а переиндексация
    через полгода упёрлась бы в удалённый объект. После копирования все три
    источника документов неразличимы.
    """
    from app.tasks.conn import ingest_queue

    storage = ObjectStorage()
    created = skipped = 0

    try:
        async with session_scope() as session:
            batch = await session.get(ImportBatch, batch_id)
            if batch is None:
                return 0
            rag_id = batch.rag_id
            batch.status = "processing"

        objects = storage.list_prefix(settings.s3_bucket_staging, prefix)
        for obj in objects:
            key = obj["Key"]
            filename = key.rsplit("/", 1)[-1]
            if not filename:
                continue
            suffix = "." + \
                filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
            if suffix not in settings.allowed_suffixes:
                skipped += 1
                continue

            payload, _ = storage.get_bytes(settings.s3_bucket_staging, key)
            content_hash = hashlib.blake2b(payload, digest_size=16).hexdigest()

            async with session_scope() as session:
                if await repo.find_by_hash(session, rag_id, content_hash) is not None:
                    skipped += 1
                    continue

            document_id = uuid.uuid4()
            dst = document_key(str(rag_id), str(document_id), filename)
            size = storage.copy(settings.s3_bucket_staging,
                                key, settings.s3_bucket_docs, dst)

            async with session_scope() as session:
                session.add(
                    Document(
                        id=document_id,
                        rag_id=rag_id,
                        filename=filename,
                        size_bytes=size,
                        content_hash=content_hash,
                        storage_key=dst,
                        origin="s3",
                        origin_ref=batch_id,
                    )
                )
            job = ingest_queue().enqueue(ingest_document, str(document_id))
            async with session_scope() as session:
                await repo.record_job(session, document_id, job.id)
            created += 1

        async with session_scope() as session:
            await repo.finish_batch(
                session,
                batch_id,
                status="success",
                created=created,
                skipped=skipped,
                total=created + skipped,
            )
    except Exception as exc:
        logger.exception("Импорт префикса %s провалился", prefix)
        async with session_scope() as session:
            await repo.finish_batch(session, batch_id, status="failed", error=str(exc)[:2000])
        raise
    finally:
        from app.db import dispose

        await dispose()

    return created


def purge_rag(rag_id: str) -> int:
    """Чанки, объекты и строки — после того, как набор уже скрыт от владельца."""
    return asyncio.run(_purge_rag(uuid.UUID(rag_id)))


async def _purge_rag(rag_id: uuid.UUID) -> int:
    storage = ObjectStorage()
    os_client = _opensearch()
    loader = OpenSearchLoader(os_client, settings.index_name)
    try:
        deleted = await loader.delete_rag(str(rag_id))
        storage.delete_prefix(settings.s3_bucket_docs, f"{rag_id}/")
        storage.delete(settings.s3_bucket_icons, str(rag_id))
        async with session_scope() as session:
            obj = await session.get(RagSet, rag_id)
            if obj is not None:
                await session.delete(obj)
        return deleted
    finally:
        await os_client.close()
        from app.db import dispose

        await dispose()


def purge_document(rag_id: str, document_id: str, storage_key: str) -> int:
    return asyncio.run(_purge_document(rag_id, document_id, storage_key))


async def _purge_document(rag_id: str, document_id: str, storage_key: str) -> int:
    storage = ObjectStorage()
    os_client = _opensearch()
    loader = OpenSearchLoader(os_client, settings.index_name)
    try:
        deleted = await loader.delete_document(rag_id, document_id)
        storage.delete(settings.s3_bucket_docs, storage_key)
        return deleted
    finally:
        await os_client.close()
        from app.db import dispose

        await dispose()
