from __future__ import annotations

"""Платформенный API наборов. Проксируется мастером через catch-all."""

import io
import uuid

from fastapi import (
    APIRouter,
    Depends,
    File,
    HTTPException,
    Response,
    UploadFile,
    status,
)
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.db import Document, ImportBatch, RagSet, get_session, repo
from app.deps import current_user
from app.schemas.rags import (
    BatchOut,
    DocumentAccepted,
    DocumentCountsOut,
    DocumentOut,
    RagConfig,
    RagCreate,
    RagOut,
    RagUpdate,
    UploadAccepted,
)
from app.storage import ObjectStorage, document_key, stream_upload
from app.tasks.conn import ingest_queue, maintenance_queue
from app.tasks.jobs import expand_archive, ingest_document, purge_document, purge_rag

router = APIRouter(
    prefix="/v1/platform/rags", tags=["rags"]
)


def _to_out(rag: RagSet, counts: repo.DocumentCounts) -> RagOut:
    return RagOut(
        id=rag.id,
        name=rag.name,
        description=rag.description,
        status=counts.status,
        has_icon=rag.icon_key is not None,
        documents=DocumentCountsOut(
            total=counts.total,
            ready=counts.ready,
            failed=counts.failed,
            pending=counts.pending,
        ),
        chunks_total=counts.chunks,
        has_pending=counts.has_pending,
        config=RagConfig(
            prompt=rag.prompt,
            temperature=rag.temperature,
            top_k=rag.top_k,
            score_threshold=rag.score_threshold,
        ),
        created_at=rag.created_at,
        updated_at=rag.updated_at,
    )


async def _owned(session: AsyncSession, rag_id: uuid.UUID, user_id: uuid.UUID) -> RagSet:
    rag = await repo.get_rag(session, rag_id, user_id)
    if rag is None:
        raise HTTPException(status_code=404, detail="Набор не найден")
    return rag


@router.post("", response_model=RagOut, status_code=status.HTTP_201_CREATED)
async def create_rag(
    payload: RagCreate,
    user_id: uuid.UUID = Depends(current_user),
    session: AsyncSession = Depends(get_session),
) -> RagOut:
    if await repo.count_rags(session, user_id) >= settings.rag_max_sets_per_owner:
        raise HTTPException(
            status_code=409,
            detail=f"Достигнут предел в {settings.rag_max_sets_per_owner} наборов",
        )

    rag = RagSet(owner_id=user_id, **payload.model_dump())
    session.add(rag)
    try:
        await session.commit()
    except IntegrityError:
        await session.rollback()
        raise HTTPException(
            status_code=409, detail="Набор с таким именем уже есть")
    await session.refresh(rag)
    return _to_out(rag, repo.DocumentCounts())


@router.get("", response_model=list[RagOut])
async def list_rags(
    user_id: uuid.UUID = Depends(current_user),
    session: AsyncSession = Depends(get_session),
) -> list[RagOut]:
    rags = await repo.list_rags(session, user_id)
    counts = await repo.counts_for_many(session, [r.id for r in rags])
    return [_to_out(r, counts[r.id]) for r in rags]


@router.get("/{rag_id}", response_model=RagOut)
async def get_rag(
    rag_id: uuid.UUID,
    user_id: uuid.UUID = Depends(current_user),
    session: AsyncSession = Depends(get_session),
) -> RagOut:
    rag = await _owned(session, rag_id, user_id)
    return _to_out(rag, await repo.counts_for(session, rag_id))


@router.patch("/{rag_id}", response_model=RagOut)
async def update_rag(
    rag_id: uuid.UUID,
    payload: RagUpdate,
    user_id: uuid.UUID = Depends(current_user),
    session: AsyncSession = Depends(get_session),
) -> RagOut:
    """Все поля query-time, поэтому правка применяется со следующего
    сообщения и никогда не требует переиндексации."""
    rag = await _owned(session, rag_id, user_id)
    for field, value in payload.model_dump(exclude_unset=True).items():
        setattr(rag, field, value)
    try:
        await session.commit()
    except IntegrityError:
        await session.rollback()
        raise HTTPException(
            status_code=409, detail="Набор с таким именем уже есть")
    await session.refresh(rag)
    return _to_out(rag, await repo.counts_for(session, rag_id))


@router.delete("/{rag_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_rag(
    rag_id: uuid.UUID,
    user_id: uuid.UUID = Depends(current_user),
    session: AsyncSession = Depends(get_session),
) -> Response:
    """Soft-delete отвечает сразу; чанки, объекты и строки убирает фон.
    Порядок обратный загрузке: сначала строка, чтобы пользователь мгновенно
    перестал видеть набор, потом всё остальное.
    """
    await _owned(session, rag_id, user_id)
    await repo.soft_delete_rag(session, rag_id)
    await session.commit()
    maintenance_queue().enqueue(purge_rag, str(rag_id))
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.put("/{rag_id}/icon", response_model=RagOut)
async def put_icon(
    rag_id: uuid.UUID,
    file: UploadFile = File(...),
    user_id: uuid.UUID = Depends(current_user),
    session: AsyncSession = Depends(get_session),
) -> RagOut:
    """SVG не принимаем"""
    if file.content_type not in settings.icon_allowed_types:
        raise HTTPException(
            status_code=415,
            detail=f"Допустимы {', '.join(settings.icon_allowed_types)}",
        )
    rag = await _owned(session, rag_id, user_id)

    payload = await file.read(settings.icon_max_bytes + 1)
    if len(payload) > settings.icon_max_bytes:
        raise HTTPException(
            status_code=413, detail=f"Иконка больше {settings.icon_max_bytes} байт"
        )

    try:
        from PIL import Image

        image = Image.open(io.BytesIO(payload))
        image.verify()
        image = Image.open(io.BytesIO(payload)).convert("RGBA")
        image.thumbnail((settings.icon_size_px, settings.icon_size_px))
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        normalized = buffer.getvalue()
    except Exception:
        raise HTTPException(
            status_code=415, detail="Файл не является изображением")

    ObjectStorage().put_bytes(
        settings.s3_bucket_icons, str(rag_id), normalized, "image/png"
    )
    rag.icon_key = str(rag_id)
    await session.commit()
    await session.refresh(rag)
    return _to_out(rag, await repo.counts_for(session, rag_id))


@router.get("/{rag_id}/icon")
async def get_icon(
    rag_id: uuid.UUID,
    user_id: uuid.UUID = Depends(current_user),
    session: AsyncSession = Depends(get_session),
) -> Response:
    rag = await _owned(session, rag_id, user_id)
    if rag.icon_key is None:
        raise HTTPException(status_code=404, detail="Иконка не задана")
    data, content_type = ObjectStorage().get_bytes(
        settings.s3_bucket_icons, rag.icon_key)
    return Response(content=data, media_type=content_type)


@router.post(
    "/{rag_id}/documents",
    response_model=UploadAccepted,
    status_code=status.HTTP_202_ACCEPTED,
)
async def upload_documents(
    rag_id: uuid.UUID,
    files: list[UploadFile] = File(...),
    user_id: uuid.UUID = Depends(current_user),
    session: AsyncSession = Depends(get_session),
) -> UploadAccepted:
    """Объект -> строка -> задача, порядок фиксирован"""
    await _owned(session, rag_id, user_id)
    storage = ObjectStorage()
    limit = settings.max_upload_mb * 1024 * 1024
    used = await repo.total_bytes(session, rag_id)

    accepted: list[DocumentAccepted] = []
    for file in files:
        if not file.filename:
            raise HTTPException(status_code=400, detail="Файл без имени")
        suffix = "." + \
            file.filename.rsplit(
                ".", 1)[-1].lower() if "." in file.filename else ""
        if suffix not in settings.allowed_suffixes:
            raise HTTPException(
                status_code=415, detail=f"Расширение {suffix or '<нет>'} не поддерживается"
            )

        document_id = uuid.uuid4()
        key = document_key(str(rag_id), str(document_id), file.filename)

        async def _chunks(upload: UploadFile = file):
            while chunk := await upload.read(1024 * 1024):
                yield chunk

        try:
            stored = await stream_upload(
                storage, settings.s3_bucket_docs, key, _chunks(), limit_bytes=limit
            )
        except ValueError as exc:
            raise HTTPException(status_code=413, detail=str(exc))

        used += stored.size_bytes
        if used > settings.rag_max_bytes_per_set:
            storage.delete(settings.s3_bucket_docs, key)
            raise HTTPException(
                status_code=413, detail="Превышен размер набора")

        existing = await repo.find_by_hash(session, rag_id, stored.content_hash)
        if existing is not None:
            storage.delete(settings.s3_bucket_docs, key)
            accepted.append(
                DocumentAccepted(
                    id=existing.id,
                    filename=existing.filename,
                    status=existing.status,
                    duplicate_of=existing.id,
                )
            )
            continue

        session.add(
            Document(
                id=document_id,
                rag_id=rag_id,
                filename=file.filename,
                size_bytes=stored.size_bytes,
                content_hash=stored.content_hash,
                storage_key=key,
                origin="upload",
            )
        )
        await session.commit()

        job = ingest_queue().enqueue(ingest_document, str(document_id))
        await repo.record_job(session, document_id, job.id)
        await session.commit()

        accepted.append(
            DocumentAccepted(
                id=document_id, filename=file.filename, status="pending")
        )

    return UploadAccepted(documents=accepted)


@router.get("/{rag_id}/documents", response_model=list[DocumentOut])
async def list_documents(
    rag_id: uuid.UUID,
    user_id: uuid.UUID = Depends(current_user),
    session: AsyncSession = Depends(get_session),
) -> list[Document]:
    await _owned(session, rag_id, user_id)
    return await repo.list_documents(session, rag_id)


@router.delete("/{rag_id}/documents/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(
    rag_id: uuid.UUID,
    document_id: uuid.UUID,
    user_id: uuid.UUID = Depends(current_user),
    session: AsyncSession = Depends(get_session),
) -> Response:
    await _owned(session, rag_id, user_id)
    document = await repo.get_document(session, rag_id, document_id)
    if document is None:
        raise HTTPException(status_code=404, detail="Документ не найден")
    storage_key = document.storage_key
    await session.delete(document)
    await session.commit()
    maintenance_queue().enqueue(
        purge_document, str(rag_id), str(document_id), storage_key
    )
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post(
    "/{rag_id}/imports/archive",
    response_model=BatchOut,
    status_code=status.HTTP_202_ACCEPTED,
)
async def import_archive(
    rag_id: uuid.UUID,
    file: UploadFile = File(...),
    user_id: uuid.UUID = Depends(current_user),
    session: AsyncSession = Depends(get_session),
) -> ImportBatch:
    """Архив кладётся в staging и разворачивается воркером."""
    await _owned(session, rag_id, user_id)
    if not file.filename or not file.filename.lower().endswith(".zip"):
        raise HTTPException(
            status_code=415, detail="Поддерживается только zip")

    batch = ImportBatch(rag_id=rag_id, kind="archive")
    session.add(batch)
    await session.commit()
    await session.refresh(batch)

    staging_key = f"archives/{rag_id}/{batch.id}.zip"

    async def _chunks():
        while chunk := await file.read(1024 * 1024):
            yield chunk

    try:
        await stream_upload(
            ObjectStorage(),
            settings.s3_bucket_staging,
            staging_key,
            _chunks(),
            limit_bytes=settings.archive_max_uncompressed,
        )
    except ValueError as exc:
        raise HTTPException(status_code=413, detail=str(exc))

    maintenance_queue().enqueue(expand_archive, str(batch.id), staging_key)
    return batch


@router.get("/{rag_id}/imports/{batch_id}", response_model=BatchOut)
async def get_batch(
    rag_id: uuid.UUID,
    batch_id: uuid.UUID,
    user_id: uuid.UUID = Depends(current_user),
    session: AsyncSession = Depends(get_session),
) -> ImportBatch:
    await _owned(session, rag_id, user_id)
    batch = await repo.get_batch(session, batch_id, rag_id)
    if batch is None:
        raise HTTPException(status_code=404, detail="Импорт не найден")
    return batch
