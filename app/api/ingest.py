from __future__ import annotations

import logging
import shutil
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, UploadFile

from app.config import settings
from app.deps import require_api_key
from app.jobs import Job, JobResponse, JobStatus
from app.state import state


logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ingest", tags=["ingest"])


@router.post(
    "/file",
    response_model=JobResponse,
    status_code=202,
    dependencies=[Depends(require_api_key)],
)
async def ingest_file(
    background: BackgroundTasks,
    file: UploadFile = File(...),
    rag_id: str = Form(...),
    document_id: str = Form(...),
) -> JobResponse:
    """Принять файл и поставить в обработку.

    Отвечает сразу: конверсия скана занимает секунды на страницу.

    rag_id и document_id обязательны — от них строится _id чанка.
    """
    for name, value in (("rag_id", rag_id), ("document_id", document_id)):
        try:
            uuid.UUID(value)
        except ValueError:
            raise HTTPException(
                status_code=400, detail=f"{name} должен быть UUID")
    if not file.filename:
        raise HTTPException(status_code=400, detail="Файл без имени")

    suffix = Path(file.filename).suffix.lower()
    if suffix not in settings.allowed_suffixes:
        raise HTTPException(
            status_code=415,
            detail=f"Расширение {suffix or '<нет>'} не поддерживается",
        )

    if state.jobs.pending_count() >= settings.max_pending_jobs:
        raise HTTPException(
            status_code=429,
            detail=f"В очереди уже {settings.max_pending_jobs} задач, попробуйте позже",
        )

    tmp_dir = Path(tempfile.mkdtemp(prefix="ingest-"))
    tmp_path = tmp_dir / Path(file.filename).name
    limit = settings.max_upload_mb * 1024 * 1024
    written = 0

    try:
        with tmp_path.open("wb") as out:
            while chunk := await file.read(1024 * 1024):
                written += len(chunk)
                if written > limit:
                    raise HTTPException(
                        status_code=413,
                        detail=f"Файл больше {settings.max_upload_mb} МБ",
                    )
                out.write(chunk)
    except Exception:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise

    job = state.jobs.create(file.filename, rag_id, document_id)
    background.add_task(_run_ingest, job, tmp_path, tmp_dir)
    return JobResponse.of(job)


@router.get(
    "/status/{job_id}",
    response_model=JobResponse,
    dependencies=[Depends(require_api_key)],
)
async def job_status(job_id: str) -> JobResponse:
    job = state.jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Задача не найдена")
    return JobResponse.of(job)


async def _run_ingest(job: Job, path: Path, tmp_dir: Path) -> None:
    job.status = JobStatus.RUNNING
    try:
        job.chunks_loaded = await state.pipeline.ingest_file(
            path, rag_id=job.rag_id, document_id=job.document_id
        )
        job.status = JobStatus.SUCCESS
    except Exception as exc:
        logger.exception("Задача %s провалилась", job.job_id)
        job.status = JobStatus.FAILURE
        job.error = str(exc)
    finally:
        job.finished_at = datetime.now(timezone.utc)
        shutil.rmtree(tmp_dir, ignore_errors=True)
