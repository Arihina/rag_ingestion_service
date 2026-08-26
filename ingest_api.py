"""
Два назначения:
  1. Загрузка. Принять файл, отдать в docling, посчитать эмбеддинги,
     положить в OpenSearch. Долгая работа уходит в фон.
  2. Симметрия. /embed отдаёт вектор тем же экземпляром bge-m3, который
     считал векторы документов. Поисковому бэкенду не нужен ни torch,
     ни GPU, а вектор запроса не может разойтись с индексом.
Модель грузится один раз на старте: инициализация bge-m3 не мгновенная.
"""

from __future__ import annotations

import asyncio
import logging
import secrets
import shutil
import tempfile
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, AsyncIterator

from fastapi import (
    BackgroundTasks,
    Depends,
    FastAPI,
    File,
    Header,
    HTTPException,
    UploadFile,
)
from opensearchpy import AsyncOpenSearch
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from docling_client import ChunkingOptions, ConversionOptions, DoclingFileClient
from rag_ingest import BgeM3Embedder, IngestPipeline, OpenSearchLoader

logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="INGEST_", env_file="ingest.env", extra="ignore"
    )

    docling_url: str = "http://localhost:5001"
    docling_api_key: str | None = None

    opensearch_url: str = "http://localhost:9200"
    opensearch_user: str | None = None
    opensearch_password: str | None = None
    index_name: str = "kb-v1"

    embed_model: str = "BAAI/bge-m3"
    embed_batch_size: int = 16
    embed_max_length: int = 1024
    embed_device: str | None = None

    chunk_max_tokens: int = 768
    ocr_engine: str = "easyocr"
    ocr_lang: str = "ru,en"
    force_ocr: bool = True

    api_key: str | None = None
    max_upload_mb: int = 200
    embed_concurrency: int = 1
    max_pending_jobs: int = 50

    allowed_suffixes: tuple[str, ...] = (
        ".pdf", ".docx", ".pptx", ".xlsx", ".html", ".htm",
        ".md", ".txt", ".csv", ".png", ".jpg", ".jpeg", ".tiff",
    )


settings = Settings()


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILURE = "failure"


@dataclass
class Job:
    job_id: str
    filename: str
    status: JobStatus = JobStatus.PENDING
    chunks_loaded: int = 0
    error: str | None = None
    created_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc))
    finished_at: datetime | None = None


class JobRegistry:
    def __init__(self, max_entries: int = 1000) -> None:
        self._jobs: dict[str, Job] = {}
        self._max_entries = max_entries

    def create(self, filename: str) -> Job:
        job = Job(job_id=uuid.uuid4().hex, filename=filename)
        self._jobs[job.job_id] = job
        self._evict()
        return job

    def get(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)

    def pending_count(self) -> int:
        return sum(
            1
            for j in self._jobs.values()
            if j.status in (JobStatus.PENDING, JobStatus.RUNNING)
        )

    def _evict(self) -> None:
        if len(self._jobs) <= self._max_entries:
            return
        finished = sorted(
            (j for j in self._jobs.values() if j.finished_at),
            key=lambda j: j.finished_at,  # type: ignore[arg-type]
        )
        for job in finished[: len(self._jobs) - self._max_entries]:
            self._jobs.pop(job.job_id, None)


class JobResponse(BaseModel):
    job_id: str
    filename: str
    status: JobStatus
    chunks_loaded: int = 0
    error: str | None = None

    @classmethod
    def of(cls, job: Job) -> "JobResponse":
        return cls(
            job_id=job.job_id,
            filename=job.filename,
            status=job.status,
            chunks_loaded=job.chunks_loaded,
            error=job.error,
        )


class EmbedRequest(BaseModel):
    texts: list[str] = Field(min_length=1, max_length=256)


class EmbedItem(BaseModel):
    dense: list[float]
    sparse: dict[str, float]


class EmbedResponse(BaseModel):
    model: str
    embeddings: list[EmbedItem]


class HealthResponse(BaseModel):
    status: str
    docling: bool
    opensearch: bool
    model: str
    pending_jobs: int


class AppState:
    embedder: BgeM3Embedder
    chunker: DoclingFileClient
    loader: OpenSearchLoader
    pipeline: IngestPipeline
    os_client: AsyncOpenSearch
    jobs: JobRegistry
    embed_sem: asyncio.Semaphore


state = AppState()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    logger.info("Загрузка %s...", settings.embed_model)
    state.embedder = BgeM3Embedder(
        settings.embed_model,
        batch_size=settings.embed_batch_size,
        max_length=settings.embed_max_length,
        device=settings.embed_device,
    )

    auth = (
        (settings.opensearch_user, settings.opensearch_password)
        if settings.opensearch_user
        else None
    )
    state.os_client = AsyncOpenSearch(
        hosts=[settings.opensearch_url], http_auth=auth, verify_certs=bool(
            auth)
    )
    state.loader = OpenSearchLoader(state.os_client, settings.index_name)
    await state.loader.ensure_index()

    state.chunker = DoclingFileClient(
        settings.docling_url, api_key=settings.docling_api_key
    )
    state.pipeline = IngestPipeline(
        state.chunker,
        state.embedder,
        state.loader,
        conversion=ConversionOptions(
            do_ocr=True,
            force_ocr=settings.force_ocr,
            ocr_engine=settings.ocr_engine,
            ocr_lang=tuple(
                lang.strip() for lang in settings.ocr_lang.split(",") if lang.strip()
            ),
        ),
        chunking=ChunkingOptions(
            tokenizer=settings.embed_model, max_tokens=settings.chunk_max_tokens
        ),
    )
    state.jobs = JobRegistry()
    state.embed_sem = asyncio.Semaphore(settings.embed_concurrency)

    logger.info("Готов")
    yield

    await state.chunker.aclose()
    await state.os_client.close()


app = FastAPI(title="RAG Ingest API", version="1.1.0", lifespan=lifespan)


async def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    if not settings.api_key:
        return

    if not x_api_key or not secrets.compare_digest(x_api_key, settings.api_key):
        raise HTTPException(
            status_code=401, detail="Неверный или отсутствующий ключ")


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    docling_ok = await state.chunker.health()
    try:
        opensearch_ok = await state.os_client.ping()
    except Exception:
        logger.warning("opensearch недоступен", exc_info=True)
        opensearch_ok = False

    return HealthResponse(
        status="ok" if (docling_ok and opensearch_ok) else "degraded",
        docling=docling_ok,
        opensearch=opensearch_ok,
        model=settings.embed_model,
        pending_jobs=state.jobs.pending_count(),
    )


@app.post(
    "/ingest/file",
    response_model=JobResponse,
    status_code=202,
    dependencies=[Depends(require_api_key)],
)
async def ingest_file(
    background: BackgroundTasks, file: UploadFile = File(...)
) -> JobResponse:
    """Принять файл и поставить в обработку.

    Отвечает сразу: конверсия скана занимает секунды на страницу.
    """
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

    job = state.jobs.create(file.filename)
    background.add_task(_run_ingest, job, tmp_path, tmp_dir)
    return JobResponse.of(job)


@app.get(
    "/ingest/status/{job_id}",
    response_model=JobResponse,
    dependencies=[Depends(require_api_key)],
)
async def job_status(job_id: str) -> JobResponse:
    job = state.jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Задача не найдена")
    return JobResponse.of(job)


@app.post(
    "/embed", response_model=EmbedResponse, dependencies=[Depends(require_api_key)]
)
async def embed(request: EmbedRequest) -> EmbedResponse:
    """Тот же экземпляр модели, что векторизует документы.

    Никаких префиксов: bge-m3 не различает запрос и пассаж на входе.
    """
    async with state.embed_sem:
        results = await asyncio.to_thread(state.embedder.embed, request.texts)

    return EmbedResponse(
        model=settings.embed_model,
        embeddings=[EmbedItem(dense=r.dense, sparse=r.sparse)
                    for r in results],
    )


async def _run_ingest(job: Job, path: Path, tmp_dir: Path) -> None:
    job.status = JobStatus.RUNNING
    try:
        async with state.embed_sem:
            job.chunks_loaded = await state.pipeline.ingest_file(
                path, source_uri=job.filename
            )
        job.status = JobStatus.SUCCESS
    except Exception as exc:
        logger.exception("Задача %s провалилась", job.job_id)
        job.status = JobStatus.FAILURE
        job.error = str(exc)
    finally:
        job.finished_at = datetime.now(timezone.utc)
        shutil.rmtree(tmp_dir, ignore_errors=True)
