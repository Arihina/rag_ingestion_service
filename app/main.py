from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI
from opensearchpy import AsyncOpenSearch

from app.api import embed, health, ingest
from app.config import settings
from app.docling import ChunkingOptions, ConversionOptions, DoclingFileClient
from app.embeddings import BgeM3Embedder
from app.jobs import JobRegistry
from app.pipeline import IngestPipeline
from app.search import OpenSearchLoader
from app.state import state

import asyncio

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    logger.info("Загрузка %s...", settings.embed_model)
    state.embedder = BgeM3Embedder(
        settings.embed_model,
        batch_size=settings.embed_batch_size,
        max_length=settings.embed_max_length,
        device=settings.embed_device,
    )

    if settings.separate_query_embedder:
        logger.info("Загрузка второго экземпляра %s под запросы...",
                    settings.embed_model)
        state.query_embedder = BgeM3Embedder(
            settings.embed_model,
            batch_size=settings.embed_query_batch_size,
            max_length=settings.embed_max_length,
            device=settings.embed_device,
        )
    else:
        state.query_embedder = state.embedder

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
    await state.loader.verify_cosine_calibration()

    state.chunker = DoclingFileClient(
        settings.docling_url, api_key=settings.docling_api_key
    )
    state.pipeline = IngestPipeline(
        state.chunker,
        state.embedder,
        state.loader,
        embed_sem=state.embed_sem,
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
    state.query_sem = asyncio.Semaphore(settings.embed_query_concurrency)

    logger.info("Готов")
    yield

    await state.chunker.aclose()
    await state.os_client.close()


app = FastAPI(title="RAG Ingest API", version="2.0.0", lifespan=lifespan)

app.include_router(health.router)
app.include_router(embed.router)
app.include_router(ingest.router)
