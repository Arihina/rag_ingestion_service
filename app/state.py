from __future__ import annotations


"""Разделяемое состояние процесса, собирается в lifespan."""

import asyncio

from opensearchpy import AsyncOpenSearch

from app.docling import DoclingFileClient
from app.embeddings import BgeM3Embedder
from app.jobs import JobRegistry
from app.pipeline import IngestPipeline
from app.search import OpenSearchLoader


class AppState:
    embedder: BgeM3Embedder
    query_embedder: BgeM3Embedder
    chunker: DoclingFileClient
    loader: OpenSearchLoader
    pipeline: IngestPipeline
    os_client: AsyncOpenSearch
    jobs: JobRegistry
    embed_sem: asyncio.Semaphore
    query_sem: asyncio.Semaphore


state = AppState()
