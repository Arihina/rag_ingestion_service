from __future__ import annotations

"""Разделяемое состояние процесса, собирается в lifespan.

Модель живёт ЗДЕСЬ, а не в воркерах.
"""

import asyncio

from opensearchpy import AsyncOpenSearch

from app.embeddings import BgeM3Embedder
from app.search import OpenSearchLoader
from app.storage import ObjectStorage


class AppState:
    embedder: BgeM3Embedder
    query_embedder: BgeM3Embedder
    loader: OpenSearchLoader
    os_client: AsyncOpenSearch
    storage: ObjectStorage
    embed_sem: asyncio.Semaphore
    query_sem: asyncio.Semaphore


state = AppState()
