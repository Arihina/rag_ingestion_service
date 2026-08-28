from __future__ import annotations


"""docling -> bge-m3 -> OpenSearch для одного файла или пачки."""

import asyncio
import logging
from pathlib import Path
from typing import Iterable

from app.docling import BaseDoclingClient, ChunkingOptions, ConversionOptions
from app.embeddings import BaseEmbedder
from app.search import OpenSearchLoader

logger = logging.getLogger(__name__)


class IngestPipeline:
    """docling -> bge-m3 -> OpenSearch для одного файла или пачки."""

    def __init__(
        self,
        chunker: BaseDoclingClient,
        embedder: BaseEmbedder,
        loader: OpenSearchLoader,
        *,
        conversion: ConversionOptions | None = None,
        chunking: ChunkingOptions | None = None,
        embed_sem: asyncio.Semaphore | None = None,
    ) -> None:
        self._chunker = chunker
        self._embedder = embedder
        self._loader = loader
        self._conversion = conversion or ConversionOptions()
        self._chunking = chunking or ChunkingOptions()
        self._embed_sem = embed_sem or asyncio.Semaphore(1)

    async def ingest_file(
        self,
        path: Path,
        *,
        rag_id: str,
        document_id: str,
        skip_unchanged: bool = True,
    ) -> int:
        """Идентичность документа приходит из control plane, не из файла."""
        return await self.ingest_source(
            path, rag_id=rag_id, document_id=document_id, skip_unchanged=skip_unchanged
        )

    async def ingest_source(
        self,
        source: object,
        *,
        rag_id: str,
        document_id: str,
        skip_unchanged: bool = True,
    ) -> int:
        """DoclingFileClient принимает путь, DoclingUrlClient — presigned URL.
        Воркер использует второй: docling качает файл из хранилища сам, и
        байты не проходят через наш процесс второй раз.
        """
        chunks = await self._chunker.chunk(
            source,
            rag_id=rag_id,
            document_id=document_id,
            conversion=self._conversion,
            chunking=self._chunking,
        )
        if not chunks:
            logger.warning("%s/%s: чанков не получено", rag_id, document_id)
            return 0

        if skip_unchanged:
            known = await self._loader.existing_hashes(rag_id, document_id)
            chunks = [c for c in chunks if known.get(
                c.chunk_id) != c.content_hash]
            if not chunks:
                logger.info("%s/%s: изменений нет, пропуск",
                            rag_id, document_id)
                return 0

        async with self._embed_sem:
            embeddings = await asyncio.to_thread(
                self._embedder.embed, [c.embed_text for c in chunks]
            )
        loaded = await self._loader.load(chunks, embeddings)
        logger.info("%s/%s: загружено %d чанков", rag_id, document_id, loaded)
        return loaded

    async def ingest_all(
        self,
        items: Iterable[tuple[Path, str, str]],
        *,
        concurrency: int = 3,
    ) -> int:
        """items — тройки (path, rag_id, document_id)."""
        semaphore = asyncio.Semaphore(concurrency)

        async def _guarded(path: Path, rag_id: str, document_id: str) -> int:
            async with semaphore:
                try:
                    return await self.ingest_file(
                        path, rag_id=rag_id, document_id=document_id
                    )
                except Exception:
                    logger.exception("Не удалось загрузить %s", path)
                    return 0

        results = await asyncio.gather(*(_guarded(*item) for item in items))
        return sum(results)
