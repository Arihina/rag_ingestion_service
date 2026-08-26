"""Загрузчик: docling -> bge-m3 -> OpenSearch.

bge-m3 НЕ использует инструкционные префиксы — ни "query: ", ни "passage: ".
Забытый от e5 префикс не даст ошибки, он просто просадит выдачу.
"""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from opensearchpy import AsyncOpenSearch
from opensearchpy.helpers import async_bulk

from docling_client import (
    ChunkingOptions,
    ConversionOptions,
    DoclingChunk,
    DoclingFileClient,
)

logger = logging.getLogger(__name__)

EMBED_DIM = 1024

SEARCH_SOURCE_EXCLUDES = ["content_vector", "content_sparse"]


@dataclass(frozen=True)
class EmbeddingResult:
    dense: list[float]
    sparse: dict[str, float]


class BaseEmbedder(ABC):
    """Считает эмбеддинги пачками. Подклассы реализуют только вызов модели."""

    def __init__(self, batch_size: int = 16) -> None:
        self._batch_size = batch_size

    def embed(self, texts: Sequence[str]) -> list[EmbeddingResult]:
        results: list[EmbeddingResult] = []
        for start in range(0, len(texts), self._batch_size):
            batch = list(texts[start: start + self._batch_size])
            try:
                results.extend(self._embed_batch(batch))
            except Exception as exc:
                raise RuntimeError(
                    f"Ошибка эмбеддинга батча {start}..{start + len(batch)}: {exc}"
                ) from exc
        return results

    @abstractmethod
    def _embed_batch(self, texts: list[str]) -> list[EmbeddingResult]:
        """Прогнать один батч через модель."""


class BgeM3Embedder(BaseEmbedder):
    """bge-m3 через FlagEmbedding: dense и sparse из одного прохода.

    FlagEmbedding, а не sentence-transformers: последний отдаёт только dense,
    доступа к sparse-голове там нет. А lexical_weights — это готовая замена
    SPLADE для третьей ветки гибрида.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        *,
        batch_size: int = 16,
        max_length: int = 1024,
        use_fp16: bool = True,
        device: str | None = None,
    ) -> None:
        super().__init__(batch_size=batch_size)
        from FlagEmbedding import BGEM3FlagModel

        self._max_length = max_length
        self._model = BGEM3FlagModel(
            model_name, use_fp16=use_fp16, devices=device)

    def _embed_batch(self, texts: list[str]) -> list[EmbeddingResult]:
        output = self._model.encode(
            texts,
            batch_size=len(texts),
            max_length=self._max_length,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
        )
        dense = output["dense_vecs"]
        sparse = output["lexical_weights"]

        return [
            EmbeddingResult(
                dense=[float(x) for x in dense[i]],
                sparse={
                    str(token): float(weight)
                    for token, weight in sparse[i].items()
                    if float(weight) > 0
                },
            )
            for i in range(len(texts))
        ]


class HttpEmbedder(BaseEmbedder):
    """Обращение к вынесенному сервису эмбеддингов (ingest_api /embed).

    Модель существует в одном экземпляре на всю систему, поэтому вектор
    запроса физически не может разойтись с вектором документа.
    """

    def __init__(
        self,
        url: str,
        *,
        batch_size: int = 32,
        api_key: str | None = None,
        timeout: float = 120.0,
    ) -> None:
        super().__init__(batch_size=batch_size)
        import httpx

        self._url = url.rstrip("/")
        self._client = httpx.Client(
            timeout=timeout,
            headers={"X-Api-Key": api_key} if api_key else {},
        )

    def _embed_batch(self, texts: list[str]) -> list[EmbeddingResult]:
        response = self._client.post(
            f"{self._url}/embed", json={"texts": texts})
        response.raise_for_status()
        return [
            EmbeddingResult(dense=item["dense"], sparse=item.get("sparse", {}))
            for item in response.json()["embeddings"]
        ]

    def close(self) -> None:
        self._client.close()


INDEX_BODY: dict[str, Any] = {
    "settings": {
        "index": {
            "knn": True,
            "number_of_shards": 1,
            "number_of_replicas": 0,
            "refresh_interval": "30s",
        },
        "analysis": {
            "filter": {
                "russian_stop": {"type": "stop", "stopwords": "_russian_"},
                "russian_stemmer": {"type": "stemmer", "language": "russian"},
            },
            "analyzer": {
                "ru_en": {
                    "type": "custom",
                    "tokenizer": "standard",
                    "filter": ["lowercase", "russian_stop", "russian_stemmer"],
                }
            },
        },
    },
    "mappings": {
        "properties": {
            "source_uri": {"type": "keyword"},
            "chunk_index": {"type": "integer"},
            "content": {"type": "text", "analyzer": "ru_en"},
            "headings": {"type": "keyword"},
            "pages": {"type": "integer"},
            "content_hash": {"type": "keyword"},
            "content_vector": {
                "type": "knn_vector",
                "dimension": EMBED_DIM,
                "method": {
                    "name": "hnsw",
                    "engine": "faiss",
                    "space_type": "cosinesimil",
                    "parameters": {"ef_construction": 256, "m": 16},
                },
            },
            "content_sparse": {"type": "rank_features"},
        }
    },
}


class OpenSearchLoader:
    """Создание индекса и идемпотентная bulk-загрузка чанков."""

    def __init__(self, client: AsyncOpenSearch, index: str) -> None:
        self._client = client
        self._index = index

    async def ensure_index(self, body: dict[str, Any] | None = None) -> None:
        """В проде создание индекса — это миграция, а не побочный эффект
        старта сервиса: смена маппинга требует reindex. Вызывать явно."""
        if await self._client.indices.exists(index=self._index):
            logger.info("Индекс %s уже существует", self._index)
            return
        await self._client.indices.create(index=self._index, body=body or INDEX_BODY)
        logger.info("Создан индекс %s", self._index)

    async def load(
        self,
        chunks: Sequence[DoclingChunk],
        embeddings: Sequence[EmbeddingResult],
    ) -> int:
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Чанков {len(chunks)}, векторов {len(embeddings)} — не сходится"
            )
        if not chunks:
            return 0

        actions = [
            {
                "_op_type": "index",
                "_index": self._index,
                "_id": chunk.chunk_id,
                "_source": {
                    **chunk.to_source(),
                    "content_vector": emb.dense,
                    "content_sparse": emb.sparse,
                },
            }
            for chunk, emb in zip(chunks, embeddings)
        ]

        succeeded, errors = await async_bulk(
            self._client, actions, raise_on_error=False
        )
        if errors:
            logger.error("Bulk: %d ошибок, первая: %s", len(errors), errors[0])
        return succeeded

    async def existing_hashes(self, source_uri: str) -> dict[str, str]:
        """Хэши уже загруженных чанков документа."""
        if not await self._client.indices.exists(index=self._index):
            return {}
        response = await self._client.search(
            index=self._index,
            body={
                "query": {"term": {"source_uri": source_uri}},
                "_source": ["content_hash"],
                "size": 10_000,
            },
        )
        return {
            hit["_id"]: hit["_source"]["content_hash"]
            for hit in response["hits"]["hits"]
        }


class IngestPipeline:
    """docling -> bge-m3 -> OpenSearch для одного файла или пачки."""

    def __init__(
        self,
        chunker: DoclingFileClient,
        embedder: BaseEmbedder,
        loader: OpenSearchLoader,
        *,
        conversion: ConversionOptions | None = None,
        chunking: ChunkingOptions | None = None,
    ) -> None:
        self._chunker = chunker
        self._embedder = embedder
        self._loader = loader
        self._conversion = conversion or ConversionOptions()
        self._chunking = chunking or ChunkingOptions()

    async def ingest_file(
        self,
        path: Path,
        *,
        source_uri: str | None = None,
        skip_unchanged: bool = True,
    ) -> int:
        """source_uri — устойчивый идентификатор документа.

        Обязателен, когда path указывает на временный файл: _id чанка
        строится от source_uri, и при загрузке через HTTP путь вида
        /tmp/ingest-XXXX/... меняется на каждый запрос. Тогда повторная
        загрузка того же документа не перезапишет старые чанки, а создаст
        новые с другими _id — дубли накопятся молча.
        """
        uri = source_uri or path.as_posix()

        chunks = await self._chunker.chunk(
            path,
            conversion=self._conversion,
            chunking=self._chunking,
            source_uri=uri,
        )
        if not chunks:
            logger.warning("%s: чанков не получено", uri)
            return 0

        if skip_unchanged:
            known = await self._loader.existing_hashes(uri)
            chunks = [c for c in chunks if known.get(
                c.chunk_id) != c.content_hash]
            if not chunks:
                logger.info("%s: изменений нет, пропуск", uri)
                return 0

        embeddings = await asyncio.to_thread(
            self._embedder.embed, [c.embed_text for c in chunks]
        )
        loaded = await self._loader.load(chunks, embeddings)
        logger.info("%s: загружено %d чанков", uri, loaded)
        return loaded

    async def ingest_all(self, paths: Iterable[Path], *, concurrency: int = 3) -> int:
        semaphore = asyncio.Semaphore(concurrency)

        async def _guarded(path: Path) -> int:
            async with semaphore:
                try:
                    return await self.ingest_file(path)
                except Exception:
                    logger.exception("Не удалось загрузить %s", path)
                    return 0

        results = await asyncio.gather(*(_guarded(p) for p in paths))
        return sum(results)


async def main() -> None:
    logging.basicConfig(level=logging.INFO)

    os_client = AsyncOpenSearch(hosts=["http://localhost:9200"])
    embedder = BgeM3Embedder(batch_size=16, max_length=1024)
    loader = OpenSearchLoader(os_client, index="kb-v1")
    await loader.ensure_index()

    async with DoclingFileClient("http://localhost:5001") as chunker:
        pipeline = IngestPipeline(
            chunker,
            embedder,
            loader,
            conversion=ConversionOptions(
                do_ocr=True, force_ocr=True, ocr_lang=("eslav",)
            ),
            chunking=ChunkingOptions(tokenizer="BAAI/bge-m3", max_tokens=768),
        )
        total = await pipeline.ingest_all(sorted(Path("./corpus").glob("**/*.pdf")))

    logger.info("Всего загружено чанков: %d", total)
    await os_client.close()


if __name__ == "__main__":
    asyncio.run(main())
