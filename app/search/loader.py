from __future__ import annotations


"""Создание индекса, идемпотентная bulk-загрузка, удаление по набору."""

import logging
from typing import Any, Sequence

from opensearchpy import AsyncOpenSearch
from opensearchpy.helpers import async_bulk

from app.config import settings
from app.docling import DoclingChunk
from app.embeddings import EmbeddingResult

from .index import INDEX_BODY
from .scoring import cosine_from_score

logger = logging.getLogger(__name__)


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

        if succeeded and settings.refresh_after_load:
            await self._client.indices.refresh(index=self._index)

        return succeeded

    async def existing_hashes(self, rag_id: str, document_id: str) -> dict[str, str]:
        """Хэши уже загруженных чанков документа в границах набора.

        rag_id в фильтре избыточен при корректном document_id-UUID, но стоит
        дёшево и исключает целый класс аварий, если UUID когда-нибудь начнут
        генерировать снаружи.
        """
        if not await self._client.indices.exists(index=self._index):
            return {}
        response = await self._client.search(
            index=self._index,
            body={
                "query": {
                    "bool": {
                        "filter": [
                            {"term": {"rag_id": rag_id}},
                            {"term": {"document_id": document_id}},
                        ]
                    }
                },
                "_source": ["content_hash"],
                "size": 10_000,
            },
        )
        return {
            hit["_id"]: hit["_source"]["content_hash"]
            for hit in response["hits"]["hits"]
        }

    async def verify_cosine_calibration(self, tolerance: float = 0.02) -> None:
        """Взять любой чанк, найти его же по собственному вектору и убедиться,
        что обратное преобразование _score даёт ~1.0.

        Пустой индекс — не ошибка: проверять нечего, выходим молча.
        """
        if not await self._client.indices.exists(index=self._index):
            return
        probe = await self._client.search(
            index=self._index,
            body={"query": {"match_all": {}}, "_source": [
                "content_vector"], "size": 1},
        )
        hits = probe["hits"]["hits"]
        if not hits:
            return

        vector = hits[0]["_source"]["content_vector"]
        found = await self._client.search(
            index=self._index,
            body={
                "query": {"knn": {"content_vector": {"vector": vector, "k": 1}}},
                "_source": False,
                "size": 1,
            },
        )
        score = found["hits"]["hits"][0]["_score"]
        cosine = cosine_from_score(score)
        if abs(cosine - 1.0) > tolerance:
            raise RuntimeError(
                f"Калибровка косинуса не сошлась: _score={score}, "
                f"cosine_from_score={cosine:.4f}, ожидалось ~1.0. "
                "Формула преобразования не соответствует движку/версии "
                "OpenSearch — score_threshold будет значить не то, что задал "
                "пользователь. Проверь space_type и engine в маппинге."
            )
        logger.info("Калибровка косинуса: _score=%.4f -> cos=%.4f",
                    score, cosine)

    async def delete_rag(self, rag_id: str) -> int:
        """Снести все чанки набора. Вызывается при удалении набора."""
        response = await self._client.delete_by_query(
            index=self._index,
            body={"query": {"term": {"rag_id": rag_id}}},
            params={"conflicts": "proceed", "refresh": "true"},
        )
        deleted = response.get("deleted", 0)
        logger.info("Набор %s: удалено %d чанков", rag_id, deleted)
        return deleted

    async def delete_document(self, rag_id: str, document_id: str) -> int:
        response = await self._client.delete_by_query(
            index=self._index,
            body={
                "query": {
                    "bool": {
                        "filter": [
                            {"term": {"rag_id": rag_id}},
                            {"term": {"document_id": document_id}},
                        ]
                    }
                }
            },
            params={"conflicts": "proceed", "refresh": "true"},
        )
        return response.get("deleted", 0)
