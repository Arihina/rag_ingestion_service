from __future__ import annotations


"""Клиент к вынесенному /embed. Им пользуется agentic_rag: поисковому
бэкенду не нужен ни torch, ни GPU, а вектор запроса не может разойтись
с вектором документа."""

from .base import BaseEmbedder, EmbeddingResult


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
        timeout: float = 120.0,
        pool: str = "query",
    ) -> None:
        super().__init__(batch_size=batch_size)
        import httpx

        self._url = url.rstrip("/")
        self._pool = pool
        self._client = httpx.Client(timeout=timeout)

    def _embed_batch(self, texts: list[str]) -> list[EmbeddingResult]:
        response = self._client.post(
            f"{self._url}/embed", json={"texts": texts, "pool": self._pool})
        response.raise_for_status()
        return [
            EmbeddingResult(dense=item["dense"], sparse=item.get("sparse", {}))
            for item in response.json()["embeddings"]
        ]

    def close(self) -> None:
        self._client.close()
