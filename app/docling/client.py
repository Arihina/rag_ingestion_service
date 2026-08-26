from __future__ import annotations


"""Транспорт к docling-serve: submit -> poll -> result, разбор чанков.

Чанкинг выполняет ВСТРОЕННЫЙ HybridChunker на стороне сервиса: мы лишь
передаём ему токенизатор и лимит токенов.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, AsyncIterator, Iterable

import httpx

from .chunk import DoclingChunk
from .options import ChunkingOptions, ConversionOptions

logger = logging.getLogger(__name__)


class DoclingError(RuntimeError):
    """Базовая ошибка взаимодействия с docling-serve."""


class DoclingTimeoutError(DoclingError):
    """Задача не завершилась за отведённое время."""


class BaseDoclingClient(ABC):
    """Общий транспорт: submit -> poll -> result, ошибки, разбор чанков.

    Подклассы описывают только то, как собрать запрос на сабмит.
    """

    CHUNK_PATH = "/v1/chunk/hybrid"

    def __init__(
        self,
        base_url: str,
        *,
        api_key: str | None = None,
        client: httpx.AsyncClient | None = None,
        poll_interval: float = 2.0,
        max_wait: float = 900.0,
        request_timeout: float = 60.0,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._poll_interval = poll_interval
        self._max_wait = max_wait
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(timeout=request_timeout)

    async def __aenter__(self) -> "BaseDoclingClient":
        return self

    async def __aexit__(self, *exc_info: Any) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    @property
    def _headers(self) -> dict[str, str]:
        return {"X-Api-Key": self._api_key} if self._api_key else {}

    async def health(self) -> bool:
        try:
            response = await self._client.get(f"{self._base_url}/health")
            return response.status_code == 200
        except httpx.HTTPError:
            return False

    async def chunk(
        self,
        source: Any,
        *,
        rag_id: str,
        document_id: str,
        conversion: ConversionOptions | None = None,
        chunking: ChunkingOptions | None = None,
    ) -> list[DoclingChunk]:
        """rag_id и document_id обязательны: от них строится _id чанка."""
        conversion = conversion or ConversionOptions()
        chunking = chunking or ChunkingOptions()
        uri = self._default_uri(source)

        try:
            task_id = await self._submit(source, conversion, chunking)
            await self._wait(task_id)
            payload = await self._fetch_result(task_id)
        except httpx.HTTPStatusError as exc:
            body = exc.response.text[:500]
            raise DoclingError(
                f"docling-serve вернул {exc.response.status_code} для {uri}: {body}"
            ) from exc
        except httpx.HTTPError as exc:
            raise DoclingError(
                f"Сетевая ошибка при обработке {uri}: {exc}") from exc

        return self._parse_chunks(payload, uri, rag_id=rag_id, document_id=document_id)

    async def chunk_many(
        self,
        sources: Iterable[Any],
        *,
        concurrency: int = 4,
        **kwargs: Any,
    ) -> AsyncIterator[list[DoclingChunk]]:
        semaphore = asyncio.Semaphore(concurrency)

        async def _guarded(src: Any) -> list[DoclingChunk]:
            async with semaphore:
                return await self.chunk(src, **kwargs)

        tasks = [asyncio.create_task(_guarded(src)) for src in sources]
        for task in asyncio.as_completed(tasks):
            yield await task

    @abstractmethod
    async def _submit(
        self,
        source: Any,
        conversion: ConversionOptions,
        chunking: ChunkingOptions,
    ) -> str:
        """Отправить задачу на /async и вернуть task_id."""

    @abstractmethod
    def _default_uri(self, source: Any) -> str:
        """Человекочитаемое имя источника — только для логов и текстов ошибок.

        В идентичность чанка больше не входит: её задают rag_id/document_id.
        """

    async def _wait(self, task_id: str) -> None:
        url = f"{self._base_url}/v1/status/poll/{task_id}"
        waited = 0.0
        while waited < self._max_wait:
            response = await self._client.get(url, headers=self._headers)
            response.raise_for_status()
            status = response.json().get("task_status")

            if status == "success":
                return
            if status == "failure":
                raise DoclingError(f"Задача {task_id} завершилась с ошибкой")

            await asyncio.sleep(self._poll_interval)
            waited += self._poll_interval

        raise DoclingTimeoutError(
            f"Задача {task_id} не завершилась за {self._max_wait} с"
        )

    async def _fetch_result(self, task_id: str) -> dict[str, Any]:
        response = await self._client.get(
            f"{self._base_url}/v1/result/{task_id}", headers=self._headers
        )
        response.raise_for_status()
        return response.json()

    def _parse_chunks(
        self,
        payload: dict[str, Any],
        uri: str,
        *,
        rag_id: str,
        document_id: str,
    ) -> list[DoclingChunk]:
        raw_chunks = payload.get("chunks") or []
        if not raw_chunks:
            logger.warning("Пустой результат чанкинга для %s", uri)

        chunks: list[DoclingChunk] = []
        for i, raw in enumerate(raw_chunks):
            raw_text, embed_text = self._extract_texts(raw)
            if not raw_text.strip():
                continue
            meta = raw.get("meta") or raw.get("metadata") or {}
            chunks.append(
                DoclingChunk(
                    rag_id=rag_id,
                    document_id=document_id,
                    index=i,
                    text=raw_text,
                    embed_text=embed_text,
                    headings=tuple(
                        raw.get("headings") or meta.get("headings") or ()
                    ),
                    pages=self._extract_pages(raw, meta),
                )
            )
        return chunks

    @staticmethod
    def _extract_texts(raw: dict[str, Any]) -> tuple[str, str]:
        def pick(*keys: str) -> str:
            for key in keys:
                value = raw.get(key)
                if isinstance(value, str) and value:
                    return value
            return ""

        contextual = pick("contextualized_text", "text")
        plain = pick("raw_text", "text")
        return plain or contextual, contextual or plain

    @staticmethod
    def _extract_pages(raw: dict[str, Any], meta: dict[str, Any]) -> tuple[int, ...]:
        pages: set[int] = set()

        for page in raw.get("page_numbers") or ():
            if isinstance(page, int):
                pages.add(page)

        for item in meta.get("doc_items") or ():
            if not isinstance(item, dict):
                continue
            for prov in item.get("prov") or ():
                page = prov.get("page_no")
                if isinstance(page, int):
                    pages.add(page)

        return tuple(sorted(pages))


class DoclingFileClient(BaseDoclingClient):
    """Загрузка локального файла через multipart."""

    async def _submit(
        self,
        source: Path | str,
        conversion: ConversionOptions,
        chunking: ChunkingOptions,
    ) -> str:
        path = Path(source)
        data = {**conversion.as_form_data(), **chunking.as_form_data()}

        payload = await asyncio.to_thread(path.read_bytes)

        response = await self._client.post(
            f"{self._base_url}{self.CHUNK_PATH}/file/async",
            headers=self._headers,
            files={"files": (path.name, payload, "application/octet-stream")},
            data=data,
        )
        response.raise_for_status()
        return response.json()["task_id"]

    def _default_uri(self, source: Path | str) -> str:
        return Path(source).as_posix()


class DoclingUrlClient(BaseDoclingClient):
    """Документ по HTTP-ссылке — сервис скачивает его сам."""

    async def _submit(
        self,
        source: str,
        conversion: ConversionOptions,
        chunking: ChunkingOptions,
    ) -> str:
        body = {
            "http_sources": [{"url": source}],
            "options": {
                **conversion.as_json_options(),
                **chunking.as_json_options(),
            },
        }
        response = await self._client.post(
            f"{self._base_url}{self.CHUNK_PATH}/source/async",
            headers=self._headers,
            json=body,
        )
        response.raise_for_status()
        return response.json()["task_id"]

    def _default_uri(self, source: str) -> str:
        return source
