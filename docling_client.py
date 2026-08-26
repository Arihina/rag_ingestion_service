"""HTTP-клиент docling-serve.

Чанкинг выполняет ВСТРОЕННЫЙ HybridChunker на стороне сервиса: мы лишь
передаём ему токенизатор и лимит токенов. Здесь только транспорт —
отправить файл, дождаться задачи, разобрать JSON.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncIterator, Iterable, Sequence

import httpx

logger = logging.getLogger(__name__)


class DoclingError(RuntimeError):
    """Базовая ошибка взаимодействия с docling-serve."""


class DoclingTimeoutError(DoclingError):
    """Задача не завершилась за отведённое время."""


@dataclass(frozen=True)
class ConversionOptions:
    """Параметры конверсии. Точные имена полей сверяй с /docs живого сервера."""

    do_ocr: bool = True
    force_ocr: bool = False

    ocr_engine: str | None = "easyocr"
    ocr_lang: Sequence[str] = ("ru", "en")

    ocr_preset: str | None = None
    table_mode: str = "accurate"
    pdf_backend: str | None = None

    def as_form_data(self) -> dict[str, Any]:
        """httpx проверяет data на Mapping. Список кортежей он принимает
        за сырое тело запроса и молча игнорирует files."""
        data: dict[str, Any] = {
            "do_ocr": str(self.do_ocr).lower(),
            "force_ocr": str(self.force_ocr).lower(),
            "table_mode": self.table_mode,
            "ocr_lang": list(self.ocr_lang),
        }
        if self.ocr_engine:
            data["ocr_engine"] = self.ocr_engine
        if self.ocr_preset:
            data["ocr_preset"] = self.ocr_preset
        if self.pdf_backend:
            data["pdf_backend"] = self.pdf_backend
        return data

    def as_json_options(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "do_ocr": self.do_ocr,
            "force_ocr": self.force_ocr,
            "ocr_lang": list(self.ocr_lang),
            "table_mode": self.table_mode,
        }
        if self.ocr_engine:
            out["ocr_engine"] = self.ocr_engine
        if self.ocr_preset:
            out["ocr_preset"] = self.ocr_preset
        if self.pdf_backend:
            out["pdf_backend"] = self.pdf_backend
        return out


@dataclass(frozen=True)
class ChunkingOptions:
    """Параметры встроенного HybridChunker. Уходят с префиксом chunking_."""

    tokenizer: str = "BAAI/bge-m3"

    max_tokens: int = 768

    merge_peers: bool = True

    _PREFIX = "chunking_"

    def as_form_data(self) -> dict[str, Any]:
        return {
            f"{self._PREFIX}tokenizer": self.tokenizer,
            f"{self._PREFIX}max_tokens": str(self.max_tokens),
            f"{self._PREFIX}merge_peers": str(self.merge_peers).lower(),
        }

    def as_json_options(self) -> dict[str, Any]:
        return {
            f"{self._PREFIX}tokenizer": self.tokenizer,
            f"{self._PREFIX}max_tokens": self.max_tokens,
            f"{self._PREFIX}merge_peers": self.merge_peers,
        }


@dataclass(frozen=True)
class DoclingChunk:
    """Два представления текста, и это принципиально.

    text       — сырой. Идёт в индекс под BM25 и в выдачу пользователю.
    embed_text — контекстуализированный, с приклеенными родительскими
                 заголовками. Идёт ТОЛЬКО в модель.

    Если индексировать контекстуализированный вариант, термины заголовков
    попадут в индекс дважды (в content и в headings), и BM25 завысит им
    term frequency: документы из разделов с удачными названиями начнут
    всплывать безосновательно.
    """

    source_uri: str
    index: int
    text: str
    embed_text: str
    headings: tuple[str, ...] = ()
    pages: tuple[int, ...] = ()
    doc_meta: dict[str, Any] = field(default_factory=dict)

    @property
    def chunk_id(self) -> str:
        """Детерминированный _id: переиндексация перезаписывает, а не плодит."""
        key = f"{self.source_uri}:{self.index}".encode("utf-8")
        return hashlib.blake2b(key, digest_size=16).hexdigest()

    @property
    def content_hash(self) -> str:
        return hashlib.blake2b(self.text.encode("utf-8"), digest_size=16).hexdigest()

    def to_source(self) -> dict[str, Any]:
        return {
            "source_uri": self.source_uri,
            "chunk_index": self.index,
            "content": self.text,
            "headings": list(self.headings),
            "pages": list(self.pages),
            "content_hash": self.content_hash,
            **self.doc_meta,
        }



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
        conversion: ConversionOptions | None = None,
        chunking: ChunkingOptions | None = None,
        source_uri: str | None = None,
    ) -> list[DoclingChunk]:
        conversion = conversion or ConversionOptions()
        chunking = chunking or ChunkingOptions()
        uri = source_uri or self._default_uri(source)

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

        return self._parse_chunks(payload, uri)

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
        """Как назвать источник, если явный source_uri не передан."""


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

    def _parse_chunks(self, payload: dict[str, Any], uri: str) -> list[DoclingChunk]:
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
                    source_uri=uri,
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
        """Вернуть (сырой, контекстуализированный).

        Имена полей различаются между версиями docling-serve. Прогони один
        документ и проверь: если content в индексе начинается с заголовков,
        значит сырой вариант лежит под другим ключом — поправь списки ниже.
        Актуальная схема всегда на /docs живого сервера.
        """
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
        """Номера страниц лежат в двух разных местах в зависимости от версии.

        В v1.21.0 это page_numbers прямо на чанке, а doc_items приходит
        плоским списком ссылок вида "#/texts/4" — без вложенного prov.
        Более старые версии отдавали провенанс внутри meta.doc_items[].prov.
        Проверяем оба.
        """
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
    """Документ по HTTP-ссылке — сервис скачивает его сам.

    Для S3/MinIO выгоднее этого варианта нет: presigned URL, и байты не
    проходят через твой процесс дважды.
    """

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
