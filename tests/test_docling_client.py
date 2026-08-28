from __future__ import annotations

"""Клиент docling-serve: разбор чанков, опрос задачи, форма запроса.

Схема менялась между версиями, и оба известных промаха были молчаливыми:
неверное имя поля опций pydantic на той стороне просто выбрасывает, а
неверное поле текста даёт чанки с заголовками внутри content.
"""

import asyncio

import httpx

from app.docling import (
    ChunkingOptions,
    ConversionOptions,
    DoclingError,
    DoclingFileClient,
    DoclingTimeoutError,
    DoclingUrlClient,
)
from tests.base import FixtureTestCase, load_json
from tests.support import AsyncFixtureTestCase, transport_of


def parse(client: DoclingUrlClient, payload: dict) -> list:
    return client._parse_chunks(payload, "u", rag_id="R", document_id="D")


class ParseChunksTests(FixtureTestCase):
    def setUp(self):
        self.client = DoclingUrlClient(
            "http://docling",
            client=transport_of(lambda request: httpx.Response(200, json={})),
        )
        self.payload = load_json("docling", "chunk_result.json")
        self.chunks = parse(self.client, self.payload)

    def test_пустые_чанки_отбрасываются(self):
        """В фикстуре пять чанков, два из них пустые."""
        self.assertEqual(len(self.payload["chunks"]), 5)
        self.assertEqual(len(self.chunks), 3)

    def test_сырой_текст_идёт_в_content(self):
        """content не должен начинаться с заголовков: иначе BM25 считает
        их частью тела, а в цитате пользователь видит breadcrumb."""
        first = self.chunks[0]
        self.assertEqual(
            first.text, "Настоящий регламент устанавливает порядок обслуживания.")
        self.assertNotIn("Общие положения", first.text)

    def test_контекстуализированный_текст_идёт_в_эмбеддинг(self):
        """А вот вектор считается по тексту С заголовками — это и есть
        смысл contextualized_text."""
        first = self.chunks[0]
        self.assertIn("Общие положения", first.embed_text)
        self.assertNotEqual(first.text, first.embed_text)

    def test_единственное_поле_text_используется_для_обоих(self):
        second = self.chunks[1]
        self.assertEqual(second.text, "Поверка производится ежегодно.")
        self.assertEqual(second.embed_text, second.text)

    def test_только_contextualized_text(self):
        """Старая форма ответа: raw_text отсутствует вовсе."""
        third = self.chunks[2]
        self.assertIn("Приложение А", third.text)

    def test_breadcrumb_сохраняется(self):
        self.assertEqual(self.chunks[0].headings,
                         ("Регламент РЩ-3", "Общие положения"))

    def test_headings_из_meta_у_старой_формы(self):
        self.assertEqual(self.chunks[2].headings, ("Приложения",))

    def test_страницы_из_page_numbers(self):
        """v1.21: номера страниц прямо на чанке."""
        self.assertEqual(self.chunks[0].pages, (1,))

    def test_страницы_из_старого_провенанса(self):
        """Старые версии клали их в meta.doc_items[].prov — читаем оба места."""
        self.assertEqual(self.chunks[1].pages, (4, 5))

    def test_чанки_нумеруются_по_порядку(self):
        self.assertEqual([c.index for c in self.chunks], [0, 1, 2])

    def test_идентичность_проставляется_всем_чанкам(self):
        for chunk in self.chunks:
            self.assertEqual(chunk.rag_id, "R")
            self.assertEqual(chunk.document_id, "D")

    def test_пустой_результат_не_падает(self):
        self.assertEqual(parse(self.client, {"chunks": []}), [])
        self.assertEqual(parse(self.client, {}), [])


class SubmitPayloadTests(AsyncFixtureTestCase):
    """Форма запроса на сабмит. Оба поля тела меняли имена в v1."""

    async def test_url_клиент_шлёт_sources_с_kind(self):
        captured = {}

        async def handler(request: httpx.Request) -> httpx.Response:
            captured.update(request.read() and __import__(
                "json").loads(request.read()))
            return httpx.Response(200, json={"task_id": "t-1"})

        async with DoclingUrlClient("http://d", client=transport_of(handler)) as client:
            await client._submit("http://s3/f.pdf", ConversionOptions(), ChunkingOptions())

        self.assertNotIn("http_sources", captured)
        self.assertEqual(captured["sources"], [
                         {"kind": "http", "url": "http://s3/f.pdf"}])

    async def test_опции_лежат_под_convert_options(self):
        """У chunk-эндпоинтов объект опций называется НЕ options.
        Промах не даёт ошибки: pydantic молча выбрасывает весь блок, и
        документ обрабатывается с дефолтами — включая чужой OCR."""
        captured = {}

        async def handler(request: httpx.Request) -> httpx.Response:
            captured.update(__import__("json").loads(request.read()))
            return httpx.Response(200, json={"task_id": "t-1"})

        async with DoclingUrlClient("http://d", client=transport_of(handler)) as client:
            await client._submit(
                "http://s3/f.pdf",
                ConversionOptions(force_ocr=True),
                ChunkingOptions(max_tokens=512),
            )

        self.assertNotIn("options", captured)
        self.assertIs(captured["convert_options"]["force_ocr"], True)
        self.assertEqual(captured["chunking_options"]["max_tokens"], 512)
        self.assertEqual(captured["chunking_options"]
                         ["tokenizer"], "BAAI/bge-m3")

    async def test_api_key_уходит_заголовком(self):
        seen = {}

        async def handler(request: httpx.Request) -> httpx.Response:
            seen["key"] = request.headers.get("x-api-key")
            return httpx.Response(200, json={"task_id": "t"})

        async with DoclingUrlClient(
            "http://d", api_key="secret", client=transport_of(handler)
        ) as client:
            await client._submit("http://s/f.pdf", ConversionOptions(), ChunkingOptions())
        self.assertEqual(seen["key"], "secret")


class WaitTests(AsyncFixtureTestCase):
    """Тесты цикла опроса. Каждый под жёстким внешним пределом.

    Два уровня защиты, и оба нужны.

    Первый — короткий max_wait у клиента вместо дефолтных 900 секунд:
    иначе тест, в котором ветка выхода не сработала, честно опрашивает
    пятнадцать минут.

    Второй — asyncio.timeout поверх самого вызова. Он страхует от случая,
    когда сломан не выход по условию, а САМ механизм дедлайна: тогда
    max_wait не поможет, потому что до его проверки дело не дойдёт. Ровно
    это и происходит, если _wait копит сон вместо монотонного дедлайна —
    при poll_interval=0 счётчик не растёт, и цикл вечен.

    Без второго уровня такая поломка вешает весь прогон вместо того,
    чтобы дать красный тест.
    """

    HARD_LIMIT = 5.0

    @staticmethod
    def _client(handler):
        return DoclingUrlClient(
            "http://d", client=transport_of(handler), poll_interval=0, max_wait=0.5
        )

    async def _wait_bounded(self, client, task_id: str = "t-1") -> None:
        try:
            async with asyncio.timeout(self.HARD_LIMIT):
                await client._wait(task_id)
        except TimeoutError as exc:
            raise AssertionError(
                f"_wait не вернулся за {self.HARD_LIMIT} с при max_wait=0.5. "
                "Похоже, дедлайн считается накоплением сна, а не по "
                "time.monotonic(): при poll_interval=0 счётчик не растёт."
            ) from exc

    async def test_успех_возвращает_управление(self):
        async def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=load_json("docling", "status_success.json"))

        async with self._client(handler) as client:
            await self._wait_bounded(client)

    async def test_ошибка_несёт_причину_из_тела(self):
        """Без разбора task_meta остаётся «завершилась с ошибкой», и
        причину приходится искать в логах чужого контейнера."""
        async def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=load_json("docling", "status_failure.json"))

        async with self._client(handler) as client:
            with self.assertRaises(DoclingError) as ctx:
                await self._wait_bounded(client)
        self.assertIn("URL is not allowed", str(ctx.exception))

    async def test_таймаут_при_вечном_ожидании(self):
        async def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=load_json("docling", "status_pending.json"))

        async with self._client(handler) as client:
            with self.assertRaises(DoclingTimeoutError):
                await self._wait_bounded(client)

    async def test_нулевой_интервал_не_зацикливает(self):
        """Дедлайн по monotonic, а не накопление сна: при poll_interval=0
        накопительный счётчик не рос бы вовсе."""
        async def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"task_status": "started"})

        async with self._client(handler) as client:
            with self.assertRaises(DoclingTimeoutError):
                await self._wait_bounded(client)


class FileClientTests(AsyncFixtureTestCase):
    async def test_форма_данных_плоская(self):
        seen = {}

        async def handler(request: httpx.Request) -> httpx.Response:
            seen["body"] = request.read()
            seen["ctype"] = request.headers.get("content-type", "")
            return httpx.Response(200, json={"task_id": "t"})

        path = self.data_path("archives", "plain.zip")
        async with DoclingFileClient("http://d", client=transport_of(handler)) as client:
            await client._submit(path, ConversionOptions(), ChunkingOptions())

        self.assertIn("multipart/form-data", seen["ctype"])
        self.assertIn(b"chunking_tokenizer", seen["body"])
        self.assertIn(b"plain.zip", seen["body"])
