from __future__ import annotations

"""Изоляция наборов в общем индексе.

Индекс один на всю платформу, и rag_id — граница безопасности, а не
оптимизация: забытый фильтр даёт не деградацию выдачи, а утечку чужих
документов в чужой ответ, причём тихую.
"""

import unittest
from unittest.mock import AsyncMock, MagicMock

from app.search import OpenSearchLoader


class LoaderQueryTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.client = MagicMock()
        self.client.indices.exists = AsyncMock(return_value=True)
        self.client.search = AsyncMock(
            return_value={"hits": {"hits": []}, "_shards": {}}
        )
        self.client.delete_by_query = AsyncMock(return_value={"deleted": 3})
        self.loader = OpenSearchLoader(self.client, "kb-v2")

    async def test_поиск_хэшей_фильтрует_по_набору_и_документу(self):
        await self.loader.existing_hashes("RAG-1", "DOC-1")
        body = self.client.search.call_args.kwargs["body"]
        terms = [list(f["term"].items())[0]
                 for f in body["query"]["bool"]["filter"]]
        self.assertIn(("rag_id", "RAG-1"), terms)
        self.assertIn(("document_id", "DOC-1"), terms)

    async def test_удаление_набора_ограничено_его_rag_id(self):
        await self.loader.delete_rag("RAG-1")
        body = self.client.delete_by_query.call_args.kwargs["body"]
        self.assertEqual(body["query"]["term"]["rag_id"], "RAG-1")

    async def test_удаление_документа_требует_обоих_идентификаторов(self):
        """Только document_id было бы достаточно — UUID уникален, — но
        rag_id стоит дёшево и исключает класс аварий, если UUID когда-нибудь
        начнут генерировать снаружи."""
        await self.loader.delete_document("RAG-1", "DOC-1")
        body = self.client.delete_by_query.call_args.kwargs["body"]
        terms = [list(f["term"].items())[0]
                 for f in body["query"]["bool"]["filter"]]
        self.assertIn(("rag_id", "RAG-1"), terms)
        self.assertIn(("document_id", "DOC-1"), terms)

    async def test_удаление_не_падает_на_конфликтах(self):
        """conflicts=proceed: параллельная переиндексация не должна
        останавливать уборку на полпути."""
        await self.loader.delete_rag("RAG-1")
        self.assertEqual(
            self.client.delete_by_query.call_args.kwargs["params"]["conflicts"],
            "proceed",
        )


class BulkVisibilityTests(unittest.IsolatedAsyncioTestCase):
    """Без refresh документ помечается success и набор показывает ready, а
    запрос в ту же секунду ничего не находит — расхождение живёт
    refresh_interval (30 с) и выглядит как потеря данных."""

    @staticmethod
    def _payload():
        from app.docling import DoclingChunk
        from app.embeddings import EmbeddingResult

        chunk = DoclingChunk(rag_id="R", document_id="D", index=0,
                             text="текст", embed_text="текст")
        return [chunk], [EmbeddingResult(dense=[0.0] * 1024, sparse={})]

    async def test_после_загрузки_идёт_явный_refresh(self):
        client = MagicMock()
        client.indices.refresh = AsyncMock()
        loader = OpenSearchLoader(client, "kb-v2")
        chunks, embeddings = self._payload()

        with unittest.mock.patch(
            "app.search.loader.async_bulk", AsyncMock(return_value=(1, []))
        ):
            await loader.load(chunks, embeddings)

        client.indices.refresh.assert_awaited_once_with(index="kb-v2")

    async def test_bulk_не_ждёт_планового_обновления(self):
        """refresh="wait_for" ждал бы до refresh_interval, то есть до
        30 секунд на КАЖДОМ документе, и клиент отваливался бы по таймауту
        раньше, чем дождался."""
        client = MagicMock()
        client.indices.refresh = AsyncMock()
        loader = OpenSearchLoader(client, "kb-v2")
        chunks, embeddings = self._payload()

        with unittest.mock.patch(
            "app.search.loader.async_bulk", AsyncMock(return_value=(1, []))
        ) as bulk:
            await loader.load(chunks, embeddings)

        self.assertNotIn("refresh", bulk.call_args.kwargs)

    async def test_пустой_bulk_не_дёргает_refresh(self):
        client = MagicMock()
        client.indices.refresh = AsyncMock()
        loader = OpenSearchLoader(client, "kb-v2")
        chunks, embeddings = self._payload()

        with unittest.mock.patch(
            "app.search.loader.async_bulk", AsyncMock(return_value=(0, []))
        ):
            await loader.load(chunks, embeddings)

        client.indices.refresh.assert_not_awaited()

    async def test_refresh_отключается_настройкой(self):
        """На массовом импорте сегмент на документ становится дорогим."""
        from app.config import settings

        client = MagicMock()
        client.indices.refresh = AsyncMock()
        loader = OpenSearchLoader(client, "kb-v2")
        chunks, embeddings = self._payload()

        with unittest.mock.patch.object(settings, "refresh_after_load", False), \
            unittest.mock.patch(
            "app.search.loader.async_bulk", AsyncMock(
                return_value=(1, []))
        ):
            await loader.load(chunks, embeddings)

        client.indices.refresh.assert_not_awaited()

    async def test_несовпадение_длин_ловится_рано(self):
        loader = OpenSearchLoader(MagicMock(), "kb-v2")
        with self.assertRaises(ValueError):
            await loader.load([MagicMock()], [])
