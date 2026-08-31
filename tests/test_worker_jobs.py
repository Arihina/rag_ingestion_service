from __future__ import annotations

"""Задачи воркера: порядок операций и режим передачи файла.

Порядок «объект -> строка -> задача» проверяется явно, потому что
обратный ломается тихо: воркер получит задачу на несуществующий файл, и
документ навсегда зависнет в failed.
"""

import uuid
from unittest.mock import AsyncMock, MagicMock, patch

from app.config import settings
from tests.base import FixtureTestCase
from tests.support import AsyncFixtureTestCase


class TransferModeTests(AsyncFixtureTestCase):
    """url: docling качает сам по presigned-ссылке.
    upload: качаем мы и шлём multipart — docling не обязан ни видеть
    хранилище, ни пропускать его через свой allowlist для http_sources."""

    @staticmethod
    def _os_client():
        client = MagicMock()
        client.close = AsyncMock()
        return client

    def _patches(self, mode: str):
        storage = MagicMock()
        storage.presigned_get.return_value = "http://s3/presigned"
        storage.get_bytes.return_value = (b"%PDF-1.4", "application/pdf")

        document = MagicMock()
        document.rag_id = uuid.uuid4()
        document.storage_key = "rag/doc/file.pdf"

        session = AsyncMock()
        session.get = AsyncMock(return_value=document)
        scope = MagicMock()
        scope.return_value.__aenter__ = AsyncMock(return_value=session)
        scope.return_value.__aexit__ = AsyncMock(return_value=False)

        pipeline = MagicMock()
        pipeline.return_value.ingest_source = AsyncMock(return_value=7)

        return {
            "storage": storage,
            "patches": [
                patch.object(settings, "docling_transfer", mode),
                patch("app.tasks.jobs.ObjectStorage",
                      MagicMock(return_value=storage)),
                patch("app.tasks.jobs.session_scope", scope),
                patch("app.tasks.jobs.IngestPipeline", pipeline),
                # close() у клиента OpenSearch awaitable — обычный MagicMock
                # тут падает на await.
                patch("app.tasks.jobs._opensearch",
                      MagicMock(return_value=self._os_client())),
                patch("app.tasks.jobs.HttpEmbedder", MagicMock()),
                patch("app.tasks.jobs.OpenSearchLoader", MagicMock()),
                patch("app.tasks.jobs.repo.mark_document", AsyncMock()),
                patch("app.tasks.jobs.dispose", AsyncMock(), create=True),
            ],
            "pipeline": pipeline,
        }

    async def _run(self, mode: str):
        from app.tasks.jobs import _ingest_document

        ctx = self._patches(mode)
        for p in ctx["patches"]:
            p.start()
        try:
            await _ingest_document(uuid.uuid4())
        finally:
            for p in ctx["patches"]:
                p.stop()
        return ctx

    async def test_режим_url_подписывает_ссылку(self):
        ctx = await self._run("url")
        ctx["storage"].presigned_get.assert_called_once()
        ctx["storage"].get_bytes.assert_not_called()
        source = ctx["pipeline"].return_value.ingest_source.call_args[0][0]
        self.assertEqual(source, "http://s3/presigned")

    async def test_режим_upload_качает_сам(self):
        ctx = await self._run("upload")
        ctx["storage"].get_bytes.assert_called_once()
        ctx["storage"].presigned_get.assert_not_called()
        source = ctx["pipeline"].return_value.ingest_source.call_args[0][0]
        self.assertTrue(str(source).endswith("file.pdf"))

    async def test_временный_файл_убирается(self):
        ctx = await self._run("upload")
        source = ctx["pipeline"].return_value.ingest_source.call_args[0][0]
        self.assertFalse(source.exists(), "временный каталог не убран")


class EmbedderPoolTests(FixtureTestCase):
    """Адрес здесь — внутренний порт (8012), как в реальном развёртывании:
    /embed на платформенном порту отсутствует вовсе."""

    def test_воркер_ходит_в_пул_ingest(self):
        """Иначе воркеры и чат встают в общую очередь к одной карте."""
        from app.embeddings import HttpEmbedder

        embedder = HttpEmbedder("http://api:8012", pool="ingest")
        self.addCleanup(embedder.close)
        self.assertEqual(embedder._pool, "ingest")

    def test_по_умолчанию_пул_query(self):
        from app.embeddings import HttpEmbedder

        embedder = HttpEmbedder("http://api:8012")
        self.addCleanup(embedder.close)
        self.assertEqual(embedder._pool, "query")

    def test_запрос_уходит_на_переданный_адрес(self):
        """Опечатка в SELF_URL даёт 404 на каждой задаче индексации, и
        видно это только в логе воркера."""
        import httpx
        from app.embeddings import HttpEmbedder

        seen = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["url"] = str(request.url)
            return httpx.Response(200, json={
                "model": "BAAI/bge-m3",
                "embeddings": [{"dense": [0.0] * 1024, "sparse": {}}],
            })

        embedder = HttpEmbedder("http://api:8012/")
        embedder.close()
        embedder._client = httpx.Client(
            transport=httpx.MockTransport(handler), timeout=1.0
        )
        try:
            embedder.embed(["текст"])
        finally:
            embedder.close()
        self.assertEqual(seen["url"], "http://api:8012/embed")

    def test_pool_уходит_в_тело_запроса(self):
        import httpx
        from app.embeddings import HttpEmbedder

        seen = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen.update(__import__("json").loads(request.read()))
            return httpx.Response(200, json={
                "model": "BAAI/bge-m3",
                "embeddings": [{"dense": [0.0] * 1024, "sparse": {}}],
            })

        embedder = HttpEmbedder("http://api:8012", pool="ingest")
        # Дефолтный таймаут HttpEmbedder — 120 секунд: промах мимо
        # заглушки подвесил бы прогон, а не уронил его.
        embedder.close()
        embedder._client = httpx.Client(
            transport=httpx.MockTransport(handler), timeout=1.0
        )
        try:
            embedder.embed(["текст"])
        finally:
            embedder.close()
        self.assertEqual(seen["pool"], "ingest")
        self.assertEqual(seen["texts"], ["текст"])
