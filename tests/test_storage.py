from __future__ import annotations

"""Объектное хранилище: ключи, подпись ссылок, потоковая запись."""

import asyncio
import unittest
from unittest.mock import MagicMock, patch

from app.config import settings
from app.storage import document_key, stream_upload
from app.storage.seaweed import ObjectStorage


class DocumentKeyTests(unittest.TestCase):
    def test_layout(self):
        key = document_key("rag-1", "doc-2", "отчёт.pdf")
        self.assertEqual(key, "rag-1/doc-2/отчёт.pdf")

    def test_prefix_allows_wiping_whole_set(self):
        """Удаление набора — снос префикса, поэтому rag_id обязан быть первым."""
        keys = [document_key("R", f"d{i}", "f.pdf") for i in range(3)]
        self.assertTrue(all(k.startswith("R/") for k in keys))


class PresignEndpointTests(unittest.TestCase):
    """Ссылку скачивает docling, а не мы: подписывать её надо под тем
    хостом, по которому за ней придут. Подпись SigV4 накрывает Host,
    поэтому подменить адрес постфактум нельзя."""

    def _storage(self, endpoint: str, presign: str | None):
        with patch.object(settings, "s3_endpoint_url", endpoint), \
                patch.object(settings, "s3_presign_endpoint_url", presign), \
                patch("app.storage.seaweed.boto3.client") as factory:
            factory.side_effect = lambda *a, **kw: MagicMock(
                name=kw["endpoint_url"])
            storage = ObjectStorage()
            return storage, factory

    def test_same_endpoint_reuses_single_client(self):
        storage, factory = self._storage("http://localhost:8333", None)
        self.assertIs(storage._client, storage._presign_client)
        self.assertEqual(factory.call_count, 1)

    def test_different_endpoint_creates_second_client(self):
        storage, factory = self._storage(
            "http://localhost:8333", "http://seaweedfs:8333")
        self.assertIsNot(storage._client, storage._presign_client)
        endpoints = [c.kwargs["endpoint_url"] for c in factory.call_args_list]
        self.assertEqual(
            endpoints, ["http://localhost:8333", "http://seaweedfs:8333"])

    def test_presign_uses_presign_client(self):
        storage, _ = self._storage(
            "http://localhost:8333", "http://seaweedfs:8333")
        storage.presigned_get("rag-docs", "a/b.pdf")
        storage._presign_client.generate_presigned_url.assert_called_once()
        storage._client.generate_presigned_url.assert_not_called()


class StreamUploadTests(unittest.IsolatedAsyncioTestCase):
    @staticmethod
    async def _chunks(*parts: bytes):
        for part in parts:
            yield part

    async def test_hash_and_size_computed_on_the_fly(self):
        storage = MagicMock()
        stored = await stream_upload(
            storage, "bucket", "key", self._chunks(b"abc", b"def"), limit_bytes=100
        )
        self.assertEqual(stored.size_bytes, 6)
        self.assertEqual(len(stored.content_hash), 32)
        storage.put_stream.assert_called_once()

    async def test_same_content_same_hash(self):
        """Дедупликация в наборе держится на этом хэше."""
        storage = MagicMock()
        one = await stream_upload(storage, "b", "k1", self._chunks(b"data"), limit_bytes=100)
        two = await stream_upload(storage, "b", "k2", self._chunks(b"da", b"ta"), limit_bytes=100)
        self.assertEqual(one.content_hash, two.content_hash)

    async def test_limit_enforced_mid_stream(self):
        """Проверка по ходу, а не после: иначе лимит бесполезен."""
        storage = MagicMock()
        with self.assertRaises(ValueError):
            await stream_upload(
                storage, "b", "k", self._chunks(b"x" * 10, b"x" * 10), limit_bytes=15
            )
        storage.put_stream.assert_not_called()
