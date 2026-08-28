from __future__ import annotations

"""Идентичность чанка — граница безопасности, а не деталь реализации.

Прежняя формула blake2b(source_uri:chunk_index) в общем индексе была
коллизионной: report.pdf у двух пользователей давал один _id, операция
index (а не create) молча перезаписывала чужие чанки. Эти тесты
существуют, чтобы такая правка не прошла незамеченной.
"""

import unittest

from app.docling import DoclingChunk


def chunk(**kw) -> DoclingChunk:
    base = dict(
        rag_id="11111111-1111-1111-1111-111111111111",
        document_id="22222222-2222-2222-2222-222222222222",
        index=0,
        text="текст",
        embed_text="текст",
    )
    base.update(kw)
    return DoclingChunk(**base)


class ChunkIdentityTests(unittest.TestCase):
    def test_разные_наборы_дают_разные_ключи(self):
        a = chunk(rag_id="rag-a")
        b = chunk(rag_id="rag-b")
        self.assertNotEqual(a.chunk_id, b.chunk_id)

    def test_разные_документы_дают_разные_ключи(self):
        a = chunk(document_id="doc-a")
        b = chunk(document_id="doc-b")
        self.assertNotEqual(a.chunk_id, b.chunk_id)

    def test_разные_позиции_дают_разные_ключи(self):
        self.assertNotEqual(chunk(index=0).chunk_id, chunk(index=1).chunk_id)

    def test_ключ_не_зависит_от_содержимого(self):
        """Иначе правка документа плодила бы дубли вместо перезаписи."""
        self.assertEqual(
            chunk(text="раз", embed_text="раз").chunk_id,
            chunk(text="два", embed_text="два").chunk_id,
        )

    def test_ключ_детерминирован(self):
        self.assertEqual(chunk().chunk_id, chunk().chunk_id)

    def test_одинаковое_имя_файла_в_разных_наборах_не_коллизия(self):
        """Тот самый сценарий, на котором ломалась старая формула."""
        keys = {
            chunk(rag_id=f"rag-{i}", document_id=f"doc-{i}", index=n).chunk_id
            for i in range(2)
            for n in range(5)
        }
        self.assertEqual(len(keys), 10)

    def test_хэш_содержимого_меняется_вместе_с_текстом(self):
        self.assertNotEqual(
            chunk(text="раз").content_hash, chunk(text="два").content_hash
        )

    def test_хэш_содержимого_не_зависит_от_набора(self):
        """skip_unchanged сравнивает содержимое, а не расположение."""
        self.assertEqual(
            chunk(rag_id="a", text="одно и то же").content_hash,
            chunk(rag_id="b", text="одно и то же").content_hash,
        )


class ChunkSourceTests(unittest.TestCase):
    def test_source_содержит_границу_безопасности(self):
        src = chunk().to_source()
        self.assertIn("rag_id", src)
        self.assertIn("document_id", src)

    def test_source_не_содержит_source_uri(self):
        """Поле выведено из схемы: имя файла живёт в Postgres."""
        self.assertNotIn("source_uri", chunk().to_source())

    def test_source_несёт_провенанс(self):
        src = chunk(headings=("Раздел", "Пункт"), pages=(3, 4)).to_source()
        self.assertEqual(src["headings"], ["Раздел", "Пункт"])
        self.assertEqual(src["pages"], [3, 4])
