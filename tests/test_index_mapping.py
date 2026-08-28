from __future__ import annotations

"""Маппинг kb-v2. Ошибки здесь не дают исключений — они портят выдачу."""

import unittest

from app.search import EMBED_DIM, INDEX_BODY, SEARCH_SOURCE_EXCLUDES


class IndexMappingTests(unittest.TestCase):
    @property
    def props(self) -> dict:
        return INDEX_BODY["mappings"]["properties"]

    def test_rag_id_есть_и_не_анализируется(self):
        """term-фильтр по rag_id — граница между наборами."""
        self.assertEqual(self.props["rag_id"]["type"], "keyword")

    def test_document_id_не_анализируется(self):
        self.assertEqual(self.props["document_id"]["type"], "keyword")

    def test_source_uri_выведен_из_схемы(self):
        self.assertNotIn("source_uri", self.props)

    def test_headings_анализируется(self):
        """Как keyword поле не разбирается на термы, и BM25 по заголовкам
        вырождается в точное совпадение всей цепочки целиком."""
        self.assertEqual(self.props["headings"]["type"], "text")
        self.assertEqual(self.props["headings"]["analyzer"], "ru_en")

    def test_content_анализируется_русской_морфологией(self):
        self.assertEqual(self.props["content"]["analyzer"], "ru_en")

    def test_размерность_вектора_совпадает_с_bge_m3(self):
        self.assertEqual(EMBED_DIM, 1024)
        self.assertEqual(self.props["content_vector"]["dimension"], EMBED_DIM)

    def test_вектор_на_косинусной_метрике(self):
        method = self.props["content_vector"]["method"]
        self.assertEqual(method["space_type"], "cosinesimil")

    def test_sparse_ветка_на_rank_features(self):
        self.assertEqual(self.props["content_sparse"]["type"], "rank_features")

    def test_content_hash_есть_для_skip_unchanged(self):
        self.assertEqual(self.props["content_hash"]["type"], "keyword")

    def test_тяжёлые_поля_исключены_из_выдачи(self):
        """Возвращать 1024 float на чанк в _source незачем."""
        self.assertIn("content_vector", SEARCH_SOURCE_EXCLUDES)
        self.assertIn("content_sparse", SEARCH_SOURCE_EXCLUDES)

    def test_анализатор_ru_en_объявлен(self):
        analysis = INDEX_BODY["settings"]["analysis"]
        self.assertIn("ru_en", analysis["analyzer"])
        self.assertIn("russian_stemmer", analysis["filter"])

    def test_knn_включён(self):
        self.assertTrue(INDEX_BODY["settings"]["index"]["knn"])

    def test_refresh_interval_осознанно_растянут(self):
        """30 с вместо секунды — под bulk-загрузку.

        Из-за этого запись НЕ становится видимой сразу, и загрузчик
        обязан дожидаться refresh перед тем, как пометить документ
        success: иначе набор показывает ready, а поиск ничего не находит.
        См. refresh="wait_for" в OpenSearchLoader.load.
        """
        self.assertEqual(INDEX_BODY["settings"]
                         ["index"]["refresh_interval"], "30s")


class ClientTimeoutTests(unittest.TestCase):
    def test_таймаут_задан_явно_и_больше_refresh_interval(self):
        """Дефолт opensearch-py — 10 секунд, чего не хватает на bulk из
        сотен чанков с векторами по 1024 измерения."""
        from app.config import settings
        from app.search import INDEX_BODY

        interval = INDEX_BODY["settings"]["index"]["refresh_interval"]
        seconds = int(interval.rstrip("s"))
        self.assertGreater(settings.opensearch_timeout, seconds)
        self.assertGreaterEqual(settings.opensearch_timeout, 60)
