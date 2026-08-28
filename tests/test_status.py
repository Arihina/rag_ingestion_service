from __future__ import annotations

"""Статус набора — производная от статусов документов, а не хранимое поле.

Конфиг целиком query-time, переиндексации не бывает, поэтому машина
состояний выродилась бы в одно состояние.
"""

import unittest

from app.db.repo import DocumentCounts


class DerivedStatusTests(unittest.TestCase):
    def test_пустой_набор(self):
        self.assertEqual(DocumentCounts().status, "empty")

    def test_только_в_работе(self):
        self.assertEqual(DocumentCounts(
            total=3, pending=3).status, "ingesting")

    def test_частичная_готовность_это_ready(self):
        """Сознательно: чат уже осмыслен на том, что проиндексировано."""
        counts = DocumentCounts(total=3, ready=1, pending=2)
        self.assertEqual(counts.status, "ready")

    def test_ready_несмотря_на_упавшие(self):
        counts = DocumentCounts(total=3, ready=2, failed=1)
        self.assertEqual(counts.status, "ready")

    def test_все_упали(self):
        self.assertEqual(DocumentCounts(total=2, failed=2).status, "failed")

    def test_has_pending_подсказывает_что_список_пополняется(self):
        self.assertTrue(DocumentCounts(
            total=3, ready=1, pending=2).has_pending)
        self.assertFalse(DocumentCounts(total=1, ready=1).has_pending)

    def test_статус_не_бывает_пустым(self):
        for counts in (
            DocumentCounts(),
            DocumentCounts(total=1, pending=1),
            DocumentCounts(total=1, ready=1),
            DocumentCounts(total=1, failed=1),
            DocumentCounts(total=9, ready=3, failed=3, pending=3),
        ):
            with self.subTest(counts=counts):
                self.assertIn(
                    counts.status, {"empty", "ingesting", "ready", "failed"}
                )

    def test_счётчики_неизменяемы(self):
        counts = DocumentCounts(total=1)
        with self.assertRaises(Exception):
            counts.total = 2
