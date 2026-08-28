from __future__ import annotations

"""Разбор пользовательского zip.

Ни одна проверка здесь не про качество поиска — все про то, что архив
приходит от пользователя. Ключевой принцип: лимиты считаются ПО
ЦЕНТРАЛЬНОМУ КАТАЛОГУ, до распаковки. Проверять постфактум бессмысленно —
к тому моменту бомба уже развернулась.
"""

import zipfile
from unittest.mock import patch

from app.archive import ArchiveRejected, plan, safe_name
from app.config import settings
from tests.base import FixtureTestCase, data_path


def plan_of(name: str):
    with zipfile.ZipFile(data_path("archives", name)) as archive:
        return plan(archive)


class SafeNameTests(FixtureTestCase):
    def test_сплющивает_путь(self):
        self.assertEqual(safe_name("sub/dir/c.md"), "c.md")

    def test_обезвреживает_traversal(self):
        self.assertEqual(safe_name("../../etc/passwd.txt"), "passwd.txt")

    def test_понимает_обратные_слэши(self):
        """Архивы из Windows приходят с \\ — normpath их не тронет."""
        self.assertEqual(
            safe_name("..\\..\\windows\\system.txt"), "system.txt")

    def test_пустое_имя_отклоняется(self):
        for raw in ("", "..", "./", "/", "../.."):
            with self.subTest(raw=raw), self.assertRaises(ArchiveRejected):
                safe_name(raw)


class ArchivePlanTests(FixtureTestCase):
    def test_обычный_архив_разбирается(self):
        result = plan_of("plain.zip")
        self.assertEqual({e.name for e in result.entries},
                         {"a.txt", "b.pdf", "c.md"})
        self.assertEqual(result.skipped, [])

    def test_вложенные_каталоги_сплющиваются(self):
        """sub/dir/c.md -> c.md: имя уезжает в storage_key, путь там не нужен."""
        names = [e.name for e in plan_of("plain.zip").entries]
        self.assertNotIn("sub/dir/c.md", names)
        self.assertIn("c.md", names)

    def test_traversal_обезврежен(self):
        names = [e.name for e in plan_of("traversal.zip").entries]
        self.assertEqual(
            sorted(names), ["normal.txt", "passwd.txt", "system.txt"])
        self.assertFalse(
            any(".." in n or "/" in n or "\\" in n for n in names))

    def test_коллизия_имён_разводится(self):
        """a/doc.txt, b/doc.txt, c/doc.txt после сплющивания совпадают.
        Содержимое разное, поэтому терять их нельзя."""
        names = [e.name for e in plan_of("collision.zip").entries]
        self.assertEqual(names, ["doc.txt", "doc_1.txt", "doc_2.txt"])
        self.assertEqual(len(set(names)), 3)

    def test_чужое_расширение_пропускается_без_отказа(self):
        """Одна неподходящая запись не должна валить весь батч."""
        result = plan_of("mixed.zip")
        self.assertEqual({e.name for e in result.entries},
                         {"good.txt", "photo.png"})
        skipped = dict(result.skipped)
        self.assertIn("script.exe", skipped)
        self.assertIn("noext", skipped)

    def test_вложенный_архив_пропускается(self):
        """Рекурсивная распаковка — ещё один вектор бомбы."""
        skipped = dict(plan_of("mixed.zip").skipped)
        self.assertIn("inner.zip", skipped)
        self.assertIn("вложенный", skipped["inner.zip"])

    def test_пустой_архив_не_падает(self):
        result = plan_of("empty.zip")
        self.assertEqual(result.entries, [])

    def test_размеры_берутся_из_каталога_а_не_из_распаковки(self):
        """plan() не должен читать содержимое: на бомбе это и есть защита."""
        with zipfile.ZipFile(data_path("archives", "plain.zip")) as archive:
            with patch.object(archive, "read", side_effect=AssertionError("читать нельзя")):
                result = plan(archive)
        self.assertEqual(len(result.entries), 3)
        self.assertTrue(all(e.size > 0 for e in result.entries))


class ArchiveRejectionTests(FixtureTestCase):
    """Случаи, когда архив отклоняется ЦЕЛИКОМ."""

    def test_бомба_по_коэффициенту_сжатия(self):
        with self.assertRaises(ArchiveRejected) as ctx:
            plan_of("bomb.zip")
        self.assertIn("бомб", str(ctx.exception).lower())

    def test_слишком_много_записей(self):
        with self.assertRaises(ArchiveRejected) as ctx:
            plan_of("too_many.zip")
        self.assertIn(str(settings.archive_max_entries), str(ctx.exception))

    def test_зашифрованный_архив(self):
        with self.assertRaises(ArchiveRejected) as ctx:
            plan_of("encrypted.zip")
        self.assertIn("зашифрован", str(ctx.exception).lower())

    def test_лимит_суммарного_размера(self):
        with patch.object(settings, "archive_max_uncompressed", 10):
            with self.assertRaises(ArchiveRejected):
                plan_of("plain.zip")

    def test_лимит_на_один_файл_пропускает_запись(self):
        """Большой файл — это пропуск записи, а не отказ архива:
        отклонять целиком имеет смысл только для бомбы."""
        with patch.object(settings, "archive_max_file_size", 100):
            result = plan_of("plain.zip")
        self.assertTrue(result.skipped)
        self.assertLess(len(result.entries), 3)
