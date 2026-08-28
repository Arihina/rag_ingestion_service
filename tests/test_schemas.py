from __future__ import annotations

"""Контракты платформенного API.

Потолок top_k — не вкусовщина: пул растёт как
top_k x 3 ветки x N вариантов multi-query x итерации, и он же ограничивает
контекст, уходящий в eval агентного цикла.
"""

import unittest

from pydantic import ValidationError

from app.config import settings
from app.schemas.rags import RagConfig, RagCreate, RagUpdate


class RagCreateValidationTests(unittest.TestCase):
    def test_минимальный_набор_полей(self):
        rag = RagCreate(name="Регламенты")
        self.assertEqual(rag.temperature, settings.rag_default_temperature)
        self.assertEqual(rag.top_k, settings.rag_default_top_k)
        self.assertEqual(rag.score_threshold,
                         settings.rag_default_score_threshold)

    def test_имя_обязательно(self):
        with self.assertRaises(ValidationError):
            RagCreate()

    def test_пустое_имя_отклоняется(self):
        with self.assertRaises(ValidationError):
            RagCreate(name="")

    def test_top_k_имеет_потолок(self):
        RagCreate(name="ok", top_k=settings.rag_top_k_max)
        with self.assertRaises(ValidationError):
            RagCreate(name="bad", top_k=settings.rag_top_k_max + 1)

    def test_top_k_не_меньше_единицы(self):
        with self.assertRaises(ValidationError):
            RagCreate(name="bad", top_k=0)

    def test_температура_в_пределах_ноль_один(self):
        RagCreate(name="ok", temperature=0.0)
        RagCreate(name="ok", temperature=1.0)
        for bad in (-0.1, 1.1, 3.0):
            with self.subTest(bad=bad), self.assertRaises(ValidationError):
                RagCreate(name="bad", temperature=bad)

    def test_порог_в_терминах_косинуса(self):
        RagCreate(name="ok", score_threshold=0.0)
        RagCreate(name="ok", score_threshold=1.0)
        with self.assertRaises(ValidationError):
            RagCreate(name="bad", score_threshold=1.5)

    def test_промпт_ограничен_по_длине(self):
        limit = settings.rag_prompt_max_chars
        RagCreate(name="ok", prompt="я" * limit)
        with self.assertRaises(ValidationError):
            RagCreate(name="bad", prompt="я" * (limit + 1))


class RagUpdateTests(unittest.TestCase):
    def test_все_поля_необязательны(self):
        self.assertEqual(RagUpdate().model_dump(exclude_unset=True), {})

    def test_exclude_unset_не_затирает_непереданное(self):
        """Иначе PATCH одного поля обнулял бы остальные."""
        payload = RagUpdate(temperature=0.1).model_dump(exclude_unset=True)
        self.assertEqual(payload, {"temperature": 0.1})

    def test_явный_null_отличается_от_непереданного(self):
        payload = RagUpdate(prompt=None).model_dump(exclude_unset=True)
        self.assertEqual(payload, {"prompt": None})

    def test_валидация_та_же_что_при_создании(self):
        with self.assertRaises(ValidationError):
            RagUpdate(top_k=settings.rag_top_k_max + 1)


class RagConfigTests(unittest.TestCase):
    def test_конфиг_целиком_query_time(self):
        """Ни одно поле не влияет на индексацию — значит переиндексация
        как операция не нужна вообще."""
        self.assertEqual(
            set(RagConfig.model_fields),
            {"prompt", "temperature", "top_k", "score_threshold"},
        )
