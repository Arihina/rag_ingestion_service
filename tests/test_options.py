from __future__ import annotations

"""Опции docling.

Цена ошибки в именах полей высокая: pydantic на стороне docling-serve по
умолчанию МОЛЧА отбрасывает неизвестные поля. Промахнулся — настройки не
применились, ошибки нет, результат другой.
"""

import unittest

from app.docling import ChunkingOptions, ConversionOptions


class ConversionOptionsTests(unittest.TestCase):
    def test_json_keeps_types(self):
        """В JSON булевы остаются булевыми, а язык — списком."""
        out = ConversionOptions(do_ocr=True, force_ocr=True).as_json_options()
        self.assertIs(out["do_ocr"], True)
        self.assertIs(out["force_ocr"], True)
        self.assertEqual(out["ocr_lang"], ["ru", "en"])

    def test_form_stringifies(self):
        """multipart не знает типов — всё уезжает строками в нижнем регистре."""
        out = ConversionOptions(do_ocr=True, force_ocr=False).as_form_data()
        self.assertEqual(out["do_ocr"], "true")
        self.assertEqual(out["force_ocr"], "false")

    def test_optional_fields_omitted_not_nulled(self):
        """None в теле — это не «не задано», а явный null: docling может
        принять его за сброс значения."""
        out = ConversionOptions(ocr_engine=None, ocr_preset=None,
                                pdf_backend=None).as_json_options()
        for key in ("ocr_engine", "ocr_preset", "pdf_backend"):
            self.assertNotIn(key, out)


class ChunkingOptionsTests(unittest.TestCase):
    def test_json_has_no_prefix(self):
        """В JSON поля лежат внутри chunking_options — префикс не нужен."""
        out = ChunkingOptions().as_json_options()
        self.assertEqual(set(out), {"tokenizer", "max_tokens", "merge_peers"})
        self.assertFalse(any(k.startswith("chunking_") for k in out))

    def test_form_has_prefix(self):
        """На /file/async всё плоское, разделять объекты нечем — отсюда префикс."""
        out = ChunkingOptions().as_form_data()
        self.assertTrue(all(k.startswith("chunking_") for k in out))

    def test_tokenizer_defaults_to_embedding_model(self):
        """Токенизатор чанкера обязан совпадать с эмбеддером, иначе границы
        чанков считаются по чужому словарю."""
        self.assertEqual(ChunkingOptions().tokenizer, "BAAI/bge-m3")
