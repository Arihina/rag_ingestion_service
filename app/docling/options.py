from __future__ import annotations


"""Параметры конверсии и чанкинга."""

from dataclasses import dataclass
from typing import Any, Sequence


@dataclass(frozen=True)
class ConversionOptions:
    """Параметры конверсии. Точные имена полей сверяй с /docs живого сервера."""

    do_ocr: bool = True
    force_ocr: bool = False

    ocr_engine: str | None = "easyocr"
    ocr_lang: Sequence[str] = ("ru", "en")

    ocr_preset: str | None = None
    table_mode: str = "accurate"
    pdf_backend: str | None = None

    def as_form_data(self) -> dict[str, Any]:
        """httpx проверяет data на Mapping. Список кортежей он принимает
        за сырое тело запроса и молча игнорирует files."""
        data: dict[str, Any] = {
            "do_ocr": str(self.do_ocr).lower(),
            "force_ocr": str(self.force_ocr).lower(),
            "table_mode": self.table_mode,
            "ocr_lang": list(self.ocr_lang),
        }
        if self.ocr_engine:
            data["ocr_engine"] = self.ocr_engine
        if self.ocr_preset:
            data["ocr_preset"] = self.ocr_preset
        if self.pdf_backend:
            data["pdf_backend"] = self.pdf_backend
        return data

    def as_json_options(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "do_ocr": self.do_ocr,
            "force_ocr": self.force_ocr,
            "ocr_lang": list(self.ocr_lang),
            "table_mode": self.table_mode,
        }
        if self.ocr_engine:
            out["ocr_engine"] = self.ocr_engine
        if self.ocr_preset:
            out["ocr_preset"] = self.ocr_preset
        if self.pdf_backend:
            out["pdf_backend"] = self.pdf_backend
        return out


@dataclass(frozen=True)
class ChunkingOptions:
    """Параметры встроенного HybridChunker. Уходят с префиксом chunking_."""

    tokenizer: str = "BAAI/bge-m3"

    max_tokens: int = 768

    merge_peers: bool = True

    _PREFIX = "chunking_"

    def as_form_data(self) -> dict[str, Any]:
        return {
            f"{self._PREFIX}tokenizer": self.tokenizer,
            f"{self._PREFIX}max_tokens": str(self.max_tokens),
            f"{self._PREFIX}merge_peers": str(self.merge_peers).lower(),
        }

    def as_json_options(self) -> dict[str, Any]:
        """Для JSON-тела префикса НЕТ: поля лежат внутри chunking_options.

        Префикс chunking_ нужен только form-варианту на /file/async, где
        всё плоское и разделять объекты нечем.
        """
        return {
            "tokenizer": self.tokenizer,
            "max_tokens": self.max_tokens,
            "merge_peers": self.merge_peers,
        }
