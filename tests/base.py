from __future__ import annotations

"""Общая база для тестов: путь к фикстурам и загрузчики.

Данные в tests/data создаются генератором (tests/fixtures_build.py) при
первом запуске, а не коммитятся бинарниками: тесты проверяют не байты, а
структуру и диапазоны, зато набор воспроизводится на чистой машине одной
командой.
"""

import json
import unittest
from pathlib import Path

from tests import fixtures_build

DATA = Path(__file__).parent / "data"


fixtures_build.ensure()


def data_path(*parts: str) -> Path:
    path = DATA.joinpath(*parts)
    if not path.exists():
        raise FileNotFoundError(f"Нет фикстуры {path}")
    return path


def load_json(*parts: str) -> dict:
    return json.loads(data_path(*parts).read_text(encoding="utf-8"))


def load_bytes(*parts: str) -> bytes:
    return data_path(*parts).read_bytes()


class FixtureTestCase(unittest.TestCase):
    """Тесты, работающие с файлами из tests/data."""

    data_path = staticmethod(data_path)
    load_json = staticmethod(load_json)
    load_bytes = staticmethod(load_bytes)
