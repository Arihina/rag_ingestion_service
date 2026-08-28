from __future__ import annotations

"""Вспомогательное для тестов: транспорт httpx без сети."""

import unittest
from typing import Awaitable, Callable

import httpx

from tests.base import data_path, load_bytes, load_json

Handler = Callable[[httpx.Request], Awaitable[httpx.Response]]


def transport_of(handler) -> httpx.AsyncClient:
    """Клиент, отвечающий заданной функцией вместо реального запроса.

    MockTransport, а не патч метода: так проверяется настоящее тело
    запроса ровно в том виде, в каком httpx его сериализует, включая
    multipart-границы и заголовки.
    """

    return httpx.AsyncClient(
        transport=httpx.MockTransport(handler), timeout=1.0
    )


class AsyncFixtureTestCase(unittest.IsolatedAsyncioTestCase):
    data_path = staticmethod(data_path)
    load_json = staticmethod(load_json)
    load_bytes = staticmethod(load_bytes)
