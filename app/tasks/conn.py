from __future__ import annotations

"""Подключение к Redis и очереди RQ."""

from functools import lru_cache

from redis import Redis
from rq import Queue

from app.config import settings


@lru_cache(maxsize=1)
def redis_conn() -> Redis:
    return Redis.from_url(settings.redis_url)


@lru_cache(maxsize=None)
def queue(name: str) -> Queue:
    return Queue(name, connection=redis_conn(), default_timeout=settings.rq_job_timeout)


def ingest_queue() -> Queue:
    return queue(settings.rq_queue_ingest)


def maintenance_queue() -> Queue:
    """Разворачивание архивов, импорт и уборка.

    Отдельная очередь, чтобы разворачивание архива на триста файлов не
    стояло в одной очереди с обработкой этих же файлов и не выглядело
    как зависший импорт.
    """
    return queue(settings.rq_queue_maintenance)
