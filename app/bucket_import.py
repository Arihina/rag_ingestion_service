from __future__ import annotations

"""Отбор объектов чужого бакета под импорт."""

import hashlib
import logging
from dataclasses import dataclass

from app.config import settings
from app.storage import RemoteObject

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Rejected:
    source_key: str
    reason: str


@dataclass(frozen=True)
class Selection:
    accepted: list[RemoteObject]
    rejected: list[Rejected]

    @property
    def total(self) -> int:
        return len(self.accepted) + len(self.rejected)


def normalize_extensions(requested: list[str]) -> set[str]:
    """Привести к виду ".pdf" и пересечь с тем, что умеет docling."""
    allowed = set(settings.allowed_suffixes)
    if not requested:
        return allowed
    asked = {
        ("." + ext.lower().lstrip(".")) for ext in requested if ext.strip(". ")
    }
    return asked & allowed


def content_hash_for(bucket: str, obj: RemoteObject) -> str:
    """Ключ дедупликации для объекта, который не читали."""
    key = f"{bucket}:{obj.etag}".encode("utf-8")
    return hashlib.blake2b(key, digest_size=16).hexdigest()


def select(bucket: str, objects: list[RemoteObject], extensions: list[str]) -> Selection:
    """Разложить листинг на принятое и отклонённое."""
    suffixes = normalize_extensions(extensions)
    accepted: list[RemoteObject] = []
    rejected: list[Rejected] = []
    seen: set[str] = set()

    for obj in objects:
        if not obj.suffix:
            rejected.append(Rejected(obj.key, "нет расширения"))
        elif obj.suffix not in suffixes:
            rejected.append(
                Rejected(obj.key, f"расширение {obj.suffix} не поддерживается"))
        elif obj.size == 0:
            rejected.append(Rejected(obj.key, "пустой файл"))
        elif obj.size > settings.bucket_import_max_file_size:
            rejected.append(
                Rejected(
                    obj.key, f"размер > {settings.bucket_import_max_file_size} байт")
            )
        elif len(accepted) >= settings.bucket_import_max_files:
            rejected.append(
                Rejected(
                    obj.key, f"превышен лимит в {settings.bucket_import_max_files} файлов")
            )
        else:
            digest = content_hash_for(bucket, obj)
            if digest in seen:
                rejected.append(Rejected(obj.key, "дубль внутри префикса"))
                continue
            seen.add(digest)
            accepted.append(obj)

    return Selection(accepted=accepted, rejected=rejected)


def drop_known(
    bucket: str, selection: Selection, known_hashes: set[str]
) -> Selection:
    """Отсеять то, что уже лежит в наборе."""
    if not known_hashes:
        return selection

    accepted: list[RemoteObject] = []
    rejected = list(selection.rejected)
    for obj in selection.accepted:
        if content_hash_for(bucket, obj) in known_hashes:
            rejected.append(Rejected(obj.key, "уже в наборе"))
        else:
            accepted.append(obj)
    return Selection(accepted=accepted, rejected=rejected)
