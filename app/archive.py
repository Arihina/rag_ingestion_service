from __future__ import annotations

"""Разбор пользовательского zip.
Ключевой принцип: первые три проверки идут ДО распаковки, по центральному
каталогу zip, и нарушение любой отклоняет архив целиком.
"""

import logging
import posixpath
import zipfile
from dataclasses import dataclass
from pathlib import PurePosixPath

from app.config import settings

logger = logging.getLogger(__name__)


class ArchiveRejected(ValueError):
    """Архив отклонён целиком."""


@dataclass(frozen=True)
class ArchiveEntry:
    name: str
    size: int
    raw_name: str


@dataclass(frozen=True)
class ArchivePlan:
    entries: list[ArchiveEntry]
    skipped: list[tuple[str, str]]


_NESTED_ARCHIVE_SUFFIXES = {".zip", ".tar",
                            ".gz", ".tgz", ".bz2", ".xz", ".7z", ".rar"}


def safe_name(raw: str) -> str:
    """Сплющить путь и отрезать всё, что уводит из каталога."""
    normalized = raw.replace("\\", "/")
    name = PurePosixPath(posixpath.normpath(normalized)).name
    if not name or name in (".", ".."):
        raise ArchiveRejected(f"Некорректное имя записи: {raw!r}")
    return name


def plan(archive: zipfile.ZipFile) -> ArchivePlan:
    """Проверить архив и решить, что из него брать. Не распаковывает."""
    infos = [i for i in archive.infolist() if not i.is_dir()]

    if archive.namelist() and any(
        getattr(i, "flag_bits", 0) & 0x1 for i in archive.infolist()
    ):
        raise ArchiveRejected(
            "Архив зашифрован. Распакуйте его и загрузите файлы напрямую."
        )

    if len(infos) > settings.archive_max_entries:
        raise ArchiveRejected(
            f"В архиве {len(infos)} файлов, максимум {settings.archive_max_entries}"
        )

    total_uncompressed = sum(i.file_size for i in infos)
    if total_uncompressed > settings.archive_max_uncompressed:
        raise ArchiveRejected(
            f"Распакованный размер {total_uncompressed} байт превышает "
            f"{settings.archive_max_uncompressed}"
        )

    for info in infos:
        if info.compress_size <= 0:
            continue
        ratio = info.file_size / info.compress_size
        if ratio > settings.archive_max_ratio:
            raise ArchiveRejected(
                f"Запись {info.filename!r} сжата в {ratio:.0f} раз — "
                f"предел {settings.archive_max_ratio:.0f}. Похоже на zip-бомбу."
            )

    entries: list[ArchiveEntry] = []
    skipped: list[tuple[str, str]] = []
    seen: set[str] = set()

    for info in infos:
        try:
            name = safe_name(info.filename)
        except ArchiveRejected as exc:
            skipped.append((info.filename, str(exc)))
            continue

        suffix = PurePosixPath(name).suffix.lower()

        if suffix in _NESTED_ARCHIVE_SUFFIXES:
            skipped.append((name, "вложенный архив не разворачивается"))
            continue

        if suffix not in settings.allowed_suffixes:
            skipped.append(
                (name, f"расширение {suffix or '<нет>'} не поддерживается"))
            continue

        if info.file_size > settings.archive_max_file_size:
            skipped.append(
                (name, f"файл больше {settings.archive_max_file_size} байт"))
            continue

        unique = name
        counter = 1
        while unique in seen:
            stem = PurePosixPath(name).stem
            unique = f"{stem}_{counter}{suffix}"
            counter += 1
        seen.add(unique)

        entries.append(ArchiveEntry(
            name=unique, size=info.file_size, raw_name=info.filename))

    return ArchivePlan(entries=entries, skipped=skipped)
