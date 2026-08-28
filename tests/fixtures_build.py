from __future__ import annotations

"""Генерация фикстур в tests/data.

Вызывается автоматически при импорте tests.base, если данных нет. Файлы
детерминированные: одно и то же содержимое при каждом запуске.
"""

import json
import pathlib
import struct
import zipfile
import zlib

DATA = pathlib.Path(__file__).parent / "data"


def _zip(name: str, entries: list[tuple[str, bytes | str]]) -> None:
    path = DATA / "archives" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for arcname, payload in entries:
            archive.writestr(arcname, payload)


def _png(width: int = 8, height: int = 8) -> bytes:
    """Минимальный валидный RGBA-PNG без внешних зависимостей."""

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + tag
            + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        )

    raw = b"".join(b"\0" + bytes((200, 30, 30, 255))
                   * width for _ in range(height))
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw))
        + chunk(b"IEND", b"")
    )


def _encrypted_zip(path: pathlib.Path) -> None:
    """zipfile не умеет писать шифрованные архивы, поэтому взводим флаг
    вручную: разбор архива смотрит именно на бит 0 общих флагов."""
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("secret.txt", "содержимое под паролем")
    raw = bytearray(path.read_bytes())

    for signature, offset in ((b"PK\x03\x04", 6), (b"PK\x01\x02", 8)):
        index = raw.find(signature)
        while index != -1:
            raw[index + offset] |= 0x01
            index = raw.find(signature, index + 1)
    path.write_bytes(bytes(raw))


def build() -> None:
    """Создать все фикстуры заново, перезаписав существующие."""
    for sub in ("archives", "docling", "icons"):
        (DATA / sub).mkdir(parents=True, exist_ok=True)

    _zip("plain.zip", [
        ("a.txt", "Инструкция по монтажу щита.\n" * 20),
        ("b.pdf", b"%PDF-1.4 fake\n" + b"x" * 500),
        ("sub/dir/c.md", "# Вложенный\nТекст.\n" * 20),
    ])
    _zip("traversal.zip", [
        ("../../etc/passwd.txt", "root:x:0:0\n"),
        ("..\\..\\windows\\system.txt", "boot\n"),
        ("normal.txt", "обычный файл\n"),
    ])
    _zip("collision.zip", [
        ("a/doc.txt", "содержимое A\n"),
        ("b/doc.txt", "содержимое B\n"),
        ("c/doc.txt", "содержимое C\n"),
    ])
    # 20 МБ нулей: коэффициент сжатия в сотни раз при любом zlib
    _zip("bomb.zip", [("bomb.txt", b"\0" * (20 * 1024 * 1024))])
    _zip("mixed.zip", [
        ("inner.zip", b"PK\x03\x04fake"),
        ("script.exe", b"MZfake"),
        ("photo.png", _png()),
        ("good.txt", "нормальный документ\n"),
        ("noext", "без расширения"),
    ])
    _zip("too_many.zip", [(f"f{i:05d}.txt", f"файл {i}\n")
         for i in range(2100)])
    _zip("empty.zip", [])
    _encrypted_zip(DATA / "archives" / "encrypted.zip")

    def write_json(*parts: str, payload: object) -> None:
        path = DATA.joinpath(*parts)
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    write_json("docling", "chunk_result.json", payload={
        "chunks": [
            {
                "text": "Общие положения\nНастоящий регламент устанавливает порядок обслуживания.",
                "raw_text": "Настоящий регламент устанавливает порядок обслуживания.",
                "contextualized_text": "Общие положения\nНастоящий регламент устанавливает порядок обслуживания.",
                "headings": ["Регламент РЩ-3", "Общие положения"],
                "page_numbers": [1],
            },
            {
                "text": "Поверка производится ежегодно.",
                "headings": ["Регламент РЩ-3", "Поверка"],
                "meta": {"doc_items": [{"prov": [{"page_no": 4}, {"page_no": 5}]}]},
            },
            {
                "contextualized_text": "Приложение А\nФорма акта.",
                "meta": {"headings": ["Приложения"]},
            },
            {"text": "   ", "headings": ["Пустой"]},
            {"text": "", "headings": ["Совсем пустой"]},
        ]
    })
    write_json("docling", "status_success.json",
               payload={"task_id": "t-1", "task_status": "success"})
    write_json("docling", "status_pending.json",
               payload={"task_id": "t-1", "task_status": "started"})
    write_json("docling", "status_failure.json", payload={
        "task_id": "t-1",
        "task_status": "failure",
        "task_meta": {
            "error": "URL is not allowed: http://localhost:8333/rag-docs/x.pdf"
        },
    })
    write_json("docling", "submit_accepted.json",
               payload={"task_id": "c201dd03-c49e-494e-9e76-bf9613289569"})
    write_json("embed_response.json", payload={
        "model": "BAAI/bge-m3",
        "embeddings": [
            {"dense": [0.1] * 1024, "sparse": {"12345": 0.21, "678": 0.05}},
            {"dense": [0.2] * 1024, "sparse": {"999": 0.4}},
        ],
    })

    (DATA / "icons" / "icon.png").write_bytes(_png())
    (DATA / "icons" / "large.png").write_bytes(_png(512, 512))
    (DATA / "icons" / "evil.svg").write_text(
        '<svg xmlns="http://www.w3.org/2000/svg"><script>alert(1)</script></svg>',
        encoding="utf-8",
    )

    (DATA / "icons" / "not_an_image.png").write_bytes(
        b"\x89PNG\r\n\x1a\nfake, not really a png"
    )


REQUIRED = (
    "archives/plain.zip",
    "archives/traversal.zip",
    "archives/collision.zip",
    "archives/bomb.zip",
    "archives/mixed.zip",
    "archives/too_many.zip",
    "archives/empty.zip",
    "archives/encrypted.zip",
    "docling/chunk_result.json",
    "docling/status_success.json",
    "docling/status_pending.json",
    "docling/status_failure.json",
    "docling/submit_accepted.json",
    "embed_response.json",
    "icons/icon.png",
    "icons/large.png",
    "icons/evil.svg",
    "icons/not_an_image.png",
)


def ensure() -> None:
    if all((DATA / name).exists() for name in REQUIRED):
        return
    build()


if __name__ == "__main__":
    build()
    print(f"Фикстуры пересозданы в {DATA}")
