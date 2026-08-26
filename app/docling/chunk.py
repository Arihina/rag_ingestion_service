from __future__ import annotations


"""Чанк на выходе docling: текст, провенанс и детерминированный ключ."""

import hashlib
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class DoclingChunk:
    """Два представления текста.

    text       — сырой. Идёт в индекс под BM25 и в выдачу пользователю.
    embed_text — контекстуализированный, с приклеенными родительскими
                 заголовками. Идёт ТОЛЬКО в модель.
    """

    rag_id: str
    document_id: str
    index: int
    text: str
    embed_text: str
    headings: tuple[str, ...] = ()
    pages: tuple[int, ...] = ()
    doc_meta: dict[str, Any] = field(default_factory=dict)

    @property
    def chunk_id(self) -> str:
        """Детерминированный _id: переиндексация перезаписывает.

        В ключе именно (rag_id, document_id), а не имя файла.
        """
        key = f"{self.rag_id}:{self.document_id}:{self.index}".encode("utf-8")
        return hashlib.blake2b(key, digest_size=16).hexdigest()

    @property
    def content_hash(self) -> str:
        return hashlib.blake2b(self.text.encode("utf-8"), digest_size=16).hexdigest()

    def to_source(self) -> dict[str, Any]:
        return {
            "rag_id": self.rag_id,
            "document_id": self.document_id,
            "chunk_index": self.index,
            "content": self.text,
            "headings": list(self.headings),
            "pages": list(self.pages),
            "content_hash": self.content_hash,
            **self.doc_meta,
        }
