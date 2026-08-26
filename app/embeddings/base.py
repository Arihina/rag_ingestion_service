from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class EmbeddingResult:
    dense: list[float]
    sparse: dict[str, float]


class BaseEmbedder(ABC):
    """Считает эмбеддинги пачками. Подклассы реализуют только вызов модели."""

    def __init__(self, batch_size: int = 16) -> None:
        self._batch_size = batch_size

    def embed(self, texts: Sequence[str]) -> list[EmbeddingResult]:
        results: list[EmbeddingResult] = []
        for start in range(0, len(texts), self._batch_size):
            batch = list(texts[start: start + self._batch_size])
            try:
                results.extend(self._embed_batch(batch))
            except Exception as exc:
                raise RuntimeError(
                    f"Ошибка эмбеддинга батча {start}..{start + len(batch)}: {exc}"
                ) from exc
        return results

    @abstractmethod
    def _embed_batch(self, texts: list[str]) -> list[EmbeddingResult]:
        """Прогнать один батч через модель."""
