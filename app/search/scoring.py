from __future__ import annotations


"""Шкала скоров kNN."""


def score_from_cosine(cosine: float) -> float:
    """Порог в терминах косинуса -> min_score для kNN-ветки."""
    return 1.0 / (2.0 - cosine)


def cosine_from_score(score: float) -> float:
    """Обратное преобразование, для диагностики и калибровки."""
    return 2.0 - 1.0 / score
