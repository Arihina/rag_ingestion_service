from __future__ import annotations

"""Шкала скоров kNN.

_score от OpenSearch — монотонное преобразование расстояния, а не косинус.
Ошибка в преобразовании не даёт исключения: она тихо смещает порог,
который задал пользователь.
"""

import unittest

from app.search import cosine_from_score, score_from_cosine


class ScoringTests(unittest.TestCase):
    def test_преобразование_обратимо(self):
        for cos in (1.0, 0.9, 0.75, 0.5, 0.25, 0.0, -0.5, -1.0):
            with self.subTest(cos=cos):
                self.assertAlmostEqual(
                    cosine_from_score(score_from_cosine(cos)), cos, places=9
                )

    def test_идеальное_совпадение_даёт_единицу(self):
        """На этом же строится проверка калибровки при старте."""
        self.assertAlmostEqual(score_from_cosine(1.0), 1.0, places=9)
        self.assertAlmostEqual(cosine_from_score(1.0), 1.0, places=9)

    def test_монотонность(self):
        """Порядок выдачи не должен переворачиваться преобразованием."""
        cosines = [-1.0, -0.5, 0.0, 0.3, 0.6, 0.9, 1.0]
        scores = [score_from_cosine(c) for c in cosines]
        self.assertEqual(scores, sorted(scores))

    def test_ортогональные_векторы(self):
        self.assertAlmostEqual(score_from_cosine(0.0), 0.5, places=9)

    def test_порог_переводится_в_min_score(self):
        """Пользователь задаёт косинус, в запрос уходит min_score."""
        self.assertGreater(score_from_cosine(0.6), score_from_cosine(0.4))
