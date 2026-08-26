"""Слой OpenSearch: маппинг kb-v2, загрузка чанков, шкала скоров."""

from .index import EMBED_DIM, INDEX_BODY, SEARCH_SOURCE_EXCLUDES
from .loader import OpenSearchLoader
from .scoring import cosine_from_score, score_from_cosine

__all__ = [
    "EMBED_DIM",
    "INDEX_BODY",
    "SEARCH_SOURCE_EXCLUDES",
    "OpenSearchLoader",
    "cosine_from_score",
    "score_from_cosine",
]
