from __future__ import annotations

"""Маппинг индекса kb-v2.
rag_id обязателен и является границей безопасности, а не оптимизацией:
индекс общий на всю платформу, наборы разделяются только фильтром по нему.
Забытый фильтр — не деградация выдачи, а утечка чужих документов.
"""

from typing import Any

EMBED_DIM = 1024

SEARCH_SOURCE_EXCLUDES = ["content_vector", "content_sparse"]


INDEX_BODY: dict[str, Any] = {
    "settings": {
        "index": {
            "knn": True,
            "number_of_shards": 1,
            "number_of_replicas": 0,
            "refresh_interval": "30s",
        },
        "analysis": {
            "filter": {
                "russian_stop": {"type": "stop", "stopwords": "_russian_"},
                "russian_stemmer": {"type": "stemmer", "language": "russian"},
            },
            "analyzer": {
                "ru_en": {
                    "type": "custom",
                    "tokenizer": "standard",
                    "filter": ["lowercase", "russian_stop", "russian_stemmer"],
                }
            },
        },
    },
    "mappings": {
        "properties": {
            "rag_id": {"type": "keyword"},
            "document_id": {"type": "keyword"},
            "chunk_index": {"type": "integer"},
            "content": {"type": "text", "analyzer": "ru_en"},
            "headings": {"type": "text", "analyzer": "ru_en"},
            "pages": {"type": "integer"},
            "content_hash": {"type": "keyword"},
            "content_vector": {
                "type": "knn_vector",
                "dimension": EMBED_DIM,
                "method": {
                    "name": "hnsw",
                    "engine": "faiss",
                    "space_type": "cosinesimil",
                    "parameters": {"ef_construction": 256, "m": 16},
                },
            },
            "content_sparse": {"type": "rank_features"},
        }
    },
}
