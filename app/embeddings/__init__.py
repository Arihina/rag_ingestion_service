"""Эмбеддеры. Модель платформенно фиксирована: BAAI/bge-m3, 1024 dim."""

from .base import BaseEmbedder, EmbeddingResult
from .bge_m3 import BgeM3Embedder
from .http import HttpEmbedder

__all__ = ["BaseEmbedder", "BgeM3Embedder", "EmbeddingResult", "HttpEmbedder"]
