"""HTTP-клиент docling-serve: parse + structure + chunk на стороне сервиса."""

from .chunk import DoclingChunk
from .client import (
    BaseDoclingClient,
    DoclingError,
    DoclingFileClient,
    DoclingTimeoutError,
    DoclingUrlClient,
)
from .options import ChunkingOptions, ConversionOptions

__all__ = [
    "BaseDoclingClient",
    "ChunkingOptions",
    "ConversionOptions",
    "DoclingChunk",
    "DoclingError",
    "DoclingFileClient",
    "DoclingTimeoutError",
    "DoclingUrlClient",
]
