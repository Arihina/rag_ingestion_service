"""Объектное хранилище: SeaweedFS через S3-шлюз."""

from app.storage.seaweed import (
    ObjectStorage,
    StoredObject,
    document_key,
    stream_upload,
)

__all__ = ["ObjectStorage", "StoredObject", "document_key", "stream_upload"]
