"""Объектное хранилище: SeaweedFS через S3-шлюз."""

from app.storage.seaweed import (
    ObjectStorage,
    RemoteObject,
    StoredObject,
    document_key,
    stream_upload,
)

__all__ = [
    "ObjectStorage",
    "RemoteObject",
    "StoredObject",
    "document_key",
    "stream_upload",
]