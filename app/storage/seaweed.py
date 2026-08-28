from __future__ import annotations

"""SeaweedFS через S3-совместимый шлюз.

Три операции несут архитектурный смысл:
  * presigned GET — docling качает файл сам через http_sources, и байты
    не проходят через наш процесс второй раз;
  * server-side copy — импорт из staging не гонит содержимое через нас;
  * потоковая запись — файл не поднимается в память процесса целиком.
"""

import hashlib
import logging
from dataclasses import dataclass
from typing import Any, AsyncIterator, BinaryIO

import boto3
from botocore.config import Config as BotoConfig
from botocore.exceptions import ClientError

from app.config import settings

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class StoredObject:
    key: str
    size_bytes: int
    content_hash: str


class ObjectStorage:
    def __init__(self) -> None:
        self._client = self._make_client(settings.s3_endpoint_url)
        presign_url = settings.s3_presign_endpoint_url or settings.s3_endpoint_url
        self._presign_client = (
            self._client
            if presign_url == settings.s3_endpoint_url
            else self._make_client(presign_url)
        )

    @staticmethod
    def _make_client(endpoint_url: str):
        return boto3.client(
            "s3",
            endpoint_url=endpoint_url,
            aws_access_key_id=settings.s3_access_key,
            aws_secret_access_key=settings.s3_secret_key,
            region_name=settings.s3_region,
            config=BotoConfig(signature_version="s3v4", s3={
                              "addressing_style": "path"}),
        )

    def ensure_buckets(self) -> None:
        for bucket in (
            settings.s3_bucket_docs,
            settings.s3_bucket_icons,
            settings.s3_bucket_staging,
        ):
            try:
                self._client.head_bucket(Bucket=bucket)
            except ClientError:
                logger.info("Создаю бакет %s", bucket)
                self._client.create_bucket(Bucket=bucket)

    def healthy(self) -> bool:
        try:
            self._client.list_buckets()
            return True
        except Exception:
            logger.warning("Объектное хранилище недоступно", exc_info=True)
            return False

    def put_stream(self, bucket: str, key: str, body: BinaryIO) -> None:
        self._client.upload_fileobj(body, bucket, key)

    def put_bytes(self, bucket: str, key: str, data: bytes, content_type: str) -> None:
        self._client.put_object(
            Bucket=bucket, Key=key, Body=data, ContentType=content_type
        )

    def copy(self, src_bucket: str, src_key: str, dst_bucket: str, dst_key: str) -> int:
        """Server-side copy: содержимое не проходит через процесс."""
        self._client.copy_object(
            Bucket=dst_bucket,
            Key=dst_key,
            CopySource={"Bucket": src_bucket, "Key": src_key},
        )
        return self._client.head_object(Bucket=dst_bucket, Key=dst_key)["ContentLength"]

    def presigned_get(self, bucket: str, key: str, ttl: int | None = None) -> str:
        """Ссылку подписывает presign-клиент: по ней придёт docling"""
        return self._presign_client.generate_presigned_url(
            "get_object",
            Params={"Bucket": bucket, "Key": key},
            ExpiresIn=ttl or settings.presign_ttl_seconds,
        )

    def get_bytes(self, bucket: str, key: str) -> tuple[bytes, str]:
        response = self._client.get_object(Bucket=bucket, Key=key)
        return response["Body"].read(), response.get("ContentType", "application/octet-stream")

    def list_prefix(self, bucket: str, prefix: str) -> list[dict[str, Any]]:
        paginator = self._client.get_paginator("list_objects_v2")
        out: list[dict[str, Any]] = []
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            out.extend(page.get("Contents", []))
        return out

    def delete(self, bucket: str, key: str) -> None:
        self._client.delete_object(Bucket=bucket, Key=key)

    def delete_prefix(self, bucket: str, prefix: str) -> int:
        keys = [{"Key": obj["Key"]}
                for obj in self.list_prefix(bucket, prefix)]
        deleted = 0
        for start in range(0, len(keys), 1000):
            batch = keys[start: start + 1000]
            self._client.delete_objects(
                Bucket=bucket, Delete={"Objects": batch})
            deleted += len(batch)
        return deleted


def document_key(rag_id: str, document_id: str, filename: str) -> str:
    return f"{rag_id}/{document_id}/{filename}"


async def stream_upload(
    storage: ObjectStorage,
    bucket: str,
    key: str,
    chunks: AsyncIterator[bytes],
    *,
    limit_bytes: int,
) -> StoredObject:
    """Пишет во временный файл, считая хэш на лету, затем отдаёт в S3.

    Промежуточный файл нужен, потому что S3 требует знать длину тела,
    UploadFile её не гарантирует. В память при этом не поднимается ничего.
    """
    import tempfile

    digest = hashlib.blake2b(digest_size=16)
    written = 0

    with tempfile.SpooledTemporaryFile(max_size=8 * 1024 * 1024) as buffer:
        async for chunk in chunks:
            written += len(chunk)
            if written > limit_bytes:
                raise ValueError(f"Файл больше {limit_bytes} байт")
            digest.update(chunk)
            buffer.write(chunk)
        buffer.seek(0)
        storage.put_stream(bucket, key, buffer)

    return StoredObject(key=key, size_bytes=written, content_hash=digest.hexdigest())
