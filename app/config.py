from __future__ import annotations


"""Конфигурация сервиса. Префикс INGEST_, файл ingest.env."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="INGEST_", env_file="ingest.env", extra="ignore"
    )

    docling_url: str = "http://localhost:5001"
    docling_api_key: str | None = None

    opensearch_url: str = "http://localhost:9200"
    opensearch_user: str | None = None
    opensearch_password: str | None = None
    index_name: str = "kb-v2"

    embed_model: str = "BAAI/bge-m3"
    embed_batch_size: int = 16
    embed_max_length: int = 1024
    embed_device: str | None = None

    chunk_max_tokens: int = 768
    ocr_engine: str = "easyocr"
    ocr_lang: str = "ru,en"
    force_ocr: bool = True

    api_key: str | None = None
    max_upload_mb: int = 200

    embed_concurrency: int = 1
    embed_query_concurrency: int = 2
    embed_query_batch_size: int = 8
    separate_query_embedder: bool = True

    max_pending_jobs: int = 50

    allowed_suffixes: tuple[str, ...] = (
        ".pdf", ".docx", ".pptx", ".xlsx", ".html", ".htm",
        ".md", ".txt", ".csv", ".png", ".jpg", ".jpeg", ".tiff",
    )


settings = Settings()
