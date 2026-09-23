from __future__ import annotations


"""Конфигурация сервиса. Префикс INGEST_, файл ingest.env."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="INGEST_", env_file="ingest.env", extra="ignore"
    )

    host: str = "127.0.0.1"
    port: int = 8011
    internal_port: int = 8012
    internal_host: str = "127.0.0.1"

    reload: bool = False
    log_level: str = "info"
    timeout_keep_alive: int = 300

    docling_url: str = "http://localhost:5001"
    docling_api_key: str | None = None

    opensearch_url: str = "http://localhost:9200"
    opensearch_user: str | None = None
    opensearch_password: str | None = None
    index_name: str = "kb-v2"
    opensearch_timeout: float = 60.0
    refresh_after_load: bool = True

    embed_model: str = "BAAI/bge-m3"
    embed_batch_size: int = 16
    embed_max_length: int = 1024
    embed_device: str | None = None

    chunk_max_tokens: int = 768
    ocr_engine: str = "easyocr"
    ocr_lang: str = "ru,en"
    force_ocr: bool = True

    max_upload_mb: int = 200

    database_url: str = "postgresql+asyncpg://rag:rag@localhost:5437/rag_ingest"
    db_pool_size: int = 10
    db_max_overflow: int = 20

    redis_url: str = "redis://localhost:6379/1"
    rq_queue_ingest: str = "ingest"
    rq_queue_maintenance: str = "maintenance"
    rq_job_timeout: int = 3600

    self_url: str = "http://localhost:8012"

    s3_endpoint_url: str = "http://localhost:8333"
    s3_presign_endpoint_url: str | None = None

    docling_transfer: str = "url"
    s3_access_key: str = "rag"
    s3_secret_key: str = "rag"
    s3_region: str = "us-east-1"
    s3_bucket_docs: str = "rag-docs"
    s3_bucket_icons: str = "rag-icons"
    s3_bucket_staging: str = "rag-staging"
    presign_ttl_seconds: int = 3600

    archive_max_entries: int = 2000
    archive_max_uncompressed: int = 5 * 1024**3
    archive_max_ratio: float = 120.0
    archive_max_file_size: int = 200 * 1024**2

    icon_max_bytes: int = 512 * 1024
    icon_size_px: int = 256
    icon_allowed_types: tuple[str, ...] = ("image/png", "image/jpeg", "image/webp")

    rag_default_temperature: float = 0.3
    rag_default_top_k: int = 5
    rag_default_score_threshold: float = 0.0
    rag_top_k_max: int = 10
    rag_prompt_max_chars: int = 4000
    rag_max_sets_per_owner: int = 50

    bucket_import_max_file_size: int = 50 * 1024**2
    bucket_import_max_files: int = 500
    rag_max_bytes_per_set: int = 20 * 1024**3

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