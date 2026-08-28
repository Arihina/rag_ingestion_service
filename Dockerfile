# Один Dockerfile, две цели: worker и api.
#
# Роли принципиально разные по весу. Воркер RQ модель НЕ держит — за
# векторами он ходит в /embed процесса API (см. app/tasks/jobs.py), поэтому
# ему не нужны ни torch, ни FlagEmbedding, ни GPU-рантайм. Общий образ
# затащил бы в него ~3 ГБ колёс, которые он ни разу не импортирует, и
# заставил бы пробрасывать карту туда, где она не используется.
#
# Сборка:
#   docker build --target api    -t rag-ingest-api:dev .
#   docker build --target worker -t rag-ingest-worker:dev .
#
# Dev и prod собираются ОДИНАКОВО. Отличается обвязка: в деве исходники
# монтируются томом ради reload, в проде остаются те, что легли в образ.

ARG PYTHON_IMAGE=python:3.12-slim-bookworm

FROM ${PYTHON_IMAGE} AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# curl нужен healthcheck'у, остальное — сборке колёс asyncpg/psycopg.
RUN apt-get update \
 && apt-get install -y --no-install-recommends curl ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# Непривилегированный пользователь. UID 1001 совпадает с тем, под которым
# работает docling-serve, — тома с весами шарятся без возни с правами.
RUN groupadd -g 1001 app && useradd -u 1001 -g 1001 -m -s /bin/bash app

WORKDIR /srv

# Зависимости отдельным слоем: правка кода не пересобирает pip install.
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY --chown=app:app app ./app
COPY --chown=app:app alembic ./alembic
COPY --chown=app:app alembic.ini ./
COPY --chown=app:app scripts ./scripts

FROM base AS worker

# Ни torch, ни FlagEmbedding. Образ порядка 200 МБ, поднимается за секунды,
# масштабируется обычными репликами без проброса устройств.
USER app

ENV INGEST_LOG_LEVEL=info

# Две очереди одним процессом: ingest (документы) и maintenance
# (разворачивание архивов, импорт, уборка). Порядок значим — RQ разбирает
# очереди слева направо, и индексация не должна ждать уборки.
CMD ["sh", "-c", "rq worker --url \"$INGEST_REDIS_URL\" \"$INGEST_RQ_QUEUE_INGEST\" \"$INGEST_RQ_QUEUE_MAINTENANCE\""]

FROM base AS api

# cu128, а не дефолтный индекс PyPI: Blackwell (sm_120) требует CUDA >= 12.8,
# а на A100 (sm_80) та же сборка работает штатно — один образ на дев и прод.
# Для машины без карты: --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128

RUN pip install torch --index-url ${TORCH_INDEX_URL}

COPY requirements-embed.txt .
RUN pip install -r requirements-embed.txt

# Веса bge-m3 в образ НЕ запекаются: это +2.3 ГБ к каждой пересборке.
# Они лежат на том же томе, что и веса docling, и кладутся туда сервисом
# docling-models-init — он уже качает токенизатор bge-m3, теперь и модель.
# HF_HUB_OFFLINE здесь НЕ выставляется: в деве веса могут ещё не быть
# скачаны, и офлайн-режим превратил бы первый старт в невнятную ошибку.
# В проде он включается в compose, когда том с весами гарантированно полон.
ENV HF_HOME=/models/hf

USER app
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=180s --retries=3 \
  CMD curl -sf http://localhost:8000/health || exit 1

# Обе модели грузятся на старте, отсюда длинный start-period выше.
CMD ["python", "-m", "app.main"]