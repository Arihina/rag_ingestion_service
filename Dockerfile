# Blackwell (RTX 5070 Ti, sm_120) требует сборок torch с CUDA >= 12.8.
# Ampere (A100, sm_80) в этот же образ входит — dev и prod идут на одном билде.
# Для CI без GPU:
# --build-arg DOCLING_BASE=ghcr.io/docling-project/docling-serve-cpu:v1.21.0

ARG DOCLING_BASE=ghcr.io/docling-project/docling-serve-cu128:v1.21.0
FROM ${DOCLING_BASE}

USER 0

# Единственная правка образа. Явная зачистка нужна даже с --no-cache-dir:
# pip оставляет собранные колёса, если пакет ставился из sdist.
RUN pip install --no-cache-dir "rapidocr>=3.9.1" \
 && rm -rf /root/.cache /tmp/* /var/tmp/*

USER 1001

# Весов в образе НЕТ. Иначе каждая пересборка порождает новый набор
# многогигабайтных слоёв, и они копятся, пока не съедят диск.
# Каталоги наполняет одноразовый сервис docling-models-init.
ENV HF_HOME=/models/hf \
    DOCLING_SERVE_ARTIFACTS_PATH=/models/docling

EXPOSE 5001