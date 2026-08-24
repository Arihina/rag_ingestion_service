# Blackwell (RTX 5070 Ti, sm_120) требует сборок torch с CUDA >= 12.8.
# Ampere (A100, sm_80) в этот же образ входит — dev и prod идут на одном билде.
# Для CI без GPU:
# --build-arg DOCLING_BASE=ghcr.io/docling-project/docling-serve-cpu:v1.21.0

ARG DOCLING_BASE=ghcr.io/docling-project/docling-serve-cu128:v1.21.0
FROM ${DOCLING_BASE}

USER 0

ENV HF_HOME=/opt/app-root/src/.cache/huggingface \
    DOCLING_SERVE_ARTIFACTS_PATH=/opt/app-root/src/models

RUN pip install --no-cache-dir "rapidocr>=3.9.1"

RUN docling-tools models download --all -o /opt/app-root/src/models

RUN python -c "from transformers import AutoTokenizer; \
    AutoTokenizer.from_pretrained('BAAI/bge-m3')"

RUN chown -R 1001:0 /opt/app-root/src

USER 1001

EXPOSE 5001