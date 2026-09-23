#!/usr/bin/env bash
#
# Наполнение тома моделей. Единственный источник правды для dev, test и
# prod: раньше эта команда жила в двух местах и успела разъехаться —
# в проде потерялся шаг с русской моделью EasyOCR, то есть русский OCR
# на сканах там бы просто не работал.
#
# Запускается внутри образа docling-serve (у него есть docling-tools,
# easyocr и transformers).
#
# Переменные:
#   MODELS_DIR       куда класть, по умолчанию /models
#   OCR_LANGS        языки EasyOCR через пробел, по умолчанию "ru en"
#   WITH_EMBEDDINGS  1 — дотянуть ВЕСА bge-m3 (~2.3 ГБ).
#
# WITH_EMBEDDINGS нужен, только когда процесс api работает в контейнере
# и монтирует этот том (прод). Когда api запускается с хоста в venv
# (dev, test), веса живут в хостовом кэше HuggingFace, и тащить их сюда
# значит дважды занять 2.3 ГБ. Токенизатор при этом нужен ВСЕГДА: по
# нему HybridChunker внутри docling считает границы чанков, и он обязан
# совпадать с эмбеддером.

set -eux

MODELS_DIR="${MODELS_DIR:-/models}"
OCR_LANGS="${OCR_LANGS:-ru en}"
WITH_EMBEDDINGS="${WITH_EMBEDDINGS:-0}"

export HF_HOME="${MODELS_DIR}/hf"
export HOME="${MODELS_DIR}/home"

mkdir -p "${MODELS_DIR}/docling" "${HF_HOME}" "${HOME}"

# Только эти четыре. Всё остальное из каталога docling — VLM-модели
# (granitedocling, smoldocling, granite_vision), распознавание формул и
# классификатор картинок — к нашей задаче отношения не имеет и весит
# на порядок больше самого пайплайна.
#
# rapidocr в списке, хотя рабочий движок — EasyOCR: docling-serve на
# старте прогревает пайплайн с ДЕФОЛТНЫМИ опциями, а дефолтный движок
# выбирает RapidOCR. Без его весов контейнер падает с FileNotFoundError
# ещё до первого запроса. Серверной настройки движка по умолчанию пока
# нет (docling-serve issue #554), так что убрать нельзя — это mobile-
# модели PP-OCRv4 на пару десятков мегабайт, терпимо.
docling-tools models download -o "${MODELS_DIR}/docling" \
    layout tableformer easyocr rapidocr

# Русскую модель EasyOCR приходится тянуть отдельно: при заданном
# artifacts_path docling считает режим офлайновым и отключает докачку
# весов в рантайме, а docling-tools кладёт только детектор и латиницу.
python - "${OCR_LANGS}" <<'PY'
import sys

import easyocr

langs = sys.argv[1].split()
print(f"EasyOCR: langs={langs}")
easyocr.Reader(
    langs,
    model_storage_directory="/models/docling/EasyOcr",
    download_enabled=True,
    gpu=False,
)
PY

# Токенизатор — всегда. Несколько мегабайт против 2.3 ГБ у весов.
python -c "
from transformers import AutoTokenizer
AutoTokenizer.from_pretrained('BAAI/bge-m3')
print('токенизатор bge-m3: готов')
"

if [ "${WITH_EMBEDDINGS}" = "1" ]; then
    python -c "
from huggingface_hub import snapshot_download
snapshot_download('BAAI/bge-m3')
print('веса bge-m3: готовы')
"
fi

# UID 1001 — под ним работают и docling-serve, и наши образы.
chown -R 1001:0 "${MODELS_DIR}"

echo "=== ИТОГ ==="
du -sh "${MODELS_DIR}"/* || true
ls -la "${MODELS_DIR}/docling/EasyOcr" || true