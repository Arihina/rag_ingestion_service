# Развёртывание: prod

Всё в контейнерах, образы приезжают из реестра. Отличия от дева сведены в
таблицу в конце.

---

## 1. Образы

Собираются из одного `Dockerfile` двумя целями. Роли принципиально разные
по весу: воркер не держит модель — за векторами он ходит в `/embed`
процесса `api`, — поэтому ему не нужны ни torch, ни FlagEmbedding, ни
GPU-рантайм.

```bash
TAG=$(git rev-parse --short HEAD)

docker build --target api    -t registry.internal/rag-ingest-api:$TAG .
docker build --target worker -t registry.internal/rag-ingest-worker:$TAG .

docker push registry.internal/rag-ingest-api:$TAG
docker push registry.internal/rag-ingest-worker:$TAG
```

Ориентировочно: `api` около 6 ГБ, `worker` около 200 МБ.

**Пинить по digest, а не по тегу** — тег можно молча перезаписать:

```bash
docker inspect --format='{{index .RepoDigests 0}}' registry.internal/rag-ingest-api:$TAG
```

Для машины без карты (например, отдельный билд-агент):
`--build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu`.

---

## 2. Секреты и конфиги

```bash
cp ingest.env.example .env
cp deploy/seaweedfs-s3.prod.json.example deploy/seaweedfs-s3.prod.json
```

В `.env` обязательны:

```
DOCLING_KEY=...
DOCLING_IMAGE=ghcr.io/docling-project/docling-serve-cu128@sha256:...
OPENSEARCH_PASSWORD=...
POSTGRES_PASSWORD=...
S3_ACCESS_KEY=...
S3_SECRET_KEY=...
INGEST_API_KEY=...
INGEST_IMAGE_API=registry.internal/rag-ingest-api@sha256:...
INGEST_IMAGE_WORKER=registry.internal/rag-ingest-worker@sha256:...
```

Compose объявляет их через `${VAR:?required}`, поэтому отсутствие ключа
роняет запуск сразу, а не на первом обращении.

`deploy/seaweedfs-s3.prod.json` содержит ключи и **в git не коммитится**
(он в `.gitignore`). `accessKey`/`secretKey` в нём обязаны совпадать с
`S3_ACCESS_KEY`/`S3_SECRET_KEY`.

Права `Admin` в этом файле нужны только под создание бакетов на первом
старте (`ensure_buckets`). После того как `rag-docs`, `rag-icons` и
`rag-staging` созданы, их можно снять и оставить
`Read/Write/List/Tagging`.

---

## 3. Веса моделей

Том `models` должен быть наполнен до старта `api`: в проде включён
`HF_HUB_OFFLINE=1`, и рантайм в сеть не полезет.

```bash
docker compose -f docker-compose.prod.yml run --rm \
  -v rag_ingestion_service_models:/models \
  --entrypoint bash docling-api -c '
    export HF_HOME=/models/hf
    docling-tools models download -o /models/docling layout tableformer easyocr rapidocr
    python -c "from huggingface_hub import snapshot_download; snapshot_download(\"BAAI/bge-m3\")"
  '
```

`rapidocr` обязателен, даже если рабочий движок — EasyOCR: docling-serve
на старте прогревает пайплайн с дефолтными опциями, а дефолтный движок —
`auto`, который выбирает RapidOCR. Без этих весов контейнер падает с
`FileNotFoundError` ещё до первого запроса.

---

## 4. Запуск

```bash
docker compose -f docker-compose.prod.yml up -d
docker compose -f docker-compose.prod.yml ps
```

Порядок стартов задан через `depends_on: service_healthy`, так что `api`
поднимется после того, как OpenSearch, Postgres, Redis и SeaweedFS
отчитаются о готовности.

---

## 5. Миграции

```bash
docker compose -f docker-compose.prod.yml exec api alembic upgrade head
```

Отдельного init-контейнера намеренно нет: при одной реплике `api` этого
достаточно, а при двух гонку двух параллельных `upgrade head` пришлось бы
разруливать явно. Если реплик станет больше — заводить init-job с
блокировкой, а не надеяться на удачу.

Файлы миграций едут в образе, генерируются в деве и коммитятся.
`--autogenerate` на проде **не запускать**: он сравнивает модели с живой
базой и может предложить снос колонок.

---

## 6. Проверка

```bash
docker compose -f docker-compose.prod.yml exec api \
  curl -s localhost:8000/health | jq
```

Все пять флагов `true`. Наружу `api` не публикуется — доступ через мастер.

---

## Отличия от дева

| | dev | prod |
|---|---|---|
| Образы | собираются локально | из реестра, по digest |
| Исходники | том `./app:ro` | только из образа |
| Веса | том с весами docling | том `models`, `HF_HUB_OFFLINE=1` |
| OpenSearch | без TLS | TLS + пароль |
| SeaweedFS | порты на localhost | наружу не публикуется вовсе |
| API-ключ | выключен | обязателен |
| Реплик `api` | 1 | 1 |
| Реплик `worker` | 1 | 2 (подбирается) |

---

## Масштабирование

**`api` — ровно одна реплика.** Он владеет картой и обеими копиями bge-m3,
и сериализовать доступ к GPU должен ровно один набор семафоров в одном
процессе. Две реплики дали бы 2×3 ГБ VRAM и две независимые очереди к
одной карте без всякой координации. По той же причине в `app/main.py` не
выставляется `workers` у uvicorn.

**Масштабируется `worker`.** Состояние у него только в Postgres и Redis,
модели нет, GPU не нужен:

```bash
docker compose -f docker-compose.prod.yml up -d --scale worker=6
```

`replicas: 2` в файле — стартовая догадка. Померь на своём корпусе: если
документы преимущественно текстовые PDF без OCR, узкое место окажется на
CPU docling, а не на нашей стороне, и наращивать надо `docling-worker` и
`DOCLING_NUM_THREADS`.

---

## Redis: один инстанс, две очереди

`docling-serve` использует Redis под свою очередь конверсии на базе `0`
(`DOCLING_SERVE_ENG_RQ_REDIS_URL`), control plane — на базе `1`
(`INGEST_REDIS_URL`). Общий инстанс, раздельные пространства ключей;
свести их на одну базу нельзя.

`--appendonly yes` обязателен: без него задачи последней минуты теряются
при рестарте молча, а документы навсегда остаются в `pending`.

**API docling без единого воркера принимает задачи, но никто их не
выполняет** — снаружи это выглядит как вечный `pending`. То же верно и для
наших воркеров.

---

## Стоит ли держать OpenSearch здесь же

В `docker-compose.prod.yml` он включён, потому что файл задумывался
самодостаточным. Но `single-node` без реплик означает, что потеря диска
равна потере индекса, а поиск и парсинг конкурируют за память и IO
совершенно по-разному. Если кластер уже живёт отдельно — выбрось сервис
`opensearch` и правь `INGEST_OPENSEARCH_URL`.

Heap считается как половина RAM ноды, но не выше ~31 ГБ: за этой границей
JVM теряет сжатые указатели, и heap начинает работать хуже, а не лучше.
`knn.memory.circuit_breaker.limit` настраивается отдельно — HNSW-графы
лежат **вне** heap и с `-Xmx` не связаны.

---

## Обновление

```bash
# 1. образы с новым тегом, digest в .env
# 2. миграции ДО перезапуска, если схема менялась
docker compose -f docker-compose.prod.yml up -d --no-deps api worker
docker compose -f docker-compose.prod.yml exec api alembic upgrade head
```

Порядок важен, если миграция несовместима со старым кодом: тогда сначала
останавливаются воркеры, потом миграция, потом `api` и воркеры вместе.

Задачи в очереди переживают перезапуск (`appendonly`), но задача,
прерванная на середине, оставит документ в `processing`. Отдельного
сборщика таких документов пока нет — их видно запросом:

```sql
SELECT id, rag_id, filename FROM documents
WHERE status = 'processing' AND updated_at < now() - interval '1 hour';
```

---

## Чего пока нет

- **Защиты `/admin` отдельно от `INGEST_API_KEY`** — та же заглушка, что и
  в мастере, до Keycloak. Наружу сервис не торчит, но внутри сети разделения
  прав нет.
- **Фоновой сверки осиротевших объектов** в SeaweedFS. Порядок записи
  (объект → строка → задача) сделан так, чтобы осиротевший объект был
  безвредным, но со временем они копятся.
- **Пересборки зависших `processing`** — см. запрос выше, пока руками.