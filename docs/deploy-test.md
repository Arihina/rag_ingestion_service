# Развёртывание: тестовый сервер

Зависимости в контейнерах, сервис и воркер — с хоста, как в деве. Стенд
нужен для проверки кода, а не образов, поэтому `api` и `worker` в
`docker-compose.test.yml` отсутствуют вовсе — профиля `app` здесь нет.

---

## Чем отличается от dev и prod

| | dev | **test** | prod |
|---|---|---|---|
| `api` / `worker` | с хоста (или профиль `app`) | **только с хоста** | контейнеры из реестра |
| Том моделей | папка на хосте | **именованный volume** | именованный volume |
| Веса bge-m3 в томе | нет | **нет** | да (`api` в контейнере) |
| `restart` | нет | **unless-stopped** | unless-stopped |
| TLS, пароли | нет | **нет** | обязательны |
| docling | `local` | **`local`** | `rq` + воркеры |
| Публикация портов | `127.0.0.1` | **настраивается** | нет вовсе |

Две вещи заимствованы у прода: `restart: unless-stopped`, потому что
стенд переживает перезагрузку машины, и именованный том моделей —
готовить каталог руками и следить за правами на сервере некому.

---

## Запуск

```bash
cp ingest.env.example ingest.env
docker compose -f docker-compose.test.yml up -d
```

Первый запуск долгий: `docling-models-init` качает layout, tableformer,
easyocr с русской моделью, rapidocr и токенизатор bge-m3. `docling`
стартует только после его успешного выхода.

```bash
docker compose -f docker-compose.test.yml ps
docker compose -f docker-compose.test.yml logs -f docling-models-init
```

Дальше — сервис и воркер в venv на хосте:

```bash
python3 -m venv venv && source venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt -r requirements-embed.txt

alembic upgrade head
python -m app.main
```

```bash
rq worker -u redis://localhost:6379/1 ingest maintenance
```

Воркер — из корня репозитория: `Settings` читает `ingest.env`
относительно текущего каталога.

```bash
curl -s localhost:8011/health | jq
```

---

## Настройки стенда

Переопределяются переменными окружения при `up`, либо через `.env`
рядом с compose-файлом.

| Переменная | По умолчанию | Зачем |
|---|---|---|
| `BIND_ADDR` | `127.0.0.1` | адрес публикации портов зависимостей |
| `OPENSEARCH_HEAP` | `4g` | heap JVM |
| `OPENSEARCH_MEM_LIMIT` | `12g` | лимит памяти контейнера |
| `POSTGRES_PASSWORD` | `rag` | |
| `DOCLING_DEVICE` | `cuda:0` | `cpu` для стенда без карты |
| `DOCLING_NUM_THREADS` | `4` | |

### Про `BIND_ADDR`

По умолчанию зависимости слушают только `127.0.0.1`: `api` и `worker`
живут на этой же машине, и выставлять Postgres с OpenSearch в сеть
незачем. Снаружи стенд виден через порт самого сервиса, который задаётся
`INGEST_HOST` в `ingest.env`.

Если до Postgres или Dashboards нужно дотянуться с другой машины:

```bash
BIND_ADDR=0.0.0.0 docker compose -f docker-compose.test.yml up -d
```

Это открывает их всей сети без единого пароля (кроме Postgres), так что
годится только в доверенном сегменте.

### Стенд без GPU

```bash
DOCLING_DEVICE=cpu docker compose -f docker-compose.test.yml up -d
```

Плюс убрать блок `deploy.resources` у сервиса `docling` — иначе Docker
потребует nvidia-рантайм и контейнер не стартует. Конверсия сканов на
CPU идёт в разы медленнее, для функциональных проверок приемлемо.

`api` при этом всё равно попросит GPU под bge-m3. На CPU он поднимется,
но эмбеддинги будут считаться долго; для стенда, где проверяют логику, а
не пропускную способность, это терпимо.

---

## Модели

Состав задаётся одним скриптом `deploy/download-models.sh`, общим для
dev, test и prod. Раньше команда была скопирована в dev-compose и в
`deploy-prod.md`, и версии разошлись: в проде потерялся шаг с русской
моделью EasyOCR.

Качается ровно четыре модели docling:

| Модель | Зачем |
|---|---|
| `layout` | разметка страницы, основа пайплайна |
| `tableformer` | таблицы в машиночитаемых документах |
| `easyocr` + `ru`,`en` | рабочий OCR по сканам |
| `rapidocr` | только для прогрева на старте, см. ниже |

Всё остальное из каталога docling — VLM-модели (`granitedocling`,
`smoldocling`, `granite_vision`), распознавание формул, классификатор
картинок — к задаче отношения не имеет и весит на порядок больше самого
пайплайна.

**`rapidocr` убрать нельзя**, хотя рабочий движок — EasyOCR:
docling-serve на старте прогревает пайплайн с дефолтными опциями, а
дефолтный движок выбирает RapidOCR, и без его весов контейнер падает с
`FileNotFoundError` ещё до первого запроса. Серверной настройки движка
по умолчанию пока нет — это открытый feature request в docling-serve.
Речь про mobile-модели PP-OCRv4 на пару десятков мегабайт.

**Русская модель EasyOCR качается отдельным шагом.** При заданном
`artifacts_path` docling считает режим офлайновым и отключает докачку
весов в рантайме, а `docling-tools` кладёт только детектор и латиницу.

**Веса bge-m3 в том не попадают.** Здесь `api` работает с хоста и держит
их в своём кэше HuggingFace, так что 2.3 ГБ в томе были бы второй
копией. Токенизатор при этом нужен всегда: по нему HybridChunker внутри
docling считает границы чанков, и он обязан совпадать с эмбеддером.

Пересобрать том с нуля:

```bash
docker compose -f docker-compose.test.yml down
docker volume rm rag_ingestion_service_docling-models
docker compose -f docker-compose.test.yml up -d
```

---

## Обновление кода

Зависимости не трогаются, перезапускаются только процессы на хосте:

```bash
git pull
alembic upgrade head          # если менялись модели
# Ctrl+C и заново в обоих терминалах
```

RQ держит модули в памяти процесса, поэтому воркер после правки задач
обязательно перезапускать — том с исходниками, как в контейнерном деве,
тут ни при чём.

Для долгого стенда процессы удобнее завернуть в systemd-юниты, чтобы они
переживали разрыв ssh-сессии. Юнитов в репозитории нет — `tmux` или
`screen` для стенда достаточно.

---

## Диагностика

**`docling-models-init` падает**
Смотри его лог: `docker compose -f docker-compose.test.yml logs
docling-models-init`. Чаще всего — нет сети до HuggingFace либо
кончилось место под том.

**`docling` падает с `FileNotFoundError` до первого запроса**
Том наполнен не полностью, обычно нет весов RapidOCR. Пересобери том.

**Документы вечно в `pending`**
Воркер не запущен, либо `INGEST_REDIS_URL` не совпадает у сервиса и
воркера. Смотри `rq info -u redis://localhost:6379/1`.

**`URL is not allowed` в логе docling**
В `ingest.env` стоит `INGEST_DOCLING_TRANSFER=url`, а allowlist docling
режет приватные адреса как SSRF. Для стенда нужен `upload`: воркер
скачивает файл сам и отправляет multipart'ом.

**`Connection refused` у воркера на `/embed`**
`INGEST_SELF_URL` указывает на платформенный порт вместо внутреннего.
Должно быть `http://localhost:8012` — см. [`ports.md`](ports.md).

**Порт занят на хосте**
Postgres уже поднят на 5437, чтобы не конфликтовать со штатным 5432.
Остальные (9200, 6379, 8333, 5601, 5001) при конфликте меняются в
`ports` соответствующего сервиса — внутри сети compose они остаются
прежними, так что править больше ничего не надо.