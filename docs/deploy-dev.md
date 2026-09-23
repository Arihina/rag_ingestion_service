# Развёртывание: dev

Зависимости в Docker, сам сервис — с хоста. Так быстрее во всём: образ
`api` весит около 6 ГБ из-за torch и собирается минутами, отладчик к
процессу в контейнере не подцепить, а перезапуск после правки моделей всё
равно полный.

---

## 1. Хост: каталог под веса

Один раз, до первого запуска.

```bash
mkdir -p /home/dodo/storage/docling-models
sudo chown -R 1001:0 /home/dodo/storage/docling-models
sudo chmod -R g+rwX /home/dodo/storage/docling-models
```

UID 1001 — тот же, под которым работают `docling-serve` и наши образы. Том
с весами шарится между ними, и без совпадения UID сервис-инициализатор
положит файлы, которые никто не сможет прочитать.

Проверь `vm.max_map_count`, иначе OpenSearch не поднимется:

```bash
sysctl vm.max_map_count          # нужно >= 262144
sudo sysctl -w vm.max_map_count=262144
```

Чтобы пережило перезагрузку — строка `vm.max_map_count=262144` в
`/etc/sysctl.d/99-opensearch.conf`.

---

## 2. Конфиги

```bash
cp ingest.env.example ingest.env
ls -la deploy/seaweedfs-s3.json     # должен быть ФАЙЛ, не каталог
```

Вторая проверка не формальность. При bind-mount одиночного файла Docker
молча создаёт на его месте пустой **каталог**, если файла нет, и SeaweedFS
падает с `is a directory` — причём уже после того, как master, volume и
filer успешно поднялись, так что в общем логе это выглядит как отказ всего
сервиса. Сейчас монтируется каталог `./deploy` целиком, но если файла нет,
S3-шлюз всё равно не стартует.

В `ingest.env` обрати внимание на два ключа:

```
# Адрес, попадающий В САМУ presigned-ссылку. По ней приходит docling,
# а он в контейнере: localhost для него — он сам, а не SeaweedFS.
INGEST_S3_PRESIGN_ENDPOINT_URL=http://seaweedfs:8333

# url — docling качает сам по ссылке (нужен проходящий allowlist);
# upload — качает воркер и шлёт multipart (allowlist не задействуется).
INGEST_DOCLING_TRANSFER=upload
```

Остальные адреса в `ingest.env` указывают на `localhost` — это верно,
потому что сервис живёт на хосте, а порты зависимостей проброшены.

---

## 3. Зависимости

```bash
docker compose -f docker-compose.dev.yml up -d
```

Поднимаются семь сервисов: `opensearch`, `dashboards`, `postgres`,
`redis`, `seaweedfs`, `docling-models-init`, `docling`. Сервисы `api` и
`worker` помечены профилем `app` и по умолчанию не стартуют.

Первый запуск долгий: `docling-models-init` качает layout, tableformer,
easyocr, rapidocr и веса bge-m3 целиком — около 5 ГБ. `docling` стартует
только после его успешного выхода.

```bash
docker compose -f docker-compose.dev.yml ps
```

Все должны быть `healthy` или `Up`, а `docling-models-init` — `Exited (0)`.

---

## 4. Python-окружение

```bash
python3 -m venv venv && source venv/bin/activate

# torch ПЕРВЫМ и явно: FlagEmbedding иначе подтянет сборку с дефолтного
# индекса PyPI, а Blackwell (sm_120) требует CUDA >= 12.8
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt -r requirements-embed.txt
```

---

## 5. Миграции

```bash
alembic revision --autogenerate -m "control plane"
alembic upgrade head
psql postgresql://rag:rag@localhost:5437/rag_ingest -c '\dt'
```

Ожидаются пять таблиц: `rag_sets`, `documents`, `import_batches`,
`ingest_jobs`, `alembic_version`.

**Пустой `alembic/versions/` — не ошибка, а признак того, что миграция ещё
не сгенерирована.** `upgrade head` в этом случае отрабатывает вхолостую и
молча, а первый же запрос падает с `relation "rag_sets" does not exist`.

Посмотри сгенерированный файл глазами: автогенерация иногда пропускает
частичные индексы с `WHERE`, а у нас их два на `rag_sets`. Без
`uq_rag_sets_owner_name` дубли имён пройдут молча.

---

## 6. Запуск

Два терминала.

```bash
python -m app.main
```

Поднимаются два порта: 8011 (`/health`, `/v1/platform/*`) и 8012
(`/embed`, `/v1/internal/*`, `/admin/*`). Зачем — в
[`ports.md`](ports.md).

```bash
rq worker -u redis://localhost:6379/1 ingest maintenance
```

Воркер запускать **из корня репозитория**: `Settings` читает `ingest.env`
относительно текущего каталога, а воркеру нужны и адрес docling, и
`INGEST_SELF_URL`.

```bash
curl -s localhost:8011/health | jq
```

Все пять флагов должны быть `true`.

---

## 7. Когда всё-таки нужны контейнеры

Проверить саму сборку или воспроизвести прод:

```bash
# WITH_EMBEDDINGS=1 обязателен: в контейнере api берёт веса bge-m3 из
# тома моделей, а по умолчанию туда кладётся только токенизатор —
# в основном режиме дева веса живут в хостовом кэше HuggingFace.
WITH_EMBEDDINGS=1 docker compose -f docker-compose.dev.yml --profile app up -d --build
docker compose -f docker-compose.dev.yml exec api alembic upgrade head
```

Тогда исходники монтируются томом в оба контейнера, но перечитывает их
только `api` при `INGEST_RELOAD=true`. RQ держит модули в памяти процесса,
поэтому после правки задач — `docker compose restart worker`.

---

## Что чем правится

| Изменилось | Достаточно |
|---|---|
| `docker-compose*.yml` | `up -d` |
| код в `app/` (с хоста) | перезапуск процесса |
| код в `app/` (в контейнере) | `restart api worker` |
| `requirements*.txt`, `Dockerfile` | `up -d --build` |
| модели набора в `app/db/models.py` | `alembic revision --autogenerate` + `upgrade head` |

---

## Частые грабли

**`seaweedfs` падает с `is a directory`**
На месте `deploy/seaweedfs-s3.json` каталог, созданный Docker. Снести
(`sudo rm -rf`), положить файл, пересоздать контейнер.

**`seaweedfs` работает, но `unhealthy`**
Проверка идёт на `127.0.0.1`, а не `localhost`: `/etc/hosts` в alpine
резолвит `localhost` в `::1`, и weed, слушающий только IPv4, не отвечает.
Если правил compose руками — верни IPv4-адрес.

**`docling` падает с `FileNotFoundError` до первого запроса**
Не скачаны веса RapidOCR. Даже при работе на EasyOCR они нужны для
прогрева пайплайна с дефолтными опциями. Проверь, что
`docling-models-init` вышел с кодом 0.

**`relation "rag_sets" does not exist`**
Миграция не сгенерирована. См. шаг 5.

**Документы вечно в `pending`**
Воркер не запущен, либо `INGEST_REDIS_URL` не совпадает у сервиса и
воркера. Смотри `rq info -u redis://localhost:6379/1`.

**`URL is not allowed` в логе docling**
Его allowlist для `http_sources` режет приватные адреса как SSRF. Обходной
путь — `INGEST_DOCLING_TRANSFER=upload`.

**`ImportError: cannot import name 'X' from 'app.api'`**
Не хватает файла `app/api/X.py`. Пакеты намеренно не делают жадных
`from . import ...`, чтобы Python называл отсутствующий модуль, а не
рассуждал про циклический импорт.

**`X-User-Id должен быть UUID` при явно заданной переменной**
Переменные окружения не переживают смену терминала. В каждом новом окне:

```bash
export API=http://localhost:8011
export U=11111111-1111-1111-1111-111111111111
export RAG=<id набора>
```

**Пустой результат поиска при непустом `chunks_total`**
Скорее всего `$RAG` пуст, и `term rag_id: ""` честно ничего не находит.
Проверь `echo "[$RAG]"` и повтори с `match_all`.