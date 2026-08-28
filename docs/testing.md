# Тестирование

Два уровня: юнит-тесты, которым ничего не нужно, и ручной интеграционный
прогон на поднятом окружении.

---

## Юнит-тесты

```bash
source venv/bin/activate
python -m unittest discover -s tests -t .
```

137 тестов, около секунды. Внешних сервисов не требуется: Postgres, Redis,
SeaweedFS, OpenSearch и docling подменяются. Модель эмбеддингов не
загружается — `FlagEmbedding` импортируется лениво внутри `BgeM3Embedder`,
и до него тесты не доходят.

```bash
python -m unittest tests.test_archive                     # один модуль
python -m unittest tests.test_archive.ArchiveRejectionTests -v
python -m unittest discover -s tests -t . -f              # стоп на первом падении
```

`-t .` обязателен: он задаёт корень для импортов, иначе `import app` не
разрешится.

### Что где

| Модуль | Предмет |
|---|---|
| `test_chunk_identity.py` | формула `_id`, коллизии между наборами, идемпотентность |
| `test_index_mapping.py` | маппинг `kb-v2`: анализаторы, размерность, метрика |
| `test_search_isolation.py` | фильтр `rag_id` в запросах, `refresh` при bulk |
| `test_scoring.py` | косинус ↔ `_score` и обратимость преобразования |
| `test_archive.py` | zip: бомба, traversal, коллизии имён, лимиты |
| `test_docling_client.py` | разбор чанков, форма запроса v1, опрос задачи |
| `test_options.py` | имена полей опций: JSON против multipart |
| `test_status.py` | производный статус набора |
| `test_schemas.py` | границы конфига набора |
| `test_repo_queries.py` | владение внутри SQL, инварианты схемы |
| `test_storage.py` | ключи, подпись ссылок под чужим хостом, стрим |
| `test_worker_jobs.py` | режимы передачи файла в docling, пул эмбеддера |
| `test_api_rags.py` | коды ответов, скоупинг, валидация, иконки |

### Принципы

**Приоритет — то, что ломается молча.** Каждая проверка здесь выросла из
реальной ошибки, не дававшей исключения: коллизия `_id` между наборами;
имя объекта опций docling (`options` вместо `convert_options` — pydantic на
той стороне выбрасывает неизвестные поля без предупреждения); `_score`,
принятый за косинус; забытый фильтр `rag_id`; отсутствие `refresh` при
bulk.

**Транспорт мокается, а не методы.** `httpx.MockTransport` вместо патча
`post`: проверяется настоящее тело запроса в том виде, в каком его
сериализует httpx, вместе с multipart-границами.

**SQL проверяется по структуре.** Живого Postgres нет, поэтому запросы
компилируются под диалект и проверяются на наличие условий владения.

**API тестируется на настоящем приложении.** Не сборкой роутеров руками:
без обработчиков ошибок из `main.py` невалидное тело давало бы `422`
вместо `400`. `TestClient` без контекстного менеджера lifespan не
запускает, поэтому модели не грузятся.

### Фикстуры

Создаются генератором автоматически при первом запуске. Вручную:

```bash
python -m tests.fixtures_build
```

Тесты проверяют структуру и диапазоны — коэффициент сжатия выше порога,
записей больше лимита, имя после сплющивания такое-то, — а не конкретные
байты, поэтому генерация безопаснее хранения бинарников в git. Описание
каждой фикстуры — в `tests/data/README.md`.

### Если прогон подвис

Тексты ошибок unittest печатает **в конце**, поэтому подвисший прогон
выглядит как молчание. Локализовать:

```bash
for m in tests/test_*.py; do
  name=$(basename $m .py)
  echo "== $name"
  timeout 60 python -m unittest tests.$name 2>&1 | tail -3
done
```

Либо `Ctrl+C` — unittest напечатает трейсбек того места, где стоял.
Все сетевые клиенты в тестах подменены заглушками с таймаутом в секунду,
так что уход в настоящую сеть даёт падение, а не зависание.

---

## Ручной интеграционный прогон

Нужны поднятые зависимости, запущенные сервис и воркер. Переменные не
переживают смену терминала — задавай в каждом окне:

```bash
export API=http://localhost:8000
export U=11111111-1111-1111-1111-111111111111
export V=22222222-2222-2222-2222-222222222222
```

### 1. Инфраструктура

```bash
curl -s $API/health | jq
```

Все пять флагов `true`.

### 2. Набор и валидация

```bash
RAG=$(curl -s -X POST $API/v1/platform/rags \
  -H "Content-Type: application/json" -H "X-User-Id: $U" \
  -d '{"name":"Регламенты","top_k":5,"score_threshold":0.4}' | jq -r .id)
echo $RAG

# все три должны дать 400
for body in '{"name":"x","top_k":50}' '{"name":"x","temperature":3}' '{"name":"x","score_threshold":1.5}'; do
  curl -s -o /dev/null -w "%{http_code} " -X POST $API/v1/platform/rags \
    -H "Content-Type: application/json" -H "X-User-Id: $U" -d "$body"
done; echo

# дубль имени -> 409
curl -s -o /dev/null -w "%{http_code}\n" -X POST $API/v1/platform/rags \
  -H "Content-Type: application/json" -H "X-User-Id: $U" -d '{"name":"Регламенты"}'
```

### 3. Изоляция

```bash
curl -s -o /dev/null -w "%{http_code}\n" $API/v1/platform/rags/$RAG -H "X-User-Id: $V"  # 404
curl -s $API/v1/platform/rags -H "X-User-Id: $V" | jq                                    # []
```

`404`, а не `403`: существование чужого набора — тоже информация.

### 4. Загрузка

```bash
printf 'Регламент обслуживания РЩ-3.\nПоверка производится ежегодно.\n' > /tmp/doc.txt
curl -s -X POST $API/v1/platform/rags/$RAG/documents -H "X-User-Id: $U" \
  -F "files=@/tmp/doc.txt" | jq

watch -n2 "curl -s $API/v1/platform/rags/$RAG -H 'X-User-Id: $U' \
  | jq '{status, documents, chunks_total}'"
```

Переходы: `empty → ingesting → ready`. Ошибки по документам:

```bash
curl -s $API/v1/platform/rags/$RAG/documents -H "X-User-Id: $U" \
  | jq '.[] | {filename, status, error, chunks_count}'
```

### 5. Дедупликация

```bash
curl -s -X POST $API/v1/platform/rags/$RAG/documents -H "X-User-Id: $U" \
  -F "files=@/tmp/doc.txt" | jq '.documents[0]'
```

`duplicate_of` заполнен, новый документ не создан.

### 6. Идемпотентность ключа — главная проверка

Ловит ровно тот класс багов, ради которого делался этап 0.

```bash
RAG2=$(curl -s -X POST $API/v1/platform/rags -H "Content-Type: application/json" \
  -H "X-User-Id: $U" -d '{"name":"Второй"}' | jq -r .id)

BEFORE=$(curl -s "localhost:9200/kb-v2/_count" -H 'Content-Type: application/json' \
  -d "{\"query\":{\"term\":{\"rag_id\":\"$RAG\"}}}" | jq .count)

# файл с ТЕМ ЖЕ именем, но другим содержимым
printf 'Совсем другой документ.\n' > /tmp/doc2.txt && mv /tmp/doc2.txt /tmp/doc.txt
curl -s -X POST $API/v1/platform/rags/$RAG2/documents -H "X-User-Id: $U" \
  -F "files=@/tmp/doc.txt" > /dev/null

sleep 15
AFTER=$(curl -s "localhost:9200/kb-v2/_count" -H 'Content-Type: application/json' \
  -d "{\"query\":{\"term\":{\"rag_id\":\"$RAG\"}}}" | jq .count)
echo "было $BEFORE, стало $AFTER — должны совпасть"
```

Со старой формулой `blake2b(source_uri:chunk_index)` второй набор молча
затирал бы чанки первого: без ошибки, без конфликта.

### 7. Содержимое индекса

```bash
curl -s "localhost:9200/kb-v2/_search?size=5" -H 'Content-Type: application/json' \
  -d '{"query":{"range":{"chunk_index":{"gte":3}}},
       "_source":["chunk_index","headings","pages","document_id"]}' \
  | jq '.hits.hits[]._source'
```

`headings` у чанков из середины документа обязан быть непустым — иначе
breadcrumb не собирается, и `_extract_texts` в `app/docling/client.py`
забирает не то поле ответа.

### 8. `/embed`

```bash
curl -s -X POST $API/embed -H "Content-Type: application/json" \
  -d '{"texts":["поверка приборов","расторжение договора"],"pool":"query"}' \
  | jq '{model, n:(.embeddings|length), dim:(.embeddings[0].dense|length),
         sparse:(.embeddings[0].sparse|length)}'
```

`dim: 1024`, `sparse` непустой. С `"pool":"ingest"` — то же самое, это путь
воркера.

### 9. Внутренняя ручка

```bash
curl -s "$API/v1/internal/rags/$RAG?user_id=$U" | jq
curl -s -o /dev/null -w "%{http_code}\n" "$API/v1/internal/rags/$RAG?user_id=$V"  # 404

# правка конфига видна сразу
curl -s -X PATCH $API/v1/platform/rags/$RAG -H "Content-Type: application/json" \
  -H "X-User-Id: $U" -d '{"prompt":"Отвечай кратко.","temperature":0.1}' > /dev/null
curl -s "$API/v1/internal/rags/$RAG?user_id=$U" | jq '{prompt,temperature}'
```

### 10. Иконки

```bash
curl -s -X PUT $API/v1/platform/rags/$RAG/icon -H "X-User-Id: $U" \
  -F "file=@tests/data/icons/large.png" | jq .has_icon
curl -s $API/v1/platform/rags/$RAG/icon -H "X-User-Id: $U" -o /tmp/out.png
file /tmp/out.png                                    # PNG, не больше 256 px

curl -s -o /dev/null -w "%{http_code}\n" -X PUT $API/v1/platform/rags/$RAG/icon \
  -H "X-User-Id: $U" \
  -F "file=@tests/data/icons/evil.svg;type=image/svg+xml"          # 415
```

### 11. Архивы

```bash
B=$(curl -s -X POST $API/v1/platform/rags/$RAG/imports/archive -H "X-User-Id: $U" \
  -F "file=@tests/data/archives/plain.zip" | jq -r .id)
sleep 5
curl -s $API/v1/platform/rags/$RAG/imports/$B -H "X-User-Id: $U" | jq

# бомба -> rejected, набор при этом цел
B2=$(curl -s -X POST $API/v1/platform/rags/$RAG/imports/archive -H "X-User-Id: $U" \
  -F "file=@tests/data/archives/bomb.zip" | jq -r .id)
sleep 3
curl -s $API/v1/platform/rags/$RAG/imports/$B2 -H "X-User-Id: $U" | jq '{status,error}'
```

### 12. Хранилище

```bash
AWS_ACCESS_KEY_ID=rag AWS_SECRET_ACCESS_KEY=rag \
  aws --endpoint-url http://localhost:8333 s3 ls s3://rag-docs/$RAG/ --recursive
```

Структура `{rag_id}/{document_id}/{filename}`. В `rag-staging` после
успешного разворачивания архива пусто — он удаляется в `finally`.

### 13. Удаление и уборка

```bash
DOC=$(curl -s $API/v1/platform/rags/$RAG/documents -H "X-User-Id: $U" | jq -r '.[0].id')
curl -s -o /dev/null -w "%{http_code}\n" -X DELETE \
  $API/v1/platform/rags/$RAG/documents/$DOC -H "X-User-Id: $U"          # 204
sleep 5
curl -s "localhost:9200/kb-v2/_count" -H 'Content-Type: application/json' \
  -d "{\"query\":{\"term\":{\"document_id\":\"$DOC\"}}}" | jq .count     # 0

curl -s -o /dev/null -w "%{http_code}\n" -X DELETE \
  $API/v1/platform/rags/$RAG -H "X-User-Id: $U"                         # 204
curl -s -o /dev/null -w "%{http_code}\n" $API/v1/platform/rags/$RAG \
  -H "X-User-Id: $U"                                                    # 404 сразу
sleep 5
curl -s "localhost:9200/kb-v2/_count" -H 'Content-Type: application/json' \
  -d "{\"query\":{\"term\":{\"rag_id\":\"$RAG\"}}}" | jq .count          # 0 после уборки
```

Здесь важен разрыв: `404` приходит мгновенно (soft-delete), а чанки и
объекты убираются фоном.

---

## Что не покрыто

**Автоматической интеграции нет.** Прогон выше делается руками. Довести
его до CI можно, но нужен поднятый compose и GPU для docling, так что это
отдельный набор, а не расширение юнит-тестов.

**`import_s3_prefix` и `expand_archive`** покрыты только через `plan()` —
сами задачи в юнит-тестах не гоняются.

**Операторский импорт из staging** в ручном прогоне тоже отсутствует: для
него надо сначала положить файлы в `rag-staging` через S3-клиент, затем
дёрнуть `POST /admin/rags/{id}/imports/s3` с `{"prefix":"..."}`.

**Качество поиска.** Меряется отдельно, через `_rank_eval` по NDCG@10 на
размеченном наборе запросов, и к этим тестам отношения не имеет.