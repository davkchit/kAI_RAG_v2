# kAI — RAG-чатбот НЧФ КНИТУ-КАИ

Умный помощник для студентов и абитуриентов НЧФ КНИТУ-КАИ. Работает в Telegram и на сайте. Отвечает на вопросы о поступлении, программах, жизни в филиале — на основе официальных документов.

## Стек

| Компонент | Технология |
|---|---|
| Эмбеддинги | Jina AI API (`jina-embeddings-v3`) |
| Реранкер | Jina AI API (`jina-reranker-v2-base-multilingual`) |
| Sparse поиск | BM25 (fastembed, локально) |
| Векторная БД | Qdrant Cloud |
| LLM | Groq API (`llama-3.3-70b-versatile`) |
| Telegram бот | aiogram 3.x |
| API сервер | FastAPI + uvicorn |
| Хостинг | Railway |

---

## Быстрый старт

### 1. Клонировать репо

```bash
git clone https://github.com/davkchit/kAI_RAG_v2.git
cd kAI_RAG_v2
```

### 2. Установить зависимости

```bash
pip install -r requirements.txt
pip install opendataloader-pdf  # только для переиндексации PDF
```

> Нужен Python 3.11+ и Java (для opendataloader при инджесте)

### 3. Создать .env

```
GROQ_API_KEY=gsk_...
JINA_API_KEY=jina_...
QDRANT_URL=https://xxx.aws.cloud.qdrant.io:6333
QDRANT_API_KEY=eyJhbGci...
TELEGRAM_BOT_TOKEN=123456:ABC...
```

### 4. Проиндексировать документы

Положи PDF-файлы в папку `data/` и запусти:

```bash
python scripts/ingest.py
```

Это создаст коллекцию в Qdrant Cloud и загрузит все чанки через Jina API.

### 5. Запустить локально

```bash
# Только API сервер (для чат-виджета)
python api.py

# Только Telegram бот
python bot.py

# Всё вместе (API + бот)
python main.py
```

---

## Деплой на Railway

### Один раз

1. Зарегайся на [railway.app](https://railway.app)
2. New Project → Deploy from GitHub → выбери `davkchit/kAI_RAG_v2`
3. Railway сам найдёт `Dockerfile`
4. В **Variables** добавь все 5 переменных из `.env`
5. Deploy

### При обновлении кода

```bash
git add .
git commit -m "описание изменений"
git push
```

Railway задеплоит автоматически.

### Если добавил новые PDF

```bash
# Локально переиндексируй
python scripts/ingest.py

# Данные улетят в Qdrant Cloud — Railway подхватит без передеплоя
```

---

## Нужные API-ключи

| Сервис | Где взять | Бесплатно |
|---|---|---|
| **Groq** | [console.groq.com](https://console.groq.com) | Да |
| **Jina AI** | [jina.ai](https://jina.ai) | Да (1M токенов/мес) |
| **Qdrant Cloud** | [cloud.qdrant.io](https://cloud.qdrant.io) | Да (1 кластер) |
| **Telegram** | [@BotFather](https://t.me/BotFather) | Да |

---

## Структура проекта

```
kAI_RAG_v2/
├── data/                  # PDF-документы для индексации
├── scripts/
│   ├── ingest.py          # Парсинг PDF → эмбеддинги → Qdrant
│   └── rag.py             # Поиск + реранкинг + генерация ответа
├── api.py                 # FastAPI сервер (POST /ask, GET /health)
├── bot.py                 # Telegram бот
├── main.py                # Запускает API + бот вместе
├── Dockerfile             # Для Railway
├── railway.toml           # Конфиг Railway
├── requirements.txt       # Зависимости
└── DOWNGRADE_LOG.txt      # Что упрощено для бесплатного Railway
```

---

## Как работает поиск

1. Вопрос пользователя → Jina API → dense вектор
2. Вопрос → BM25 → sparse вектор  
3. Семантический роутер определяет нужный раздел документов (поступление / аспирантура / СПО / регламенты / филиал)
4. Qdrant выполняет гибридный поиск (RRF fusion)
5. Jina Reranker переранжирует топ-20 результатов → берём топ-5
6. Groq LLM генерирует ответ на основе найденных чанков

---

## Чат-виджет для сайта

Виджет находится в `nchf-kai/components/kai/KAIChat.tsx`. Подключён в `app/layout.tsx`.

В `.env.local` сайта нужно добавить:
```
NEXT_PUBLIC_KAI_API_URL=https://твой-railway-url.up.railway.app
```
