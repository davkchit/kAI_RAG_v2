FROM python:3.11-slim AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

FROM python:3.11-slim

WORKDIR /app

COPY --from=builder /install /usr/local

# Pre-bake ML models (no reranker = no PyTorch = ~800MB image)
RUN python -c "\
from fastembed import TextEmbedding, SparseTextEmbedding; \
TextEmbedding('intfloat/multilingual-e5-small'); \
SparseTextEmbedding('Qdrant/bm25')"

COPY . .

EXPOSE 8000

CMD ["python", "main.py"]
