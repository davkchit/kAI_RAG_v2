# Stage 1: install deps (needs build-essential for some packages)
FROM python:3.11-slim AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# CPU-only torch BEFORE sentence-transformers to prevent CUDA 2GB pull
RUN pip install --no-cache-dir --prefix=/install \
    torch --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# Stage 2: lean runtime image (no build tools)
FROM python:3.11-slim

WORKDIR /app

COPY --from=builder /install /usr/local

# Pre-bake ML models into image for fast cold starts
RUN python -c "\
from fastembed import TextEmbedding, SparseTextEmbedding; \
TextEmbedding('intfloat/multilingual-e5-large'); \
SparseTextEmbedding('Qdrant/bm25')"

RUN python -c "\
from sentence_transformers import CrossEncoder; \
CrossEncoder('BAAI/bge-reranker-v2-m3')"

COPY . .

EXPOSE 8000

CMD ["python", "main.py"]
