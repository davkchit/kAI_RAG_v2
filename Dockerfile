# Stage 1: install deps
FROM python:3.11-slim AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# CPU wheel index as PRIMARY so torch (and any transitive dep) never pulls CUDA
RUN pip install --no-cache-dir --prefix=/install \
    --index-url https://download.pytorch.org/whl/cpu \
    --extra-index-url https://pypi.org/simple/ \
    torch -r requirements.txt

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
