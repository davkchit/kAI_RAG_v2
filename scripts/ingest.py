import hashlib
import json
import locale
import os
import sys
import uuid
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

locale.getpreferredencoding = lambda *_: "utf-8"
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import httpx
import opendataloader_pdf
from fastembed import SparseTextEmbedding
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient, models

JINA_MODEL = "jina-embeddings-v3"
JINA_DIMS = 1024
SPARSE_MODEL = "Qdrant/bm25"
DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "sparse"
COLLECTION_NAME = "university_docs_odl"

DATA_DIR = Path("data")
OUTPUT_DIR = Path("odl_output")
RECREATE_COLLECTION = True
UPLOAD_BATCH_SIZE = 32

JINA_API_KEY = os.getenv("JINA_API_KEY")

splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=150,
    separators=["\n\n", "\n", ". ", " "],
)

# Smaller chunks for legal docs — each clause stays isolated
splitter_legal = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=80,
    separators=["\n\n", "\n", ". ", " "],
)

client = QdrantClient(
    url=os.getenv("QDRANT_URL", "http://localhost:6333"),
    api_key=os.getenv("QDRANT_API_KEY") or None,
    timeout=120,
    trust_env=False,
    check_compatibility=False,
)

_bm25 = SparseTextEmbedding(SPARSE_MODEL)

FILTERABLE_FIELDS = ("source", "doc_group", "doc_type", "doc_scope", "program_level")


def embed_dense_batch(texts: list[str], task: str = "retrieval.passage") -> list[list[float]]:
    resp = httpx.post(
        "https://api.jina.ai/v1/embeddings",
        headers={"Authorization": f"Bearer {JINA_API_KEY}", "Content-Type": "application/json"},
        json={"model": JINA_MODEL, "input": texts, "task": task, "dimensions": JINA_DIMS},
        timeout=60,
    )
    resp.raise_for_status()
    return [item["embedding"] for item in resp.json()["data"]]


def embed_sparse_batch(texts: list[str]):
    return list(_bm25.embed(texts))


def setup_collection():
    if RECREATE_COLLECTION and client.collection_exists(COLLECTION_NAME):
        client.delete_collection(COLLECTION_NAME)
        print(f"Старая коллекция '{COLLECTION_NAME}' удалена.")

    if not client.collection_exists(COLLECTION_NAME):
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config={
                DENSE_VECTOR_NAME: models.VectorParams(
                    size=JINA_DIMS, distance=models.Distance.COSINE
                )
            },
            sparse_vectors_config={
                SPARSE_VECTOR_NAME: models.SparseVectorParams()
            },
        )
        print(f"Коллекция '{COLLECTION_NAME}' создана.")

    for field in FILTERABLE_FIELDS:
        client.create_payload_index(
            collection_name=COLLECTION_NAME,
            field_name=field,
            field_schema=models.PayloadSchemaType.KEYWORD,
        )


def make_chunk_id(source: str, page: int, chunk_index: int) -> str:
    key = f"{source}:{page}:{chunk_index}"
    return str(uuid.UUID(hashlib.md5(key.encode()).hexdigest()))


def upload_batches(documents, metadatas, ids):
    total = len(documents)
    for start in range(0, total, UPLOAD_BATCH_SIZE):
        end = min(start + UPLOAD_BATCH_SIZE, total)
        batch_docs = documents[start:end]
        batch_meta = metadatas[start:end]
        batch_ids = ids[start:end]

        print(f"Эмбеддинг чанков {start+1}-{end} из {total}...")
        dense_vecs = embed_dense_batch(batch_docs)
        sparse_vecs = embed_sparse_batch(batch_docs)

        points = []
        for doc, meta, uid, dv, sv in zip(batch_docs, batch_meta, batch_ids, dense_vecs, sparse_vecs):
            points.append(models.PointStruct(
                id=uid,
                vector={
                    DENSE_VECTOR_NAME: dv,
                    SPARSE_VECTOR_NAME: models.SparseVector(
                        indices=sv.indices.tolist(),
                        values=sv.values.tolist(),
                    ),
                },
                payload={**meta, "document": doc},
            ))

        client.upsert(collection_name=COLLECTION_NAME, points=points)
        print(f"  загружено {end} из {total}")


def build_preview_text(pages, page_limit=2, max_chars=2500):
    parts = []
    for _, text in pages[:page_limit]:
        compact = " ".join(text.split()).strip()
        if compact:
            parts.append(compact)
    return " ".join(parts)[:max_chars].lower()


def infer_document_profile(pdf_name, preview_text):
    lowered = pdf_name.lower()
    combined = f"{lowered}\n{preview_text}"
    profile = {"doc_group": "other", "doc_type": "reference", "doc_scope": "university", "program_level": "generic", "doc_title": pdf_name}

    if any(k in combined for k in ("набережночелнинский филиал", "нчф", "филиал книту-каи", "university.pdf", "nchf", "nabereghnochelninskogo", "filiala_knitu")):
        profile.update({"doc_group": "branch", "doc_type": "overview", "doc_scope": "branch", "program_level": "branch"})
        return profile

    if "правила приема" in combined:
        profile.update({"doc_group": "admission", "doc_type": "rules", "doc_scope": "university"})
        if any(k in combined for k in ("аспирантур", "научно-педагогических")):
            profile["program_level"] = "asp"
        elif any(k in combined for k in ("среднего профессионального", "спо")):
            profile["program_level"] = "spo"
        else:
            profile["program_level"] = "bo"
        return profile

    if any(k in combined for k in ("образовательных отношений", "отчисл", "перевод", "восстанов")):
        profile.update({"doc_group": "regulations", "doc_type": "regulation", "doc_scope": "university"})
        return profile

    return profile


def table_to_text(node):
    rows = node.get("rows", [])
    if not rows:
        return str(node.get("content", "")).strip()
    row_texts = []
    for row in rows:
        cell_values = []
        for cell in row.get("cells", []):
            parts = [str(kid.get("content", "")).strip() for kid in cell.get("kids", []) if isinstance(kid, dict) and kid.get("content")]
            cell_values.append(" ".join(parts).strip())
        if any(cell_values):
            row_texts.append(" | ".join(cell_values))
    return "\n".join(row_texts).strip()


def walk_elements(node):
    if isinstance(node, list):
        for item in node:
            yield from walk_elements(item)
        return
    if not isinstance(node, dict):
        return
    node_type = node.get("type")
    page_number = node.get("page number")
    if node_type in {"heading", "paragraph", "caption", "list item"}:
        content = str(node.get("content", "")).strip()
        if page_number and content:
            yield page_number, content
    if node_type == "table":
        table_text = table_to_text(node)
        if page_number and table_text:
            yield page_number, table_text
    yield from walk_elements(node.get("kids", []))
    yield from walk_elements(node.get("list items", []))


def extract_pages_from_json(json_path):
    data = json.loads(json_path.read_text(encoding="utf-8"))
    pages = {}
    for page_number, text in walk_elements(data.get("kids", [])):
        pages.setdefault(page_number, []).append(text)
    result = []
    for page_number in sorted(pages):
        page_text = "\n\n".join(pages[page_number]).strip()
        if page_text:
            result.append((page_number, page_text))
    return result


def extract_pages_from_md(md_path: Path) -> list[tuple[int, str]]:
    import re
    text = md_path.read_text(encoding="utf-8")
    sections = re.split(r'\n(?=#{1,3} )', text)
    return [(i + 1, s.strip()) for i, s in enumerate(sections) if s.strip()]


def main():
    if not JINA_API_KEY:
        raise RuntimeError("JINA_API_KEY не задан в .env")

    pdf_files = sorted(DATA_DIR.glob("*.pdf"))
    md_files  = sorted(DATA_DIR.glob("*.md"))

    if not pdf_files and not md_files:
        print("В папке data нет PDF или MD файлов.")
        return

    setup_collection()

    documents, metadatas, ids = [], [], []

    # ── PDF files ──────────────────────────────────────────────────────────
    if pdf_files:
        if OUTPUT_DIR.exists():
            import subprocess
            subprocess.run(["cmd", "/c", f"rmdir /s /q {OUTPUT_DIR}"], check=False)
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        os.environ["PYTHONIOENCODING"] = "utf-8"
        os.environ.setdefault("JAVA_TOOL_OPTIONS", "-Dfile.encoding=UTF-8")
        opendataloader_pdf.convert(
            input_path=[str(p) for p in pdf_files],
            output_dir=str(OUTPUT_DIR),
            format="json,markdown",
            use_struct_tree=True,
        )

        for pdf in pdf_files:
            json_path = OUTPUT_DIR / f"{pdf.stem}.json"
            if not json_path.exists():
                print(f"Нет JSON для {pdf.name}")
                continue
            pages = extract_pages_from_json(json_path)
            preview = build_preview_text(pages)
            profile = infer_document_profile(pdf.name, preview)
            print(f"Обработан (PDF): {pdf.name}")
            _splitter = splitter_legal if profile["doc_group"] in ("admission", "regulations") else splitter
            for page_number, page_text in pages:
                for i, chunk in enumerate(_splitter.split_text(page_text)):
                    chunk = chunk.strip()
                    if not chunk:
                        continue
                    documents.append(chunk)
                    metadatas.append({"source": pdf.name, "page": page_number, "chunk_index": i, "parser": "opendataloader", **profile})
                    ids.append(make_chunk_id(pdf.name, page_number, i))

    # ── Markdown files ─────────────────────────────────────────────────────
    for md in md_files:
        pages = extract_pages_from_md(md)
        preview = build_preview_text(pages)
        profile = infer_document_profile(md.name, preview)
        print(f"Обработан (MD): {md.name}")
        _splitter = splitter_legal if profile["doc_group"] in ("admission", "regulations") else splitter
        for page_number, page_text in pages:
            for i, chunk in enumerate(_splitter.split_text(page_text)):
                chunk = chunk.strip()
                if not chunk:
                    continue
                documents.append(chunk)
                metadatas.append({"source": md.name, "page": page_number, "chunk_index": i, "parser": "markdown", **profile})
                ids.append(make_chunk_id(md.name, page_number, i))

    if not documents:
        print("Нет документов для индексации.")
        return

    print(f"Подготовлено {len(documents)} чанков.")
    upload_batches(documents, metadatas, ids)
    print("Готово!")


if __name__ == "__main__":
    main()
