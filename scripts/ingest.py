import hashlib
import json
import locale
import os
import shutil
import uuid
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

# opendataloader runner uses locale.getpreferredencoding() for subprocess stdout;
# on Windows this returns cp1251 which can't decode UTF-8 JAR output (e.g. Cyrillic paths)
locale.getpreferredencoding = lambda *_: "utf-8"

import opendataloader_pdf
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient, models

DENSE_MODEL = "intfloat/multilingual-e5-small"
SPARSE_MODEL = "Qdrant/bm25"
COLLECTION_NAME = "university_docs_odl"

DATA_DIR = Path("data")
OUTPUT_DIR = Path("odl_output")
RECREATE_COLLECTION = True
UPLOAD_BATCH_SIZE = 64
FILTERABLE_FIELDS = (
    "source",
    "doc_group",
    "doc_type",
    "doc_scope",
    "program_level",
)

splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=150,
    separators=["\n\n", "\n", ". ", " "],
)

client = QdrantClient(
    url=os.getenv("QDRANT_URL", "http://localhost:6333"),
    api_key=os.getenv("QDRANT_API_KEY") or None,
    timeout=60,
    trust_env=False,
    check_compatibility=False,
)


def setup_collection():
    client.set_model(DENSE_MODEL)
    client.set_sparse_model(SPARSE_MODEL)

    if RECREATE_COLLECTION and client.collection_exists(COLLECTION_NAME):
        client.delete_collection(COLLECTION_NAME)
        print(f"Старая коллекция '{COLLECTION_NAME}' удалена.")

    if not client.collection_exists(COLLECTION_NAME):
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=client.get_fastembed_vector_params(),
            sparse_vectors_config=client.get_fastembed_sparse_vector_params(),
        )
        print(f"Коллекция '{COLLECTION_NAME}' создана.")
    else:
        print(f"Коллекция '{COLLECTION_NAME}' уже существует.")

    ensure_payload_indexes()


def ensure_payload_indexes():
    for field_name in FILTERABLE_FIELDS:
        client.create_payload_index(
            collection_name=COLLECTION_NAME,
            field_name=field_name,
            field_schema=models.PayloadSchemaType.KEYWORD,
        )


def build_preview_text(pages, page_limit: int = 2, max_chars: int = 2500):
    preview_parts = []

    for _, page_text in pages[:page_limit]:
        compact = " ".join(page_text.split()).strip()
        if compact:
            preview_parts.append(compact)

    return " ".join(preview_parts)[:max_chars].lower()


def infer_document_profile(pdf_name: str, preview_text: str):
    lowered_name = pdf_name.lower()
    combined = f"{lowered_name}\n{preview_text}"

    profile = {
        "doc_group": "other",
        "doc_type": "reference",
        "doc_scope": "university",
        "program_level": "generic",
        "doc_title": pdf_name,
    }

    if any(
        keyword in combined
        for keyword in (
            "набережночелнинский филиал",
            "нчф",
            "филиал книту-каи",
            "university.pdf",
        )
    ):
        profile.update(
            {
                "doc_group": "branch",
                "doc_type": "overview",
                "doc_scope": "branch",
                "program_level": "branch",
            }
        )
        return profile

    if "правила приема" in combined:
        profile.update(
            {
                "doc_group": "admission",
                "doc_type": "rules",
                "doc_scope": "university",
            }
        )

        if "правила приема асп" in lowered_name:
            profile["program_level"] = "asp"
        elif "правила приема спо" in lowered_name:
            profile["program_level"] = "spo"
        elif "правила приема bo" in lowered_name:
            profile["program_level"] = "bo"
        elif any(keyword in combined for keyword in ("аспирантур", "аспирант", "научно-педагогических кадров")):
            profile["program_level"] = "asp"
        elif any(keyword in combined for keyword in ("среднего профессионального", "правила приема спо", "правила приема spo", "спо")):
            profile["program_level"] = "spo"
        elif any(keyword in combined for keyword in ("бакалавриат", "специалитет", "магистратур", "правила приема bo", "правила приема во", " bo")):
            profile["program_level"] = "bo"

        return profile

    if any(
        keyword in combined
        for keyword in (
            "образовательных отношений",
            "приостанов",
            "прекращени",
            "отчисл",
            "перевод",
            "восстанов",
            "порядок оформления",
        )
    ):
        profile.update(
            {
                "doc_group": "regulations",
                "doc_type": "regulation",
                "doc_scope": "university",
            }
        )
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
            parts = []

            for kid in cell.get("kids", []):
                if isinstance(kid, dict):
                    text = str(kid.get("content", "")).strip()
                    if text:
                        parts.append(text)

            cell_text = " ".join(parts).strip()
            cell_values.append(cell_text)

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


def extract_pages_from_json(json_path: Path):
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


def make_chunk_id(source: str, page: int, chunk_index: int) -> str:
    key = f"{source}:{page}:{chunk_index}"
    return str(uuid.UUID(hashlib.md5(key.encode()).hexdigest()))


def upload_batches(documents, metadatas, ids):
    total = len(documents)

    for start in range(0, total, UPLOAD_BATCH_SIZE):
        end = start + UPLOAD_BATCH_SIZE
        print(f"Загружаю чанки {start + 1}-{min(end, total)} из {total}...")
        client.add(
            collection_name=COLLECTION_NAME,
            documents=documents[start:end],
            metadata=metadatas[start:end],
            ids=ids[start:end],
            batch_size=32,
        )


def main():
    pdf_files = sorted(DATA_DIR.glob("*.pdf"))

    if not pdf_files:
        print("В папке data нет PDF-файлов.")
        return

    setup_collection()

    if OUTPUT_DIR.exists():
        # shutil.rmtree fails on Windows with long Cyrillic paths (WinError 145);
        # use system rmdir which handles them correctly
        import subprocess as _sp
        _sp.run(["cmd", "/c", f"rmdir /s /q {OUTPUT_DIR}"], check=False)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("JAVA_TOOL_OPTIONS", "-Dfile.encoding=UTF-8")

    opendataloader_pdf.convert(
        input_path=[str(pdf) for pdf in pdf_files],
        output_dir=str(OUTPUT_DIR),
        format="json,markdown",
        use_struct_tree=True,
    )

    documents = []
    metadatas = []
    ids = []

    for pdf in pdf_files:
        json_path = OUTPUT_DIR / f"{pdf.stem}.json"

        if not json_path.exists():
            print(f"Не найден JSON-результат для {pdf.name}")
            continue

        pages = extract_pages_from_json(json_path)
        preview_text = build_preview_text(pages)
        doc_profile = infer_document_profile(pdf.name, preview_text)
        print(f"PDF обработан OpenDataLoader: {pdf.name}")

        for page_number, page_text in pages:
            chunks = splitter.split_text(page_text)

            for i, chunk_text in enumerate(chunks):
                chunk_text = chunk_text.strip()
                if not chunk_text:
                    continue

                documents.append(chunk_text)
                metadatas.append(
                    {
                        "source": pdf.name,
                        "page": page_number,
                        "chunk_index": i,
                        "parser": "opendataloader",
                        **doc_profile,
                    }
                )
                ids.append(make_chunk_id(pdf.name, page_number, i))

    if not documents:
        print("Не найдено документов для индексации.")
        return

    print(f"Подготовлено {len(documents)} чанков для индексации.")
    upload_batches(documents, metadatas, ids)
    print("Готово! Данные проиндексированы через OpenDataLoader.")


if __name__ == "__main__":
    main()
