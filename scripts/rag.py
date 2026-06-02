import os
import random

import httpx
import numpy as np
from dotenv import load_dotenv
from fastembed import SparseTextEmbedding
from groq import Groq
from qdrant_client import QdrantClient, models

load_dotenv()

JINA_MODEL = "jina-embeddings-v3"
JINA_DIMS = 1024
DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "sparse"
COLLECTION_NAME = "university_docs_odl"

CANDIDATE_LIMIT = 20
RERANK_TOP_K = 5
CONTEXT_LIMIT = 3
ROUTER_MIN_GAP = 0.03

ROUTES = [
    {
        "name": "admission_bo",
        "filters": {"doc_group": "admission", "program_level": "bo"},
        "description": "поступление бакалавриат специалитет магистратура ЕГЭ баллы направления квоты зачисление внутренние испытания особая квота",
    },
    {
        "name": "admission_asp",
        "filters": {"doc_group": "admission", "program_level": "asp"},
        "description": "аспирантура вступительные испытания научно-педагогических кадров специальная дисциплина",
    },
    {
        "name": "admission_spo",
        "filters": {"doc_group": "admission", "program_level": "spo"},
        "description": "СПО среднее профессиональное образование поступление колледж",
    },
    {
        "name": "regulations",
        "filters": {"doc_group": "regulations"},
        "description": "отчисление восстановление перевод академический отпуск образовательные отношения приостановление прекращение порядок оформления",
    },
    {
        "name": "branch",
        "filters": {"doc_scope": "branch"},
        "description": "НЧФ набережночелнинский филиал контакты директор руководство общежитие КАМАЗ партнеры учебно-методический отдел телефон адрес сайт",
    },
]

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
JINA_API_KEY = os.getenv("JINA_API_KEY")

client_groq = Groq(api_key=GROQ_API_KEY)
client = QdrantClient(
    url=os.getenv("QDRANT_URL", "http://localhost:6333"),
    api_key=os.getenv("QDRANT_API_KEY") or None,
    timeout=30,
    trust_env=False,
    check_compatibility=False,
)

_bm25 = SparseTextEmbedding("Qdrant/bm25")
_route_embs_cache: np.ndarray | None = None


def _jina_headers() -> dict:
    key = os.getenv("JINA_API_KEY") or JINA_API_KEY
    if not key:
        raise RuntimeError("JINA_API_KEY не задан")
    return {"Authorization": f"Bearer {key.strip()}", "Content-Type": "application/json"}


def _embed_dense(text: str) -> list[float]:
    resp = httpx.post(
        "https://api.jina.ai/v1/embeddings",
        headers=_jina_headers(),
        json={"model": JINA_MODEL, "input": [text], "task": "retrieval.query", "dimensions": JINA_DIMS},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["data"][0]["embedding"]


def _embed_dense_batch(texts: list[str]) -> list[list[float]]:
    resp = httpx.post(
        "https://api.jina.ai/v1/embeddings",
        headers=_jina_headers(),
        json={"model": JINA_MODEL, "input": texts, "task": "retrieval.query", "dimensions": JINA_DIMS},
        timeout=60,
    )
    resp.raise_for_status()
    return [item["embedding"] for item in resp.json()["data"]]


def _embed_sparse(text: str):
    return next(_bm25.embed([text]))


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))


def _get_route_embs() -> np.ndarray:
    global _route_embs_cache
    if _route_embs_cache is None:
        descriptions = [r["description"] for r in ROUTES]
        vecs = _embed_dense_batch(descriptions)
        _route_embs_cache = np.array(vecs)
    return _route_embs_cache


def get_routes_for_question(question: str, q_emb: list[float]) -> list[dict]:
    route_embs = _get_route_embs()
    q = np.array(q_emb)
    sims = sorted(
        [(_cosine_sim(q, route_embs[i]), i) for i in range(len(ROUTES))],
        reverse=True,
    )
    top_sim, top_idx = sims[0]
    second_sim = sims[1][0] if len(sims) > 1 else 0.0
    if top_sim - second_sim >= ROUTER_MIN_GAP:
        return [ROUTES[top_idx]]
    return []


def _rerank(question: str, hits: list) -> list:
    texts = [h.payload.get("document") or h.payload.get("text", "") for h in hits]
    resp = httpx.post(
        "https://api.jina.ai/v1/rerank",
        headers=_jina_headers(),
        json={"model": "jina-reranker-v2-base-multilingual", "query": question, "documents": texts, "top_n": RERANK_TOP_K},
        timeout=30,
    )
    resp.raise_for_status()
    results = resp.json()["results"]
    return [hits[r["index"]] for r in results]


def build_filter(criteria: dict | None) -> models.Filter | None:
    if not criteria:
        return None
    return models.Filter(
        must=[models.FieldCondition(key=k, match=models.MatchValue(value=v)) for k, v in criteria.items()]
    )


def build_hit_key(hit):
    if hit.id is not None:
        return hit.id
    p = hit.payload or {}
    return (p.get("source"), p.get("page"), p.get("chunk_index"))


def run_hybrid_query(q_dense: list[float], q_sparse, collection_name: str, limit: int, route_filter=None) -> list:
    response = client.query_points(
        collection_name=collection_name,
        prefetch=[
            models.Prefetch(
                query=q_dense,
                using=DENSE_VECTOR_NAME,
                filter=route_filter,
                limit=limit,
            ),
            models.Prefetch(
                query=models.SparseVector(
                    indices=q_sparse.indices.tolist(),
                    values=q_sparse.values.tolist(),
                ),
                using=SPARSE_VECTOR_NAME,
                filter=route_filter,
                limit=limit,
            ),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        with_payload=True,
        limit=limit,
    )
    return response.points


def hybrid_search(question: str, collection_name: str) -> list:
    # Compute vectors once — reused for both routing and search
    q_dense = _embed_dense(question)
    q_sparse = _embed_sparse(question)

    routes = get_routes_for_question(question, q_dense)
    seen: dict = {}

    if routes:
        for route in routes:
            for hit in run_hybrid_query(q_dense, q_sparse, collection_name, CANDIDATE_LIMIT, build_filter(route["filters"])):
                key = build_hit_key(hit)
                if key not in seen or (hit.score or 0) > (seen[key].score or 0):
                    seen[key] = hit

    if not routes or len(seen) < CONTEXT_LIMIT:
        for hit in run_hybrid_query(q_dense, q_sparse, collection_name, CANDIDATE_LIMIT):
            key = build_hit_key(hit)
            if key not in seen:
                seen[key] = hit

    candidates = sorted(seen.values(), key=lambda h: h.score or 0.0, reverse=True)[:CANDIDATE_LIMIT]
    if not candidates:
        return []
    return _rerank(question, candidates)[:CONTEXT_LIMIT]


_GREETINGS = {"привет", "здравствуй", "здравствуйте", "хай", "добрый день", "добрый вечер", "доброе утро", "салют", "хэй", "hey", "hi", "hello"}

_CLOSINGS = [
    "Пиши, если будут ещё вопросы 💙",
    "Если что-то непонятно — спрашивай 💙",
    "Обращайся, всегда помогу 💙",
]


def _is_greeting(text: str) -> bool:
    t = text.strip().lower().rstrip("!.,?")
    return t in _GREETINGS or any(t.startswith(g) for g in _GREETINGS)


def ask_question(question: str) -> str:
    if question.strip() == "/start":
        return "Давай же начнем наше общение! Я всегда на связи, спрашивай 💙"

    if _is_greeting(question):
        return "Привет! Я твой помощник по НЧФ КНИТУ-КАИ. Чем могу помочь? 💙"

    search_results = hybrid_search(question, COLLECTION_NAME)

    if not search_results:
        return "Извини, у меня нет информации по этому вопросу. Попробуй переформулировать."

    context_parts = []
    for hit in search_results:
        source = hit.payload.get("source", "?")
        page = hit.payload.get("page")
        content = hit.payload.get("document") or hit.payload.get("text", "")
        if not content:
            continue
        if page:
            context_parts.append(f"--- ИСТОЧНИК: {source}, стр. {page} ---\n{content}")
        else:
            context_parts.append(f"--- ИСТОЧНИК: {source} ---\n{content}")

    if not context_parts:
        return "Извини, у меня нет информации по этому вопросу. Попробуй переформулировать."

    context = "\n\n".join(context_parts)
    prompt = f"Контекст из документов:\n{context}\n\nВопрос студента: {question}"

    completion = client_groq.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": """ты — kAI, помощник нчф книту каи.
правила:

- если пользователь пишет /start — ответь: Давай же начнем наше общение! Я всегда на связи, спрашивай 💙
- отвечай только на основе context.
- если ответа в context нет — напиши: "Извини, у меня нет информации по этому вопросу. Попробуй переформулировать."
- не начинай ответы с приветствий и не представляйся.
- не говори "согласно документам", "в тексте указано" и т.п.
- не придумывай фактов.
- если ответ найден, в конце ответа на новой строке укажи только один источник в формате [src:<source>;p:<page>]
- используй только источник, который уже есть в context
- не придумывай источник
- если ответа в context нет, источник не пиши

стиль:
- отвечай коротко, ясно и дружелюбно.

предложение помощи:
- не предлагай помощь в каждом ответе.
- иногда можно добавить короткую фразу с предложением помощи (примерно в 1 из 4 ответов).
- если добавляешь предложение помощи — поставь 💙 в самом конце.
- если предложения помощи нет — 💙 использовать нельзя.
""",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )

    answer = completion.choices[0].message.content
    if random.random() < 0.10:
        answer = answer.rstrip() + "\n\n" + random.choice(_CLOSINGS)
    return answer


if __name__ == "__main__":
    try:
        print(ask_question("куда поступить в КАИ из Казани"))
    except Exception as e:
        print(f"Ошибка: {e}")
