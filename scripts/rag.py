import os

import numpy as np
from dotenv import load_dotenv
from fastembed import TextEmbedding
from groq import Groq
from qdrant_client import QdrantClient, models

load_dotenv()

DENSE_MODEL = "intfloat/multilingual-e5-small"
SPARSE_MODEL = "Qdrant/bm25"
COLLECTION_NAME = "university_docs_odl"

CANDIDATE_LIMIT = 20
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
client_groq = Groq(api_key=GROQ_API_KEY)

client = QdrantClient(
    url=os.getenv("QDRANT_URL", "http://localhost:6333"),
    api_key=os.getenv("QDRANT_API_KEY") or None,
    timeout=30,
    trust_env=False,
    check_compatibility=False,
)
client.set_model(DENSE_MODEL)
client.set_sparse_model(SPARSE_MODEL)

DENSE_VECTOR_NAME = next(iter(client.get_fastembed_vector_params().keys()))
SPARSE_VECTOR_NAME = next(iter(client.get_fastembed_sparse_vector_params().keys()))

# Reuse the same TextEmbedding instance that fastembed caches internally
_route_embedder = TextEmbedding(model_name=DENSE_MODEL)
_route_embs = np.array([
    next(_route_embedder.embed([r["description"]]))
    for r in ROUTES
])


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))


def get_routes_for_question(question: str) -> list[dict]:
    q_emb = next(_route_embedder.embed([question]))
    sims = sorted(
        [(_cosine_sim(q_emb, r_emb), i) for i, r_emb in enumerate(_route_embs)],
        reverse=True,
    )
    top_sim, top_idx = sims[0]
    second_sim = sims[1][0] if len(sims) > 1 else 0.0
    if top_sim - second_sim >= ROUTER_MIN_GAP:
        return [ROUTES[top_idx]]
    return []


def build_filter(criteria: dict | None) -> models.Filter | None:
    if not criteria:
        return None
    return models.Filter(
        must=[
            models.FieldCondition(key=key, match=models.MatchValue(value=value))
            for key, value in criteria.items()
        ]
    )


def build_hit_key(hit):
    if hit.id is not None:
        return hit.id
    payload = hit.payload or {}
    return (payload.get("source"), payload.get("page"), payload.get("chunk_index"))


def run_hybrid_query(search_text: str, collection_name: str, limit: int, route_filter=None) -> list:
    response = client.query_points(
        collection_name=collection_name,
        prefetch=[
            models.Prefetch(
                query=models.Document(text=search_text, model=DENSE_MODEL),
                using=DENSE_VECTOR_NAME,
                filter=route_filter,
                limit=limit,
            ),
            models.Prefetch(
                query=models.Document(text=search_text, model=SPARSE_MODEL),
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
    routes = get_routes_for_question(question)
    seen: dict = {}

    if routes:
        for route in routes:
            hits = run_hybrid_query(
                search_text=question,
                collection_name=collection_name,
                limit=CANDIDATE_LIMIT,
                route_filter=build_filter(route["filters"]),
            )
            for hit in hits:
                key = build_hit_key(hit)
                if key not in seen or (hit.score or 0) > (seen[key].score or 0):
                    seen[key] = hit

    if not routes or len(seen) < CONTEXT_LIMIT:
        for hit in run_hybrid_query(question, collection_name, CANDIDATE_LIMIT):
            key = build_hit_key(hit)
            if key not in seen:
                seen[key] = hit

    return sorted(seen.values(), key=lambda h: h.score or 0.0, reverse=True)[:CONTEXT_LIMIT]


def ask_question(question: str) -> str:
    if question.strip() == "/start":
        return "Давай же начнем наше общение! Я всегда на связи, спрашивай 💙"

    search_results = hybrid_search(question, COLLECTION_NAME)

    if not search_results:
        return "Извини, у меня нет информации по этому вопросу. Попробуй переформулировать."

    context_parts = []
    for hit in search_results:
        source = hit.payload.get("source", "Неизвестный источник")
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

    return completion.choices[0].message.content


if __name__ == "__main__":
    try:
        print(ask_question("привет"))
    except Exception as e:
        print(f"Произошла ошибка: {e}")
