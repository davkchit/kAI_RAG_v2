import asyncio
import json
import os
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Callable

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

_ask_question: Callable | None = None
_ask_question_stream: Callable | None = None


def _load_rag() -> None:
    global _ask_question, _ask_question_stream
    from scripts.rag import ask_question, ask_question_stream
    _ask_question = ask_question
    _ask_question_stream = ask_question_stream


@asynccontextmanager
async def lifespan(app: FastAPI):
    asyncio.get_event_loop().run_in_executor(None, _load_rag)
    yield


app = FastAPI(title="kAI API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


class HistoryItem(BaseModel):
    role: str   # "user" | "bot"
    text: str


class AskRequest(BaseModel):
    question: str
    history: list[HistoryItem] = []


def _history_to_pairs(items: list[HistoryItem]) -> list[dict]:
    """Convert flat [{role, text}] list to [{q, a}] pairs for rag.py."""
    pairs = []
    flat = [i for i in items if i.role in ("user", "bot")]
    i = 0
    while i < len(flat) - 1:
        if flat[i].role == "user" and flat[i + 1].role == "bot":
            pairs.append({"q": flat[i].text, "a": flat[i + 1].text})
            i += 2
        else:
            i += 1
    return pairs[-3:]  # last 3 turns


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/ask")
async def ask(req: AskRequest):
    if _ask_question is None:
        raise HTTPException(status_code=503, detail="Models loading, try again in a moment")
    history = _history_to_pairs(req.history)
    answer = await asyncio.to_thread(_ask_question, req.question, history)
    return {"answer": answer}


@app.post("/ask/stream")
async def ask_stream(req: AskRequest):
    if _ask_question_stream is None:
        raise HTTPException(status_code=503, detail="Models loading, try again in a moment")

    history = _history_to_pairs(req.history)
    loop = asyncio.get_event_loop()
    queue: asyncio.Queue = asyncio.Queue()

    def run():
        try:
            for token in _ask_question_stream(req.question, history):
                loop.call_soon_threadsafe(queue.put_nowait, token)
        except Exception as e:
            loop.call_soon_threadsafe(queue.put_nowait, {"__err__": str(e)})
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, None)

    threading.Thread(target=run, daemon=True).start()

    async def event_stream():
        while True:
            token = await queue.get()
            if token is None:
                yield "data: [DONE]\n\n"
                break
            if isinstance(token, dict) and "__err__" in token:
                yield f"data: {json.dumps({'token': 'Ошибка при генерации ответа.'}, ensure_ascii=False)}\n\n"
                yield "data: [DONE]\n\n"
                break
            yield f"data: {json.dumps({'token': token}, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# Serve landing page assets
_static_dir = Path(__file__).parent / "static"
if _static_dir.exists():
    app.mount("/images", StaticFiles(directory=str(_static_dir / "images")), name="images")

    @app.get("/style.css")
    async def serve_css():
        from fastapi.responses import FileResponse
        return FileResponse(str(_static_dir / "style.css"), media_type="text/css")

    @app.get("/main.js")
    async def serve_js():
        from fastapi.responses import FileResponse
        return FileResponse(str(_static_dir / "main.js"), media_type="application/javascript")

    @app.get("/")
    async def serve_index():
        from fastapi.responses import FileResponse
        return FileResponse(str(_static_dir / "index.html"), media_type="text/html")


if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
