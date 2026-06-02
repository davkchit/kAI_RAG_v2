import asyncio
import os
from contextlib import asynccontextmanager
from typing import Callable

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

_ask_question: Callable | None = None


def _load_rag() -> None:
    global _ask_question
    from scripts.rag import ask_question
    _ask_question = ask_question


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load heavy models in background thread — /health responds immediately
    asyncio.get_event_loop().run_in_executor(None, _load_rag)
    yield


app = FastAPI(title="kAI API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


class AskRequest(BaseModel):
    question: str


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/ask")
async def ask(req: AskRequest):
    if _ask_question is None:
        raise HTTPException(status_code=503, detail="Models loading, try again in a moment")
    answer = await asyncio.to_thread(_ask_question, req.question)
    return {"answer": answer}


if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
