import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, HTTPException

from query_data import init_rag_system
from rag_service import answer_question, LLMProviderError
from schemas import AskRequest, RagResult

logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    vector_db, llm, prompt_template = init_rag_system()

    app.state.vector_db = vector_db
    app.state.llm = llm
    app.state.prompt_template = prompt_template

    yield

app = FastAPI(title="Banking RAG Assistant", version="0.1.0", lifespan=lifespan)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/ask", response_model=RagResult)
def ask(payload: AskRequest, request: Request):
    vector_db = request.app.state.vector_db
    llm = request.app.state.llm
    prompt_template = request.app.state.prompt_template

    try:
        result = answer_question(
            question=payload.question,
            vector_db=vector_db,
            llm=llm,
            prompt_template=prompt_template,
            k=3
        )
    except LLMProviderError:
        logger.exception("LLM provider request failed")
        raise HTTPException(
            status_code=502,
            detail="Сервис генерации временно недоступен"
        ) from None

    return result
            

