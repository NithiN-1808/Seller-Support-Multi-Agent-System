"""
main.py
FastAPI REST API exposing the multi-agent seller support system.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import time
from orchestrator import run_pipeline

app = FastAPI(
    title="Seller Support Multi-Agent System",
    description="Multi-agent RAG system for Amazon Seller Central queries using Strands Agents + LangGraph",
    version="1.0.0"
)


class QueryRequest(BaseModel):
    question: str


class QueryResponse(BaseModel):
    question: str
    answer: str
    validation_score: float
    latency_ms: float
    error: str = ""


@app.get("/")
def root():
    return {
        "service": "Seller Support Multi-Agent System",
        "status": "running",
        "agents": ["Retriever (Strands)", "Validator (Strands)"],
        "orchestrator": "LangGraph",
        "llm_backend": "Ollama (Mistral 7B)"
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    start = time.time()
    result = run_pipeline(request.question)
    latency_ms = round((time.time() - start) * 1000, 2)

    return QueryResponse(
        question=result["question"],
        answer=result["answer"],
        validation_score=result["validation_score"],
        latency_ms=latency_ms,
        error=result.get("error", "")
    )


@app.get("/docs-ui")
def docs_redirect():
    return {"message": "Visit /docs for the Swagger UI"}