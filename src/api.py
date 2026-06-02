"""FastAPI app exposing the RAG pipeline over HTTP.

Run: uvicorn src.api:app --reload
Docs: http://localhost:8000/docs

The /docs Swagger UI is intentionally a first-class artifact — it's
the recruiter-facing demo of "this is a real API, not just a script."
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.generate import generate_answer

app = FastAPI(
    title="Engineering RAG Assistant",
    description=(
        "Answer engineering questions strictly from indexed datasheets. "
        "Returns the answer with inline citations plus the source chunks "
        "used to ground it. Out-of-corpus questions are refused, not "
        "guessed."
    ),
    version="0.1.0",
)


class QueryRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=1,
        description="Question to ask the corpus.",
        examples=["What is the typical supply voltage of NE555?"],
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of chunks to retrieve as context.",
    )


class Source(BaseModel):
    filename: str
    page: int
    chunk_index: int
    score: float
    text: str


class QueryResponse(BaseModel):
    answer: str
    sources: list[Source]


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    """Answer a question using retrieval + generation."""
    try:
        result = generate_answer(request.question, top_k=request.top_k)
    # Broad catch is intentional at the API boundary: any pipeline error
    # (Qdrant down, Anthropic auth, model load failure) becomes a clean
    # HTTP 500 instead of a leaked stack trace.
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Pipeline error: {type(e).__name__}: {e}",
        )
    return QueryResponse(answer=result["answer"], sources=result["sources"])
