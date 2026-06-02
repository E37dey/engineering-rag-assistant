"""Semantic search over the indexed corpus.

Thin orchestration layer — embed the query, hit the vector store,
return chunks. Kept separate so higher layers (api.py, generate.py)
depend on `retrieve`, not on embeddings or Qdrant directly.
"""
from src.embeddings import embed_query
from src.vectorstore import search


def retrieve(query: str, top_k: int = 5) -> list[dict]:
    """Embed the query and return top_k relevant chunks with metadata.

    Each result is shaped as
    {"text", "filename", "page", "chunk_index", "score"}.
    """
    vector = embed_query(query)
    return search(vector, top_k=top_k)
