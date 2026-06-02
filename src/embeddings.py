"""Wrapper around the embedding model.

Backend: local `sentence-transformers` with `BAAI/bge-small-en-v1.5`.
Why local: no external API dependency, zero cost per query, fully offline
inference (matters when the corpus is confidential engineering docs).

Trade-off: bge-small produces 384-dim vectors and is English-only; a
hosted model like voyage-3 would give higher absolute recall and
multilingual support. For this project the autonomy wins.

The model is loaded lazily on first use and held as a module-level
singleton — a single `SentenceTransformer` instance is ~130 MB and
takes 1-2s to initialize, so re-instantiating per call would dominate
query latency.
"""
from sentence_transformers import SentenceTransformer

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIM = 384

# bge models recommend this prefix on queries only (not on documents).
# Keeps query/document representations aligned for retrieval.
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    """Lazy singleton accessor for the embedding model."""
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def embed_query(text: str) -> list[float]:
    """Embed a single query string (used at retrieval time)."""
    model = _get_model()
    vector = model.encode(QUERY_PREFIX + text, normalize_embeddings=True)
    return vector.tolist()


def embed_batch(
    texts: list[str],
    batch_size: int = 32,
    show_progress: bool = True,
) -> list[list[float]]:
    """Embed a batch of documents (used at ingestion time).

    No query prefix here — these are documents, not queries.
    Normalization is on so cosine similarity in Qdrant reduces to a
    simple dot product.
    """
    if not texts:
        return []
    model = _get_model()
    vectors = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=show_progress,
    )
    return vectors.tolist()
