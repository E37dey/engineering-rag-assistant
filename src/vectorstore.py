"""All Qdrant interaction lives here.

Nothing else in the codebase should import `qdrant_client` directly — if
we ever swap the DB (Weaviate, pgvector, …) this is the one file to
rewrite.

Collection layout:
    name:     engineering_docs
    distance: cosine  (vectors are L2-normalized at embed time, so cosine
              is equivalent to a dot product and Qdrant can exploit that)
    payload:  filename, page, chunk_index, text

Point IDs are deterministic UUID5 hashes over (filename, page,
chunk_index). Re-running the ingestion pipeline overwrites the same
points instead of creating duplicates — the index stays clean even if
you re-run repeatedly while tuning the chunker.
"""
import uuid

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

COLLECTION_NAME = "engineering_docs"
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333

# Bigger than this and the HTTP body for a single upsert request grows
# uncomfortably (~1.5 KB vector + ~2 KB text payload per point).
UPSERT_BATCH = 128

_client: QdrantClient | None = None


def _get_client() -> QdrantClient:
    """Lazy singleton accessor for the Qdrant client."""
    global _client
    if _client is None:
        # check_compatibility=False silences a client/server version
        # warning that's noisy and not actionable in our setup.
        _client = QdrantClient(
            host=QDRANT_HOST,
            port=QDRANT_PORT,
            check_compatibility=False,
        )
    return _client


def init_collection(vector_size: int) -> None:
    """Create the collection if it doesn't exist (idempotent).

    We deliberately do NOT recreate on re-run — the UUID5 point IDs make
    upsert overwrite-safe, so dropping the collection would only throw
    away the work done so far.
    """
    client = _get_client()
    if client.collection_exists(COLLECTION_NAME):
        return
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
    )


def upsert(chunks: list[dict], embeddings: list[list[float]]) -> None:
    """Store chunks + embeddings + metadata. Lengths must match."""
    if len(chunks) != len(embeddings):
        raise ValueError(
            f"chunks/embeddings length mismatch: {len(chunks)} vs {len(embeddings)}"
        )
    if not chunks:
        return

    client = _get_client()
    points = [
        PointStruct(
            id=_chunk_id(chunk),
            vector=vector,
            payload={
                "filename": chunk["filename"],
                "page": chunk["page"],
                "chunk_index": chunk["chunk_index"],
                "text": chunk["text"],
            },
        )
        for chunk, vector in zip(chunks, embeddings)
    ]

    for start in range(0, len(points), UPSERT_BATCH):
        client.upsert(
            collection_name=COLLECTION_NAME,
            points=points[start : start + UPSERT_BATCH],
        )


def search(query_embedding: list[float], top_k: int = 5) -> list[dict]:
    """Return top_k most similar chunks with their metadata and score."""
    client = _get_client()
    response = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_embedding,
        limit=top_k,
    )
    return [
        {
            "text": hit.payload["text"],
            "filename": hit.payload["filename"],
            "page": hit.payload["page"],
            "chunk_index": hit.payload["chunk_index"],
            "score": hit.score,
        }
        for hit in response.points
    ]


def count() -> int:
    """Return the number of points currently in the collection."""
    client = _get_client()
    return client.count(collection_name=COLLECTION_NAME, exact=True).count


def _chunk_id(chunk: dict) -> str:
    """Deterministic UUID5 ID for a chunk — stable across re-runs."""
    key = f"{chunk['filename']}::{chunk['page']}::{chunk['chunk_index']}"
    return str(uuid.uuid5(uuid.NAMESPACE_URL, key))
