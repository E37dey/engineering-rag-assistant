"""Run the full ingestion pipeline end-to-end:

    PDFs -> chunks -> embeddings -> Qdrant

Run: python scripts/build_index.py

Requires Qdrant running at localhost:6333 (docker compose up).
"""
import sys
import time
from pathlib import Path

# Make project root importable when running this script directly.
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from src.embeddings import EMBEDDING_DIM, embed_batch
from src.ingest import ingest
from src.vectorstore import COLLECTION_NAME, count, init_collection, upsert

DATA_DIR = Path(__file__).parent.parent / "data"


def main() -> int:
    t0 = time.perf_counter()

    print(f"[1/4] Ingesting PDFs from {DATA_DIR}")
    chunks = ingest(DATA_DIR)
    print(f"      -> {len(chunks)} chunks")
    if not chunks:
        print("No chunks produced. Aborting.")
        return 1

    print(f"[2/4] Initializing Qdrant collection '{COLLECTION_NAME}' (dim={EMBEDDING_DIM})")
    init_collection(EMBEDDING_DIM)

    print(f"[3/4] Embedding {len(chunks)} chunks with BAAI/bge-small-en-v1.5")
    texts = [c["text"] for c in chunks]
    vectors = embed_batch(texts)
    print(f"      -> {len(vectors)} vectors x {len(vectors[0])} dim")

    print(f"[4/4] Upserting into Qdrant")
    upsert(chunks, vectors)
    stored = count()
    print(f"      -> {stored} points now in collection")

    elapsed = time.perf_counter() - t0
    print()
    print(f"Pipeline complete in {elapsed:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
