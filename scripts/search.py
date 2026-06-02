"""Manual semantic search against the Qdrant index.

Usage:
    python scripts/search.py "your question here"
    python scripts/search.py --top-k 3 "your question here"

Requires the index to be built — run scripts/build_index.py first.
"""
import argparse
import sys
from pathlib import Path

# Make project root importable when running this script directly.
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from src.embeddings import embed_query
from src.vectorstore import search

SNIPPET_CHARS = 240


def main() -> int:
    parser = argparse.ArgumentParser(description="Semantic search over the RAG index.")
    parser.add_argument("query", help="Question to ask (quote it)")
    parser.add_argument("--top-k", type=int, default=5, help="Number of results (default: 5)")
    args = parser.parse_args()

    vector = embed_query(args.query)
    results = search(vector, top_k=args.top_k)

    print(f'Query: "{args.query}"')
    print(f"Top {len(results)} results:")
    print()

    for rank, hit in enumerate(results, start=1):
        snippet = " ".join(hit["text"].split())  # collapse whitespace
        if len(snippet) > SNIPPET_CHARS:
            snippet = snippet[:SNIPPET_CHARS] + "..."
        print(
            f"[{rank}] {hit['filename']}  page {hit['page']}  "
            f"chunk #{hit['chunk_index']}  score={hit['score']:.4f}"
        )
        print(f"    {snippet}")
        print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
