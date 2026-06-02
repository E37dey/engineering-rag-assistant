"""Ask a question against the engineering RAG corpus.

Usage:
    python scripts/ask.py "your question"
    python scripts/ask.py --top-k 5 "your question"

Requires:
  * Qdrant running (docker compose up) with the index built
    (scripts/build_index.py)
  * ANTHROPIC_API_KEY in .env
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

from src.generate import generate_answer

SNIPPET_CHARS = 140


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Ask a question against the engineering RAG corpus.",
    )
    parser.add_argument("question", help="Question to ask (quote it)")
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of chunks to retrieve as context (default: 5)",
    )
    args = parser.parse_args()

    result = generate_answer(args.question, top_k=args.top_k)

    print(f"Q: {args.question}")
    print()
    print("=" * 72)
    print("Answer:")
    print("=" * 72)
    print(result["answer"])
    print()
    print("=" * 72)
    print(f"Sources ({len(result['sources'])} chunks used as context):")
    print("=" * 72)
    for i, src in enumerate(result["sources"], start=1):
        snippet = " ".join(src["text"].split())
        if len(snippet) > SNIPPET_CHARS:
            snippet = snippet[:SNIPPET_CHARS] + "..."
        print(
            f"  [{i}] {src['filename']}, p.{src['page']}  "
            f"(chunk #{src['chunk_index']}, score={src['score']:.3f})"
        )
        print(f"      {snippet}")
        print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
