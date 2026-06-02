"""Split documents into overlapping chunks for embedding.

Strategy: sentence-aware greedy packing. We split text into sentences,
then pack them into chunks up to a token budget. Each new chunk seeds
itself with the tail sentences of the previous chunk so cross-chunk
context is preserved (an idea that spans a paragraph boundary still
gets indexed in both neighbors).

Token counts are *estimated* from word counts using the common English
heuristic of ~1.3 tokens per word. The estimate is sufficient for
chunking decisions — being off by 10-20% on a 500-token budget is
inconsequential. In Task 2, when the bge tokenizer is loaded for
embedding, we can swap `_estimate_tokens` for an exact count without
touching the public API.

Defaults — chunk_size=500, overlap=50 — chosen for:
  * Enough context per chunk to embed a coherent paragraph of
    engineering prose.
  * Comfortably under bge-small-en-v1.5's 512-token max sequence
    length, so chunks are not silently truncated by the embedding model.
  * ~10% overlap preserves cross-boundary context without bloating
    the index with near-duplicates.
"""
import re


_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")


def chunk_text(
    text: str,
    chunk_size: int = 500,
    overlap: int = 50,
) -> list[str]:
    """Split a single text into overlapping ~chunk_size-token chunks.

    Sentences are kept whole when possible. A single sentence longer
    than chunk_size is emitted as its own (oversized) chunk rather than
    cut mid-sentence — rare in prose, possible in datasheets with
    inline table dumps.
    """
    sentences = _split_sentences(text)
    if not sentences:
        return []

    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0

    for sentence in sentences:
        sent_tokens = _estimate_tokens(sentence)
        if current and current_tokens + sent_tokens > chunk_size:
            chunks.append(" ".join(current))
            current, current_tokens = _take_overlap_tail(current, overlap)
        current.append(sentence)
        current_tokens += sent_tokens

    if current:
        chunks.append(" ".join(current))

    return chunks


def chunk_documents(
    documents: list[dict],
    chunk_size: int = 500,
    overlap: int = 50,
) -> list[dict]:
    """Chunk page-level documents, propagating filename + page metadata.

    Args:
        documents: Per-page records shaped as
            {"filename": str, "page": int, "text": str}.
        chunk_size: Target tokens per chunk.
        overlap: Target overlap tokens between adjacent chunks.

    Returns:
        Per-chunk records shaped as
        {"filename": str, "page": int, "chunk_index": int, "text": str}.
        chunk_index is per-page; it resets to 0 for each new page.
    """
    result: list[dict] = []
    for doc in documents:
        pieces = chunk_text(doc["text"], chunk_size=chunk_size, overlap=overlap)
        for i, piece in enumerate(pieces):
            result.append({
                "filename": doc["filename"],
                "page": doc["page"],
                "chunk_index": i,
                "text": piece,
            })
    return result


def _split_sentences(text: str) -> list[str]:
    """Split on terminal punctuation followed by whitespace + capital/digit/quote.

    Imperfect — abbreviations like 'e.g.' or 'Dr.' may produce extra
    splits — but the greedy packer downstream tolerates over-splitting
    cleanly (small sentences just pack together).
    """
    parts = _SENTENCE_BOUNDARY.split(text.strip())
    return [p.strip() for p in parts if p.strip()]


def _estimate_tokens(text: str) -> int:
    """Approximate token count from word count (English heuristic)."""
    word_count = len(text.split())
    return int(word_count * 1.3) if word_count else 0


def _take_overlap_tail(
    sentences: list[str],
    target_tokens: int,
) -> tuple[list[str], int]:
    """Return the trailing sentences summing to at most ~target_tokens.

    Always includes at least the last sentence (even if it exceeds the
    target) so overlap is never empty — a zero-overlap seed would lose
    cross-chunk context entirely.
    """
    if target_tokens <= 0:
        return [], 0
    tail: list[str] = []
    tail_tokens = 0
    for sentence in reversed(sentences):
        sent_tokens = _estimate_tokens(sentence)
        if not tail or tail_tokens + sent_tokens <= target_tokens:
            tail.insert(0, sentence)
            tail_tokens += sent_tokens
        else:
            break
    return tail, tail_tokens
