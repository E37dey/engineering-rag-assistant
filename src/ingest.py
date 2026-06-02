"""Ingestion pipeline: load PDFs and split them into chunks with metadata.

Runs offline. Task 1 covers loading + chunking only; embedding + Qdrant
storage land in Task 2 and will hook in at the end of `ingest()`.
"""
from pathlib import Path

from pypdf import PdfReader

from src.chunking import chunk_documents


def load_pdfs(data_dir: Path) -> list[dict]:
    """Read every PDF in `data_dir` and return per-page records.

    Each record is shaped as {"filename": str, "page": int, "text": str}.
    Pages with no extractable text (e.g. scanned images without OCR) are
    skipped silently — they would produce useless embeddings.

    Pages are 1-indexed to match how a human cites a document.
    """
    records: list[dict] = []
    pdf_paths = sorted(data_dir.glob("*.pdf"))

    for pdf_path in pdf_paths:
        reader = PdfReader(str(pdf_path))
        for page_num, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()
            if not text:
                continue
            records.append({
                "filename": pdf_path.name,
                "page": page_num,
                "text": text,
            })

    return records


def ingest(
    data_dir: Path,
    chunk_size: int = 500,
    overlap: int = 50,
) -> list[dict]:
    """Load PDFs from `data_dir`, split each page into chunks, return chunks.

    Returns a list of {"filename", "page", "chunk_index", "text"} records.
    Task 2 will extend this to embed + upsert into Qdrant.
    """
    pages = load_pdfs(data_dir)
    return chunk_documents(pages, chunk_size=chunk_size, overlap=overlap)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Load PDFs from a directory, chunk them, and report stats.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Directory containing PDF files (default: ./data)",
    )
    args = parser.parse_args()

    if not args.data_dir.exists():
        print(f"Data directory not found: {args.data_dir.resolve()}")
        raise SystemExit(1)

    chunks = ingest(args.data_dir)

    print(f"Generated {len(chunks)} chunks from {args.data_dir.resolve()}")
    if not chunks:
        print("(No PDFs found, or all pages were empty.)")
        raise SystemExit(0)

    print()
    print(f"{'idx':>4}  {'file':<36}  {'page':>4}  {'chunk#':>6}  {'chars':>6}")
    print("-" * 68)
    for i, chunk in enumerate(chunks):
        fname = chunk["filename"]
        if len(fname) > 36:
            fname = fname[:33] + "..."
        print(
            f"{i:>4}  {fname:<36}  {chunk['page']:>4}  "
            f"{chunk['chunk_index']:>6}  {len(chunk['text']):>6}"
        )

    char_lengths = [len(c["text"]) for c in chunks]
    print()
    print(
        f"Char length — min: {min(char_lengths)}, "
        f"max: {max(char_lengths)}, "
        f"avg: {sum(char_lengths) // len(char_lengths)}"
    )
