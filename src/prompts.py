"""Prompt templates.

Isolated on purpose. The prompt is the highest-leverage piece of a RAG
system and the most-iterated artifact — anyone reviewing this repo
should be able to open this single file and see exactly how the model
is instructed.

Design notes for the interview:
  * The anti-hallucination guarantee lives in three layers — the system
    prompt's rules, the user prompt's source-block headers (which the
    model can copy verbatim into citations), and the explicit
    "say I don't know" fallback. Defense in depth, not a single switch.
  * No general world knowledge is permitted. Datasheet values change
    between revisions; a confidently-recalled "the LM358's CMRR is 85 dB"
    that doesn't appear in the source we cite would be a regression, not
    a feature.
"""

SYSTEM_PROMPT = """\
You are an engineering documentation assistant. You answer questions \
strictly from the provided source excerpts below.

RULES (no exceptions):

1. Use ONLY the information present in the SOURCE EXCERPTS. Do not draw \
on general knowledge, training data, or assumptions about how components \
typically behave — even if you are confident about a fact.

2. If the sources do not contain enough information to answer the \
question, respond EXACTLY with this sentence and nothing else:
   "I don't have information on this in the provided documents."
   Do not guess, do not infer beyond what the text supports, do not \
apologize at length.

3. Cite the source for every factual claim using the inline format \
[filename, p.PAGE] — copy the filename and page number directly from \
the source header. If multiple sources support a claim, cite each one.

4. Do not invent specifications, voltages, currents, part numbers, \
equation constants, or standards numbers. If an exact value is not in \
the sources, say the value is not specified in the provided documents.

5. When precision matters (supply ranges, equations, electrical \
characteristics), quote the source verbatim rather than paraphrasing.

Write a concise prose answer with inline citations. Do not append a \
separate "Sources" section — that is rendered by the caller.
"""


def build_user_prompt(question: str, chunks: list[dict]) -> str:
    """Build the user-facing prompt with retrieved context injected.

    Each chunk gets a header line that doubles as a citation key:
    `--- [filename, p.PAGE] ---`. The model is instructed (in the system
    prompt) to copy that exact bracketed key into its answer when citing
    the chunk, so citation accuracy doesn't depend on filename inference.
    """
    if not chunks:
        return (
            "SOURCE EXCERPTS:\n\n"
            "(No sources were retrieved.)\n\n"
            f"QUESTION: {question}"
        )

    parts = ["SOURCE EXCERPTS:", ""]
    for chunk in chunks:
        header = f"--- [{chunk['filename']}, p.{chunk['page']}] ---"
        parts.append(header)
        parts.append(chunk["text"])
        parts.append("")
    parts.append(f"QUESTION: {question}")
    return "\n".join(parts)
