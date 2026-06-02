"""Call Claude with retrieved context and return an answer + sources.

Orchestrates the full query-time path:
    question -> retrieve top-k chunks -> build prompt -> Claude -> answer

Design notes:
  * `temperature=0.0` — RAG is a factual-retrieval task; we want maximum
    determinism, not creativity. Any temperature > 0 raises the chance
    of the model wandering off the source text.
  * The system prompt is sent with `cache_control: ephemeral` so
    Anthropic caches it across requests. The SYSTEM_PROMPT is the same
    on every call and is large enough (~1k input tokens) to make caching
    a real win at scale.
"""
import os
from pathlib import Path

from anthropic import Anthropic
from dotenv import load_dotenv

from src.prompts import SYSTEM_PROMPT, build_user_prompt
from src.retrieval import retrieve

# Load .env from the project root explicitly — independent of cwd.
load_dotenv(Path(__file__).parent.parent / ".env")

CLAUDE_MODEL = "claude-sonnet-4-6"
MAX_TOKENS = 1024
TEMPERATURE = 0.0

_client: Anthropic | None = None


def _get_client() -> Anthropic:
    """Lazy singleton accessor for the Anthropic client."""
    global _client
    if _client is None:
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set. Add it to .env "
                "and re-run."
            )
        _client = Anthropic(api_key=api_key)
    return _client


def generate_answer(question: str, top_k: int = 5) -> dict:
    """Retrieve context, ask Claude, return the answer with source chunks.

    Returns:
        {
            "answer": str,                 # Claude's response text
            "sources": list[dict],         # the chunks shown to the model,
                                           # ordered by retrieval score desc
        }
    """
    chunks = retrieve(question, top_k=top_k)
    user_prompt = build_user_prompt(question, chunks)

    response = _get_client().messages.create(
        model=CLAUDE_MODEL,
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
        system=[
            {
                "type": "text",
                "text": SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        messages=[{"role": "user", "content": user_prompt}],
    )

    answer = response.content[0].text
    return {"answer": answer, "sources": chunks}
