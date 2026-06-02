"""Streamlit UI for the Engineering RAG Assistant.

Run: streamlit run app.py

Talks to the FastAPI backend at API_BASE — start that first
(uvicorn src.api:app --port 8010), then open the Streamlit URL.

Design notes:
  * Custom CSS lives in CUSTOM_CSS at the top of this file. Streamlit's
    defaults are demo-grade; the goal here is product-grade.
  * `[filename.pdf, p.N]` citations in the answer are highlighted as
    inline badges so a reader can see grounding at a glance.
  * Refusal responses render as a distinct amber banner with an
    "OUT OF CORPUS" label — refusal is intentional behaviour, not a
    failure, and the UI should communicate that.
"""
import html
import re
import time

import requests
import streamlit as st

# --- Configuration ---------------------------------------------------------

API_BASE = "http://localhost:8010"
QUERY_URL = f"{API_BASE}/query"
HEALTH_URL = f"{API_BASE}/health"
REQUEST_TIMEOUT = 60

REFUSAL_PHRASE = "i don't have information on this in the provided documents"

# Tolerant citation match: anything enclosed in [ ] that mentions .pdf.
CITATION_PATTERN = re.compile(r"\[[^\]]*\.pdf[^\]]*\]")

SNIPPET_CHARS = 360

# --- Styling ---------------------------------------------------------------

CUSTOM_CSS = """
<style>
  /* Page chrome */
  .stApp {
      background: linear-gradient(180deg, #fafbfc 0%, #f3f4f6 100%);
  }
  .main .block-container {
      max-width: 820px;
      padding-top: 2.75rem;
      padding-bottom: 4rem;
  }
  header[data-testid="stHeader"] { background: transparent; }
  footer { visibility: hidden; }

  /* Typography */
  html, body, [class*="css"] {
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI",
                   Roboto, "Helvetica Neue", Arial, sans-serif;
  }

  /* Hero */
  .hero-title {
      font-size: 2.05rem;
      font-weight: 700;
      color: #0f172a;
      letter-spacing: -0.02em;
      margin-bottom: 0.35rem;
      line-height: 1.2;
  }
  .hero-subtitle {
      font-size: 0.98rem;
      color: #475569;
      line-height: 1.55;
      margin-bottom: 2.25rem;
      max-width: 640px;
  }

  /* Answer card */
  .answer-card {
      background: white;
      border: 1px solid #e2e8f0;
      border-radius: 12px;
      padding: 1.5rem 1.75rem;
      box-shadow: 0 1px 3px rgba(15, 23, 42, 0.04);
      margin-top: 1rem;
  }
  .answer-label {
      display: inline-block;
      font-size: 0.7rem;
      font-weight: 700;
      color: #0e7490;
      text-transform: uppercase;
      letter-spacing: 0.1em;
      background: #ecfeff;
      padding: 0.2rem 0.55rem;
      border-radius: 4px;
      margin-bottom: 0.85rem;
  }
  .answer-card p {
      font-size: 1rem;
      line-height: 1.7;
      color: #1e293b;
      margin: 0 0 0.85rem 0;
  }
  .answer-card p:last-child { margin-bottom: 0; }
  .answer-card strong { color: #0f172a; }

  /* Citation badge */
  .citation {
      display: inline-block;
      background: #ecfeff;
      color: #0e7490;
      border: 1px solid #a5f3fc;
      padding: 0.05rem 0.5rem;
      border-radius: 4px;
      font-size: 0.85em;
      font-weight: 500;
      font-family: ui-monospace, SFMono-Regular, "SF Mono", Consolas, monospace;
      white-space: nowrap;
      margin: 0 0.1rem;
      line-height: 1.4;
  }

  /* Refusal banner */
  .refusal-banner {
      background: #fffbeb;
      border: 1px solid #fcd34d;
      border-left: 4px solid #d97706;
      border-radius: 8px;
      padding: 1.1rem 1.35rem;
      margin-top: 1rem;
      display: flex;
      align-items: flex-start;
      gap: 0.9rem;
  }
  .refusal-badge {
      background: #d97706;
      color: white;
      font-size: 0.65rem;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      padding: 0.25rem 0.55rem;
      border-radius: 4px;
      flex-shrink: 0;
      margin-top: 0.1rem;
      white-space: nowrap;
  }
  .refusal-content {
      flex: 1;
  }
  .refusal-title {
      font-size: 0.95rem;
      font-weight: 600;
      color: #78350f;
      margin-bottom: 0.2rem;
  }
  .refusal-text {
      color: #92400e;
      font-size: 0.9rem;
      line-height: 1.5;
  }

  /* Metrics row */
  .metrics-row {
      display: flex;
      gap: 1.75rem;
      margin: 0.85rem 0 0.25rem;
      padding: 0.85rem 1.1rem;
      background: #f8fafc;
      border: 1px solid #eef2f7;
      border-radius: 8px;
      font-size: 0.85rem;
  }
  .metric { display: flex; flex-direction: column; gap: 0.15rem; }
  .metric-value {
      font-weight: 600;
      color: #0f172a;
      font-size: 1.05rem;
      font-variant-numeric: tabular-nums;
  }
  .metric-label {
      color: #64748b;
      font-size: 0.7rem;
      text-transform: uppercase;
      letter-spacing: 0.06em;
  }

  /* Sources section */
  .sources-heading {
      margin-top: 1.5rem;
      margin-bottom: 0.7rem;
      display: flex;
      align-items: baseline;
      gap: 0.6rem;
  }
  .sources-heading-title {
      font-size: 0.78rem;
      font-weight: 700;
      color: #64748b;
      text-transform: uppercase;
      letter-spacing: 0.1em;
  }
  .sources-heading-count {
      font-size: 0.78rem;
      color: #94a3b8;
  }
  .sources-heading-note {
      font-size: 0.8rem;
      color: #78350f;
      font-style: italic;
      margin-bottom: 0.7rem;
  }

  /* Source card */
  .source-card {
      background: white;
      border: 1px solid #e2e8f0;
      border-radius: 8px;
      padding: 0.95rem 1.2rem;
      margin-bottom: 0.6rem;
      transition: border-color 0.15s ease;
  }
  .source-card:hover {
      border-color: #cbd5e1;
  }
  .source-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 0.6rem;
      margin-bottom: 0.4rem;
      flex-wrap: wrap;
  }
  .source-id {
      font-family: ui-monospace, monospace;
      color: #94a3b8;
      font-size: 0.8rem;
      margin-right: 0.4rem;
  }
  .source-filename {
      font-weight: 600;
      color: #0f172a;
      font-size: 0.92rem;
  }
  .source-page {
      color: #64748b;
      font-size: 0.85rem;
      margin-left: 0.4rem;
  }
  .score-pill {
      display: inline-flex;
      align-items: center;
      gap: 0.4rem;
      background: #f1f5f9;
      border-radius: 999px;
      padding: 0.2rem 0.6rem 0.2rem 0.55rem;
      font-family: ui-monospace, monospace;
      font-size: 0.75rem;
      color: #334155;
  }
  .score-dot { width: 7px; height: 7px; border-radius: 50%; }
  .score-dot.high { background: #10b981; }
  .score-dot.mid  { background: #f59e0b; }
  .score-dot.low  { background: #ef4444; }

  .source-snippet {
      color: #475569;
      font-size: 0.875rem;
      line-height: 1.55;
      border-left: 3px solid #e2e8f0;
      padding-left: 0.85rem;
      margin-top: 0.55rem;
  }

  /* Sidebar status pill */
  .status-pill {
      display: inline-flex;
      align-items: center;
      gap: 0.45rem;
      padding: 0.3rem 0.65rem;
      border-radius: 999px;
      font-size: 0.8rem;
      font-weight: 500;
  }
  .status-pill.up { background: #d1fae5; color: #065f46; }
  .status-pill.down { background: #fee2e2; color: #991b1b; }
  .status-dot { width: 7px; height: 7px; border-radius: 50%; }
  .status-pill.up .status-dot { background: #10b981; }
  .status-pill.down .status-dot { background: #ef4444; }

  /* Form polish */
  div[data-testid="stTextInput"] input {
      border-radius: 8px;
  }
  div[data-testid="stForm"] {
      border: none;
      padding: 0;
  }
  button[kind="primary"] {
      border-radius: 8px;
      font-weight: 600;
  }
</style>
"""


# --- Page setup ------------------------------------------------------------

st.set_page_config(
    page_title="Engineering RAG Assistant",
    layout="centered",
)
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

if "had_first_query" not in st.session_state:
    st.session_state.had_first_query = False


# --- Helpers ---------------------------------------------------------------


@st.cache_data(ttl=10)
def check_api() -> tuple[bool, str]:
    """Return (is_up, message). Cached for 10s to avoid spamming health."""
    try:
        r = requests.get(HEALTH_URL, timeout=2)
        r.raise_for_status()
        return True, "online"
    except requests.exceptions.ConnectionError:
        return False, "unreachable"
    except requests.exceptions.Timeout:
        return False, "timeout"
    except Exception as e:
        return False, type(e).__name__


def is_refusal(answer: str) -> bool:
    return REFUSAL_PHRASE in answer.lower()


def score_tier(score: float) -> str:
    if score >= 0.70:
        return "high"
    if score >= 0.55:
        return "mid"
    return "low"


def answer_to_html(text: str) -> str:
    """Convert plain answer text into safe HTML with citation badges
    and minimal markdown (paragraphs + **bold**)."""
    escaped = html.escape(text)
    escaped = CITATION_PATTERN.sub(
        lambda m: f'<span class="citation">{m.group(0)}</span>',
        escaped,
    )
    escaped = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", escaped)
    paragraphs = [p.strip() for p in escaped.split("\n\n") if p.strip()]
    return "".join(f"<p>{p}</p>" for p in paragraphs)


def render_answer_card(answer: str) -> None:
    body_html = answer_to_html(answer)
    st.markdown(
        f"""
        <div class="answer-card">
            <span class="answer-label">Answer</span>
            {body_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_refusal_banner() -> None:
    st.markdown(
        """
        <div class="refusal-banner">
            <span class="refusal-badge">Out of corpus</span>
            <div class="refusal-content">
                <div class="refusal-title">No information in the indexed documents</div>
                <div class="refusal-text">
                    The system did not find an answer in the source material
                    and refused to synthesize one. This is intentional
                    behaviour — out-of-scope questions are surfaced rather
                    than guessed.
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metrics(elapsed: float, sources: list[dict]) -> None:
    top_score = max((s["score"] for s in sources), default=0.0)
    st.markdown(
        f"""
        <div class="metrics-row">
            <div class="metric">
                <span class="metric-value">{elapsed:.2f}s</span>
                <span class="metric-label">Response time</span>
            </div>
            <div class="metric">
                <span class="metric-value">{len(sources)}</span>
                <span class="metric-label">Chunks retrieved</span>
            </div>
            <div class="metric">
                <span class="metric-value">{top_score:.3f}</span>
                <span class="metric-label">Top similarity</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sources(sources: list[dict], refused: bool) -> None:
    if not sources:
        return

    note_html = ""
    if refused:
        note_html = (
            '<div class="sources-heading-note">'
            "Closest matches retrieved — none contained an answer the model "
            "would commit to."
            "</div>"
        )

    st.markdown(
        f"""
        <div class="sources-heading">
            <span class="sources-heading-title">Sources</span>
            <span class="sources-heading-count">
                {len(sources)} chunk{"s" if len(sources) != 1 else ""}
            </span>
        </div>
        {note_html}
        """,
        unsafe_allow_html=True,
    )

    for i, src in enumerate(sources, start=1):
        snippet = " ".join(src["text"].split())
        if len(snippet) > SNIPPET_CHARS:
            snippet = snippet[:SNIPPET_CHARS] + "…"
        snippet_html = html.escape(snippet)
        tier = score_tier(src["score"])
        st.markdown(
            f"""
            <div class="source-card">
                <div class="source-header">
                    <div>
                        <span class="source-id">[{i}]</span>
                        <span class="source-filename">{html.escape(src['filename'])}</span>
                        <span class="source-page">
                            page {src['page']} &nbsp;·&nbsp;
                            chunk #{src['chunk_index']}
                        </span>
                    </div>
                    <span class="score-pill">
                        <span class="score-dot {tier}"></span>
                        score {src['score']:.3f}
                    </span>
                </div>
                <div class="source-snippet">{snippet_html}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# --- Sidebar ---------------------------------------------------------------

with st.sidebar:
    st.markdown("### Settings")
    top_k = st.slider(
        "Chunks retrieved per question",
        min_value=1,
        max_value=10,
        value=5,
        help="How much context to show Claude. More = better recall, more tokens.",
    )

    st.markdown("---")
    st.markdown("### Backend")
    is_up, status_message = check_api()
    status_class = "up" if is_up else "down"
    st.markdown(
        f"""
        <div class="status-pill {status_class}">
            <span class="status-dot"></span>
            API {status_message}
        </div>
        """,
        unsafe_allow_html=True,
    )
    if not is_up:
        st.caption(
            "Start the backend:\n\n"
            "```\nuvicorn src.api:app --port 8010\n```"
        )

    st.markdown("---")
    st.caption(
        "**Indexed corpus**\n\nTI datasheets: NE555, LM358, MAX232, INA128. "
        "Answers are grounded in source documents with inline citations; "
        "out-of-corpus questions are refused."
    )


# --- Main panel ------------------------------------------------------------

st.markdown(
    '<div class="hero-title">Engineering RAG Assistant</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="hero-subtitle">'
    "Ask questions about engineering datasheets — every answer is grounded "
    "in sources, and the system refuses to answer when the information "
    "isn't there."
    "</div>",
    unsafe_allow_html=True,
)

with st.form("query_form"):
    question = st.text_input(
        "Question",
        placeholder="e.g. How does the MAX232 generate negative voltage from a single +5V supply?",
        label_visibility="collapsed",
    )
    submit = st.form_submit_button("Ask", type="primary")


if submit:
    if not question.strip():
        st.warning("Please enter a question.")
    elif not is_up:
        st.error(
            f"The backend API is unreachable at `{API_BASE}`. "
            "Start it before asking — see the sidebar for the command."
        )
    else:
        spinner_text = "Searching corpus and asking Claude…"
        if not st.session_state.had_first_query:
            spinner_text = (
                "Loading embedding model and querying Claude — "
                "the first request takes 10-15 seconds."
            )

        t0 = time.perf_counter()
        with st.spinner(spinner_text):
            try:
                response = requests.post(
                    QUERY_URL,
                    json={"question": question, "top_k": top_k},
                    timeout=REQUEST_TIMEOUT,
                )
                response.raise_for_status()
                data = response.json()
            except requests.exceptions.Timeout:
                st.error(
                    f"Request timed out after {REQUEST_TIMEOUT}s. "
                    "The first query loads the embedding model — try again."
                )
                st.stop()
            except requests.exceptions.HTTPError:
                try:
                    detail = response.json().get("detail", "")
                except Exception:
                    detail = response.text
                st.error(f"API error ({response.status_code}): {detail}")
                st.stop()
            except requests.exceptions.RequestException as e:
                st.error(f"Network error: {type(e).__name__}: {e}")
                st.stop()
        elapsed = time.perf_counter() - t0
        st.session_state.had_first_query = True

        refused = is_refusal(data["answer"])
        if refused:
            render_refusal_banner()
        else:
            render_answer_card(data["answer"])

        render_metrics(elapsed, data["sources"])
        render_sources(data["sources"], refused=refused)
