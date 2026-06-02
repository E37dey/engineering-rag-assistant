"""Streamlit UI for the Engineering RAG Assistant.

Run: streamlit run app.py

Talks to the FastAPI backend at http://localhost:8000 — start that
first (uvicorn src.api:app --reload), then open the Streamlit URL.
"""
import requests
import streamlit as st

API_BASE = "http://localhost:8010"
QUERY_URL = f"{API_BASE}/query"
HEALTH_URL = f"{API_BASE}/health"
REQUEST_TIMEOUT = 60  # seconds — embedding + Claude can take 5-15s

# --- Page setup -------------------------------------------------------------

st.set_page_config(
    page_title="Engineering RAG Assistant",
    layout="centered",
)


@st.cache_data(ttl=10)
def check_api() -> tuple[bool, str]:
    """Return (is_up, message). Cached for 10s to avoid spamming health."""
    try:
        r = requests.get(HEALTH_URL, timeout=2)
        r.raise_for_status()
        return True, "ok"
    except requests.exceptions.ConnectionError:
        return False, "connection refused"
    except requests.exceptions.Timeout:
        return False, "timed out"
    except Exception as e:
        return False, f"{type(e).__name__}"


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
    st.markdown("### Backend status")
    is_up, message = check_api()
    if is_up:
        st.markdown(f":green[API: {message}]")
    else:
        st.markdown(f":red[API: {message}]")
        st.caption(
            "Start the backend with:\n\n"
            "```\nuvicorn src.api:app --reload\n```"
        )

    st.markdown("---")
    st.caption(
        "Indexed corpus: TI datasheets for NE555, LM358, MAX232, INA128. "
        "Answers are grounded in the source documents with inline "
        "citations. Out-of-corpus questions are refused."
    )


# --- Main panel -----------------------------------------------------------

st.title("Engineering RAG Assistant")
st.caption(
    "Ask a question about the indexed engineering datasheets. "
    "Every claim in the answer is cited; if the answer is not in the "
    "documents, the system says so explicitly."
)

with st.form("query_form"):
    question = st.text_input(
        "Question",
        placeholder="e.g. What is the typical supply voltage of NE555?",
    )
    submit = st.form_submit_button("Ask", type="primary")


def render_answer(answer: str, sources: list[dict]) -> None:
    st.markdown("### Answer")
    st.markdown(answer)

    with st.expander(f"Sources ({len(sources)} chunks)", expanded=False):
        for i, src in enumerate(sources, start=1):
            st.markdown(
                f"**[{i}] {src['filename']}** — page {src['page']} "
                f"&nbsp;·&nbsp; chunk #{src['chunk_index']} "
                f"&nbsp;·&nbsp; score {src['score']:.3f}"
            )
            snippet = " ".join(src["text"].split())
            if len(snippet) > 400:
                snippet = snippet[:400] + "…"
            st.markdown(f"> {snippet}")
            if i < len(sources):
                st.markdown("---")


if submit:
    if not question.strip():
        st.warning("Please enter a question.")
    elif not is_up:
        st.error(
            "The backend API is not reachable at "
            f"`{API_BASE}`. Start it before asking a question."
        )
    else:
        with st.spinner("Searching the corpus and asking Claude…"):
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
                    "The first query is slow because the embedding model "
                    "loads on demand — try once more."
                )
                st.stop()
            except requests.exceptions.HTTPError as e:
                detail = ""
                try:
                    detail = response.json().get("detail", "")
                except Exception:
                    detail = response.text
                st.error(f"API returned {response.status_code}: {detail or e}")
                st.stop()
            except requests.exceptions.RequestException as e:
                st.error(f"Network error: {type(e).__name__}: {e}")
                st.stop()

        render_answer(data["answer"], data["sources"])
