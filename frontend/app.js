// Engineering RAG Assistant — frontend logic.
// Served by FastAPI at the same origin as the API, so no CORS dance.

const API_BASE = window.location.origin;
const REFUSAL_PHRASE =
  "i don't have information on this in the provided documents";

// Tolerant citation match: [filename.pdf, p.N] in any case/spacing.
const CITATION_RE = /\[([^,\]]+\.pdf)\s*,\s*p\.?\s*(\d+)\]/gi;

const els = {
  form: document.getElementById('queryForm'),
  input: document.getElementById('questionInput'),
  topK: document.getElementById('topKSelect'),
  askBtn: document.getElementById('askBtn'),
  results: document.getElementById('results'),
  apiStatus: document.getElementById('apiStatus'),
  statusLabel: document.querySelector('.api-status-label'),
};

// ----- API status ----------------------------------------------------------

async function checkApi() {
  try {
    const r = await fetch(`${API_BASE}/health`, {
      signal: AbortSignal.timeout(3000),
    });
    if (!r.ok) throw new Error();
    setApiStatus('online');
  } catch {
    setApiStatus('offline');
  }
}

function setApiStatus(state) {
  els.apiStatus.dataset.state = state;
  els.statusLabel.textContent =
    state === 'online' ? 'API online' : 'API offline';
}

// ----- Form submission -----------------------------------------------------

els.form.addEventListener('submit', async (event) => {
  event.preventDefault();
  const question = els.input.value.trim();
  if (!question) return;
  await runQuery(question, parseInt(els.topK.value, 10));
});

async function runQuery(question, topK) {
  setLoading(true);
  els.results.hidden = false;
  els.results.classList.remove('fade-in');
  els.results.innerHTML = renderSkeleton();
  els.results.scrollIntoView({ behavior: 'smooth', block: 'start' });

  const t0 = performance.now();
  try {
    const r = await fetch(`${API_BASE}/query`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question, top_k: topK }),
    });
    if (!r.ok) {
      let detail = '';
      try {
        detail = (await r.json()).detail || '';
      } catch {
        /* ignore */
      }
      throw new Error(`API ${r.status}: ${detail || r.statusText}`);
    }
    const data = await r.json();
    const elapsed = (performance.now() - t0) / 1000;
    renderResults(data, elapsed);
  } catch (err) {
    renderError(err.message || String(err));
  } finally {
    setLoading(false);
  }
}

function setLoading(isLoading) {
  els.askBtn.disabled = isLoading;
  els.askBtn.classList.toggle('loading', isLoading);
  els.results.setAttribute('aria-busy', String(isLoading));
}

// ----- Rendering -----------------------------------------------------------

function renderSkeleton() {
  return `
    <div class="skeleton-card" role="status" aria-label="Loading answer">
      <div class="skeleton-label"></div>
      <div class="skeleton-line"></div>
      <div class="skeleton-line"></div>
      <div class="skeleton-line"></div>
      <div class="skeleton-line short"></div>
    </div>
  `;
}

function renderResults(data, elapsed) {
  const refused = data.answer.toLowerCase().includes(REFUSAL_PHRASE);
  els.results.innerHTML = `
    ${refused ? renderRefusal() : renderAnswer(data.answer, data.sources)}
    ${renderMetrics(elapsed, data.sources)}
    ${renderSources(data.sources, refused)}
  `;
  // Re-trigger animation by toggling class on next frame.
  requestAnimationFrame(() => els.results.classList.add('fade-in'));
  wireCitations();
}

function renderAnswer(answerText, sources) {
  const escaped = escapeHtml(answerText);
  const withCitations = highlightCitations(escaped, sources);
  const withBold = withCitations.replace(
    /\*\*([^*]+)\*\*/g,
    '<strong>$1</strong>'
  );
  const paragraphs = withBold
    .split(/\n\n+/)
    .filter((p) => p.trim().length > 0)
    .map((p) => `<p>${p.replace(/\n/g, '<br>')}</p>`)
    .join('');

  return `
    <article class="answer-card">
      <div class="card-header">
        <span class="card-label">Answer</span>
      </div>
      <div class="answer-body">${paragraphs}</div>
    </article>
  `;
}

function renderRefusal() {
  return `
    <article class="refusal-card" role="status">
      <div class="refusal-icon" aria-hidden="true">
        <svg width="22" height="22" viewBox="0 0 24 24" fill="none"
             stroke="currentColor" stroke-width="2"
             stroke-linecap="round" stroke-linejoin="round">
          <circle cx="12" cy="12" r="10"></circle>
          <line x1="12" y1="8" x2="12" y2="12"></line>
          <line x1="12" y1="16" x2="12.01" y2="16"></line>
        </svg>
      </div>
      <div class="refusal-content">
        <span class="refusal-badge">Out of corpus</span>
        <h3 class="refusal-title">
          This information isn't in the indexed documents
        </h3>
        <p class="refusal-text">
          The system did not find an answer in the source material and
          refused to synthesize one. This is intentional behaviour —
          out-of-scope questions are surfaced rather than guessed.
        </p>
      </div>
    </article>
  `;
}

function renderMetrics(elapsed, sources) {
  const topScore =
    sources.length > 0 ? Math.max(...sources.map((s) => s.score)) : 0;
  return `
    <div class="metrics-row">
      <div class="metric">
        <span class="metric-value">${elapsed.toFixed(2)}s</span>
        <span class="metric-label">Response time</span>
      </div>
      <div class="metric">
        <span class="metric-value">${sources.length}</span>
        <span class="metric-label">Chunks retrieved</span>
      </div>
      <div class="metric">
        <span class="metric-value">${topScore.toFixed(3)}</span>
        <span class="metric-label">Top similarity</span>
      </div>
    </div>
  `;
}

function renderSources(sources, refused) {
  if (sources.length === 0) return '';
  const noteHtml = refused
    ? `<p class="sources-note">
         Closest matches retrieved — none contained an answer the model
         would commit to.
       </p>`
    : '';
  const cards = sources.map((s, i) => renderSourceCard(s, i)).join('');
  return `
    <section class="sources">
      <div class="sources-header">
        <span class="sources-title">Sources</span>
        <span class="sources-count">
          ${sources.length} chunk${sources.length === 1 ? '' : 's'}
        </span>
      </div>
      ${noteHtml}
      <div class="sources-list">${cards}</div>
    </section>
  `;
}

function renderSourceCard(src, idx) {
  const snippetRaw = src.text.replace(/\s+/g, ' ').trim();
  const truncated =
    snippetRaw.length > 400 ? snippetRaw.slice(0, 400) + '…' : snippetRaw;
  const tier = scoreTier(src.score);
  const pct = Math.max(0, Math.min(100, Math.round(src.score * 100)));
  return `
    <article class="source-card" id="source-${idx}">
      <header class="source-header">
        <div class="source-meta">
          <span class="source-id">[${idx + 1}]</span>
          <span class="source-filename">${escapeHtml(src.filename)}</span>
          <span class="source-sub">
            page ${src.page} · chunk #${src.chunk_index}
          </span>
        </div>
        <div class="score-wrap" title="cosine similarity ${src.score.toFixed(3)}">
          <div class="score-bar" aria-hidden="true">
            <div class="score-fill ${tier}" style="width: ${pct}%"></div>
          </div>
          <span class="score-num">${src.score.toFixed(3)}</span>
        </div>
      </header>
      <p class="source-snippet">${escapeHtml(truncated)}</p>
    </article>
  `;
}

function renderError(message) {
  els.results.innerHTML = `
    <div class="error-card" role="alert">
      <strong>Request failed</strong>
      <p>${escapeHtml(message)}</p>
    </div>
  `;
  requestAnimationFrame(() => els.results.classList.add('fade-in'));
}

// ----- Citation linking ----------------------------------------------------

function highlightCitations(escapedText, sources) {
  return escapedText.replace(CITATION_RE, (match, file, page) => {
    const fileTrimmed = file.trim();
    const pageNum = parseInt(page, 10);
    const idx = sources.findIndex(
      (s) => s.filename === fileTrimmed && s.page === pageNum
    );
    const target = idx >= 0 ? ` data-target="source-${idx}"` : '';
    return `<span class="citation"${target} role="button" tabindex="0">${match}</span>`;
  });
}

function wireCitations() {
  document.querySelectorAll('.citation[data-target]').forEach((el) => {
    el.addEventListener('click', () => scrollToSource(el.dataset.target));
    el.addEventListener('keydown', (event) => {
      if (event.key === 'Enter' || event.key === ' ') {
        event.preventDefault();
        scrollToSource(el.dataset.target);
      }
    });
  });
}

function scrollToSource(id) {
  const target = document.getElementById(id);
  if (!target) return;
  target.scrollIntoView({ behavior: 'smooth', block: 'center' });
  target.classList.add('highlight');
  setTimeout(() => target.classList.remove('highlight'), 1500);
}

// ----- Helpers -------------------------------------------------------------

function escapeHtml(str) {
  const div = document.createElement('div');
  div.textContent = str;
  return div.innerHTML;
}

function scoreTier(score) {
  if (score >= 0.7) return 'high';
  if (score >= 0.55) return 'mid';
  return 'low';
}

// ----- Initial -------------------------------------------------------------

checkApi();
els.input.focus();
