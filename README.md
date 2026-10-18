# YT Chatbot

A Chrome extension that answers questions using the current YouTube video's transcript, with optional external claim validation. The extension talks to a local FastAPI backend; it is not a standalone website.

## Live Gemini Pipeline

```text
YouTube -> English transcript -> MiniLM + ChromaDB cosine + BM25
                                    |
                           Claim extraction
                                    |
                   Optional web search + semantic reranking
                                    |
                     One batch validation call
                                    |
                         Final Gemini answer
```

- Transcript extraction uses `yt-dlp` and `webvtt`. ChromaDB stores and retrieves relevant transcript chunks in the running backend process.
- Manual and automatic English captions, including regional English variants, are supported. The downloader selects one preferred caption track and skips auto-translations, instead of requesting every English variant. Non-English captions are not used as English evidence. Rolling caption repetitions are removed before indexing. Downloads use isolated temporary files, and blocking retrieval runs off the async server's event loop.
- When validation is required, claims extracted from transcript evidence ground external discovery. Semantic reranking and relevance filtering run before claim validation and final answer generation.
- The normal validated path has **three logical Gemini calls**: claim extraction, one validation call for all claims, and final generation. Retries are additional API attempts within those calls, not extra pipeline stages.
- Transcript-only queries use one final-generation call and skip claim extraction and external validation.
- The validation toggle requests external sources for every question, including summaries. Explicit verification, external context or current-information wording enables research even with the toggle off.
- Structured claim extraction and validation use Pydantic schemas. The default claim limit is three.
- No live pipeline migration to OpenAI has been made.

## Chrome Extension

The popup includes the current video's title and thumbnail, suggested questions, formatted answers, expandable transcript passages, per-claim evidence, and external source links.

- **Enter** in the question field submits; **Shift+Enter** inserts a newline.
- The question button is disabled while a request is running to prevent duplicate submissions.
- The popup uses a fixed 400 x 600px document with internal scrolling and light/dark appearance.
- Frontend libraries are bundled locally in `extension/vendor/`; there are no CDN-loaded scripts. Markdown is sanitized before display.
- The extension sends requests to `http://127.0.0.1:8000/ask` and waits up to three minutes.

### Validation Statuses

| Status | Meaning |
| --- | --- |
| `SUPPORTED` | Retrieved evidence supports the claim. |
| `PARTIALLY_SUPPORTED` | Only part of the claim is supported. |
| `CONTRADICTED` | Retrieved evidence contradicts the claim. |
| `UNVERIFIED` | Evidence is insufficient, or validation could not complete safely. |

An `UNVERIFIED` claim is not a claim proven false. The popup preserves the backend's verdicts and explanations rather than assigning its own statuses.

## Windows Setup

The current environment is tested with **Python 3.12.10**. Use a healthy Python executable that can import `encodings`, not a WindowsApps alias or a broken Anaconda installation.

From the project root, create a virtual environment if needed using the actual installed interpreter path:

```powershell
$Python = 'C:\path\to\healthy\python.exe'
& $Python -c "import sys, encodings; print(sys.executable); print(sys.version)"
& $Python -m venv venv
.\venv\Scripts\python.exe -m pip install --upgrade pip
.\venv\Scripts\python.exe -m pip install -r backend\requirements.txt
```

If the project's healthy `venv` already exists, use it directly. Node.js is needed only for the frontend unit tests, not to run the extension.

### Backend Configuration

Use `.env.example` as the template for **`backend/.env`**. `backend/config.py` loads that location explicitly. Existing process environment variables take precedence.

| Variable | Purpose |
| --- | --- |
| `GEMINI_API_KEY` | Required for live Gemini answers. |
| `GEMINI_MODEL` | Optional model override; use a model available to your Gemini account. |
| `SEARCH_API_KEY` | Serper API key for external search. |
| `YOUTUBE_COOKIES_FILE` | Optional absolute path to a Netscape-format YouTube cookie file. |
| `YOUTUBE_COOKIES_BROWSER` | Optional `yt-dlp` browser-cookie source when no valid cookie file is configured. |
| `MAX_CLAIMS` | Maximum extracted claims; default `3`. |
| `TOP_K_WEB` | Maximum relevant web results returned; default `3`. |
| `WEB_CANDIDATE_COUNT` | Discovery pool before semantic filtering; default `10`, maximum `20`. |
| `TOP_K_WEB_DISPLAY` | Maximum ranked matches shown in the popup; default `3`, minimum `1`. |
| `TOP_K_EVIDENCE` | Evidence selection limit; default `5`. |
| `LLM_TIMEOUT_SECONDS` | Per-attempt Gemini timeout; default `30` seconds. |
| `GEMINI_CALL_TIMEOUT_SECONDS` | Total time budget per logical Gemini call, including queueing and retries; default `40` seconds. |
| `REQUEST_TIMEOUT_SECONDS` | Entire `/ask` time budget; default `150` seconds, shorter than the popup's 180-second timeout. |
| `TRANSCRIPT_TIMEOUT_SECONDS` | Timeout per transcript downloader process; default `20` seconds. |
| `OPENAI_API_KEY` | Used only by the separate offline Batch helper, not the live chatbot. |

Without a Serper key, search uses `ddgs`, already included in `backend/requirements.txt`. `SEARCH_ENGINE_ID` remains in the template but is not used by the current search providers.

Do not commit API keys, `.env` files, cookie exports, or virtual environments. Local Word reports and the removed Markdown documentation/report files are also ignored.

### Start the Backend

Run from the repository root:

```powershell
$env:PYTHONPATH = 'backend'
.\venv\Scripts\python.exe -m uvicorn app:app --host 127.0.0.1 --port 8000 --reload --reload-dir backend
```

The correct entrypoint is **`app:app`**, not `backend.main:app`. The backend needs outbound access to YouTube, Gemini, web search, and evidence pages. Initial ChromaDB embedding setup can require downloading its model.

API documentation: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).

### Load the Extension

1. Open `chrome://extensions/` in Chrome.
2. Enable **Developer mode**.
3. Select **Load unpacked** and choose this repository's `extension` folder.
4. Open a YouTube video, pin the extension, and open its popup.
5. Enter a question and submit it with Enter or **Ask about video**.

After extension updates, reload it in `chrome://extensions/`, close the old popup, and reopen it. Backend source reload does not reload the extension.

## API Contract

`POST /ask` accepts:

```json
{
  "query": "What is a neural network, and what do weights and biases do?",
  "video_id": "aircAruvnKk",
  "validate_externally": true
}
```

Successful responses contain `answer`, `validation_required`, `validation_status`, `video_evidence`, `claims`, and `sources`. Claim results include their status, explanation, supporting evidence, and contradicting evidence.

`sources` contains relevant filtered results sorted by descending `relevance_score` (0–100), with provider order preserved for ties. Each includes a short `relevance_reason` and an available `passage`. `source_display_limit` controls how many matches the popup displays (default three); claim evidence remains available separately.

Relevance combines MiniLM cosine similarity with dense/BM25 reciprocal rank fusion, original provider position, content availability and a small source-type prior. See the semantic retrieval section below for formulas, thresholds and fallbacks. This score is not a credibility or factual-support verdict. Claim-specific evidence overlap retains its separate 0–1 scale.

Missing transcript evidence returns HTTP 422 before Gemini generation. Provider and final-generation failures return non-success HTTP responses, not successful answers containing error text. Recoverable validation failures can leave claims `UNVERIFIED` while still allowing an evidence-contextualized final answer.

## Tests

Run unit tests from the repository root:

```powershell
.\venv\Scripts\python.exe -m pytest tests/ -v
node --test tests/test_popup.cjs
node --check extension/popup.js
```

`pytest.ini` sets the backend import path and limits default discovery to `tests/`. Unit tests use a dummy Gemini key and mocked responses; they do not require or consume your real Gemini key. Regression coverage includes request deadlines, SDK attempt counts, quota handling, caption parsing, concurrent retrieval, evidence selection, and all four validation statuses.

`test_integration_real.py` is separate from the unit suite and exercises multiple real pipeline scenarios. It consumes provider quota and depends on live transcript, network, search, and Gemini availability; run it deliberately, not as part of routine unit checks.

## Retry Behavior and Troubleshooting

- Gemini 429/503 handling is bounded to **four total attempts**, or three retries, per logical call, subject to the call's time budget. Default exponential delays are **4, 8, and 16 seconds**. SDK retries are explicitly disabled to avoid nested retries. A parsed 429 retry hint takes precedence; if the delay exceeds the remaining budget, the error returns without scheduling another attempt.
- Daily quota errors fail fast without retrying. `GeminiMetrics` records actual API attempts, 429/503 retries, and daily-quota fail-fast events across the running process.
- Quota exhaustion during validation stops final generation instead of consuming another Gemini call. Authentication and unavailable-model errors also propagate immediately. Recoverable 503 or validation-timeout failures remain safely `UNVERIFIED`.
- The backend returns an actionable HTTP 504 when its request deadline expires, before the popup's client timeout. Timeout and cancellation do not schedule additional logical calls.
- Source pages are fetched once per URL within a validation request. HTML boilerplate is removed and passages are selected by claim-term overlap, rather than always taking the first 500 characters. Snippets remain a fallback when page access fails; evidence relevance is not a guarantee of factual support.
- Provider unavailability during validation can result in safe `UNVERIFIED` claims; it does not imply that the claims are contradicted.
- For backend connection errors, ensure the server is listening on port 8000 and Chrome has localhost host permissions.
- `WinError 10013` indicates blocked socket access. Check firewall or sandbox restrictions and run the backend in a terminal with outbound network access.
- For missing transcripts, check English automatic captions, YouTube rate limits, and cookie configuration. Browser-cookie extraction can fail because of encryption or a locked browser profile.
- `yt-dlp[default]` includes the YouTube JavaScript challenge solver. A supported JavaScript runtime must also be installed; Deno is detected by default when available on `PATH`.
- Restart the backend after changing API keys or model configuration. Never paste keys into logs, screenshots, or issues.
- The current CORS configuration allows all origins for local development. Restrict it before exposing the backend publicly.

## Separate Offline Batch Support

`backend/rag/openai_batch.py` contains optional offline evaluation helpers for JSONL creation, file upload, batch submission/status polling, output/error download, `custom_id` mapping, and usage metrics. It is not called by the live `/ask` pipeline or Chrome extension.

`test_openai_batch.py` is a manually invoked two-request smoke test. Running it submits a new API batch and may incur charges; `--wait` submits a new batch and then polls that batch, rather than resuming an earlier one. Do not run it as a routine live chatbot test. No OpenAI Batch request is needed to run the Gemini extension.

## Project Layout

```text
backend/
  app.py                  FastAPI /ask endpoint
  config.py               Environment loading and settings
  rag/                    Transcript retrieval, ChromaDB, Gemini, offline Batch
  prompts/                Claim and answer prompts
  routing/                Validation routing
  schemas/                API and evidence models
  search/                 Web search and evidence retrieval
  validation/             Claim extraction and batch validation
extension/
  manifest.json           Chrome Manifest V3 configuration
  popup.html              Extension popup
  popup.js                Request handling and result rendering
  style.css               Popup layout and appearance
  vendor/                 Bundled libraries and upstream licenses
tests/                    Backend and frontend unit tests
```
## Semantic retrieval upgrade

The `/ask` request and response fields are unchanged. Video chunks remain in
`video_evidence`, independent web pages in `sources`. The existing lexical
`search/relevance.py` helpers remain for compatibility; `/ask` now uses
`search/reranker.py` before validating claims.

Previously, Chroma already generated real `all-MiniLM-L6-v2` vectors implicitly,
using the default squared-L2 metric. It reused collections without checking
embedding configuration and returned top chunks without filtering. Web discovery
used Serper when `SEARCH_API_KEY` was set, otherwise DuckDuckGo (`ddgs`). Display
ranking counted shared question words, including generic words such as
"summarize", and retained zero-score pages. Those results also entered validation
before being ranked. Summary requests inherited the frontend's default external
validation toggle. This explains why summarizer products could appear as sources.

Now one reused local ONNX MiniLM encoder explicitly embeds chunks and queries.
Chroma stores normalized 384-dimensional vectors with cosine distance. Transcript
retrieval merges vector and BM25 rankings using weighted reciprocal rank fusion:
`0.75 * 61/(60+dense_rank) + 0.25 * 61/(60+BM25_rank)` (only positive scores
participate). A chunk must pass the cosine threshold or have an exact lexical
match. Duplicate text is suppressed. Generic summaries instead sample chunks
in chronological order across the video, avoiding a topic search for "summary".
Timestamp bounds, video ID, chunk index and source URL are stored in Chroma;
returned chunk IDs now identify the actual stored chunks. Caption timestamps
are approximate bounds when a chunk splits a long caption.

External discovery searches the focused question and, when useful, a second short
query enriched with the most closely related extracted claim. The provider results
are interleaved before deduplication. Specific questions are ranked independently
of other transcript claims; vague verification requests use extracted claims to
resolve their subject. Facebook, Instagram, TikTok and Pinterest results are
excluded from the initial search unless the question requests social sources or names one of those platforms.
Every candidate without an existing passage gets a bounded page fetch, with its
snippet retained if fetching fails. Overlapping 100-word windows from up to 10,000
characters of page text and separate snippet windows supply the maximum content
cosine. Titles contribute to BM25, but cannot supply semantic evidence themselves.
Candidates must pass `WEB_MIN_SIMILARITY` and stay within
`WEB_MAX_SIMILARITY_GAP` of the best content match; title-only candidates
are rejected. Scores are `100 * (0.70*cosine + 0.20*RRF + 0.05/provider_rank +
0.03*source_prior + 0.02*selected_page_passage)`. The RRF weights match transcript
retrieval. The small source prior is 1 for `.gov`/`.edu`, otherwise 0.5; it is
not a credibility verdict. Scores measure retrieval relevance, not factual
support or a probability. Canonical URL deduplication removes fragments and
tracking parameters; content deduplication and a summarizer-product intent
filter remove additional noise. Relevant tool questions can still retrieve tools.
Only selected pages enter claim validation, and their fetched passages are reused.
If the strict result set is empty, discovery retries without domain exclusions
and with simpler topic terms. The combined pool is ranked strictly again before
relaxing the relevance thresholds. If necessary, the closest real results are
returned with unchanged scores and a "low topical match" label. Social or tool
results are used only as a last resort when no ordinary candidates exist; title-only
results explicitly disclose that page content could not be checked. Page fetches,
including failures, are cached across ranking attempts within the request.

Every query with the external-validation toggle enabled runs external discovery,
including summaries, whose searches are grounded in extracted video claims.
Turning the toggle off opts out unless the question explicitly requests external
evidence. Claim-extraction and validation failure behavior is preserved. If
embedding fails, logged BM25 fallbacks rank question-term coverage. Serper failures
or empty results also try DuckDuckGo with a separate bounded timeout. No candidate
sources are fabricated: if both providers return no usable links even after retries,
the answer remains video-grounded and the popup asks the user to retry.

## Setup and migration

From the repository root in PowerShell:

```powershell
.venv\Scripts\python.exe -m pip install -r backend/requirements.txt
Copy-Item .env.example backend/.env  # only if backend/.env does not already exist
# Edit backend/.env: set GEMINI_API_KEY and a GEMINI_MODEL available to your account.
.venv\Scripts\python.exe -m uvicorn app:app --app-dir backend --host 0.0.0.0 --port 8000
```

No dependencies were added. Chroma's existing encoder needs its MiniLM ONNX
weights on first use (downloaded by Chroma to its user cache); subsequent calls
are local. The text embedding LRU holds at most 2,048 entries, batches misses,
and bounds embedding inputs. Blocking inference and Chroma work run in threads.
Use `--log-level info` plus Python logging configuration for `rag.retriever` and
`search.reranker` to see retrieval diagnostics (never API keys).

Collections use the new `yt_semantic_v2_<video_id>` namespace with model,
metric and chunk-version metadata. Legacy `yt_transcript_*` collections are left
intact and transcripts are reindexed on first access. Incompatible v2 metadata
raises an explicit error; choose a new namespace/version and reindex rather than
relabel old vectors. The existing `Client()` lifecycle is preserved: collections
are in memory and restart requires retrieval/indexing again. There is no persistent
database migration or deletion.

`backend/.env` takes precedence over the root `.env`; shell environment variables
take precedence over both. Configuration defaults:

| Setting | Default | Meaning |
| --- | --- | --- |
| `TRANSCRIPT_MIN_SIMILARITY` | `0.25` | Minimum cosine for semantic chunks |
| `WEB_MIN_SIMILARITY` | `0.45` | Minimum cosine for external page content |
| `WEB_MAX_SIMILARITY_GAP` | `0.15` | Maximum cosine gap below the strongest content match |
| `WEB_MIN_LEXICAL_COVERAGE` | `0.5` | External coverage floor on embedding failure |
| `WEB_CANDIDATE_COUNT` | `10` | Candidate pool, 1–20; raised toward TOP_K_WEB within that bound |
| `TOP_K_TRANSCRIPT` / `TOP_K_WEB` | `5` / `3` | Maximum returned chunks/pages |

These cosine thresholds are starting values for English MiniLM, not universally
calibrated cutoffs. Tune with representative topics. Long summaries sample the
transcript and may omit details; increase `TOP_K_TRANSCRIPT` for more coverage.
Unavailable pages fall back to snippets, provider failures return no sources,
and request/provider timeouts remain bounded. Serper needs `SEARCH_API_KEY`;
DuckDuckGo needs network access and may rate-limit. Gemini and YouTube captions
still require the existing credentials/network/subtitle availability.

## Retrieval tests

```powershell
.venv\Scripts\python.exe -m pytest -q
# Optional local MiniLM smoke check (cached weights or first-run download required):
$env:RUN_LOCAL_EMBEDDING_TESTS = '1'
.venv\Scripts\python.exe -m pytest tests/test_semantic_retrieval.py -q
```

The deterministic tests inject known vectors into actual Chroma collections to
test paraphrases, BM25 exact matches, summary coverage, migration checks,
deduplication, tool rejection, relevance floors, and embedding failures without
external calls. The opt-in smoke test additionally checks actual MiniLM inference;
mocked vectors alone do not establish model quality on arbitrary videos.
