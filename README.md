# Indian Filings RAG

Search an Indian listed company's annual report, ask questions about a selected financial year, and inspect the evidence behind each answer.

The project combines a **Streamlit dashboard**, **FastAPI backend**, local PDF processing and hybrid retrieval, with **Gemini 3.8 Flash** for answers/follow-ups/optional vision and **Gemini Embedding 001** for semantic search. It is a junior-level working prototype with a small retrieval benchmark.

## Start the app

Run commands from the repository root. Python 3.12+ and [uv](https://docs.astral.sh/uv/) are required.

1. Install the pinned dependencies:

   ```bash
   uv sync --python 3.12
   ```

2. If you do not already have `.env`, create it from [.env.example](.env.example). Set your Gemini API key locally. Keep these endpoint/model settings:

   ```dotenv
   GEMINI_API_KEY=your-key
   LLM_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai
   LLM_MODEL=gemini-3.8-flash
   EMBEDDING_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai
   EMBEDDING_MODEL=gemini-embedding-001
   ```

   Preserve an existing `.env`; do not commit credentials. No model key is entered in the browser. Explicit `LLM_API_KEY`/`EMBEDDING_API_KEY` settings and compatible endpoints can be configured separately; see `Provider.__init__` for credential precedence. Google-scoped key selection is restricted to Google's endpoint.

3. Start the dashboard and API together:

   ```bash
   uv run python scripts/run_dashboard.py
   ```

   - Dashboard: [http://127.0.0.1:8501](http://127.0.0.1:8501)
   - Interactive API docs: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

No preloaded corpus is required. Both services bind to `127.0.0.1`; **Ctrl+C** in the launcher terminal stops its child processes. Restart the backend after changing model settings in `.env`.

## Use the dashboard

**Type company name/symbol → choose an exchange-available year → load report → ask questions.**

The company/year controls are on the main page, above the chat, and remain visible when the settings sidebar is collapsed.

- **Ask filings:** type into the searchable Company field, select the issuer and a year, then click **Load report & start Q&A**. An already prepared filing offers **Start Q&A session** instead. Follow-up questions can refer to the conversation. Search options control BM25/dense/hybrid retrieval, passage count and answer generation.
- **Browse documents:** inspect the active filing's extracted text or locally rendered PDF pages. Before a session starts, browse all prepared filings. PDF page numbers include cover letters and can differ from printed page labels.
- **Benchmark results:** inspect saved evaluation charts/tables. Viewing a report does not rerun evaluation or invoke a model. A corpus-hash mismatch marks a saved run as stale.

Changing a company/year choice does not silently replace an active conversation: loading/starting the next filing activates its scope and clears prior messages/page previews. Each answer request stays filtered to its active company/year. Clearing the conversation keeps that scope.

The directory exposed 3,195 symbols at the recorded live check; the count changes with NSE listings. Discovery covers **NSE main-board and SME companies**. It does not promise every BSE-only, foreign, private or delisted issuer. BSE attachments can still be supplied manually through the CLI manifest.

## What happens when a report is missing locally?

1. The backend resolves company identity from official NSE CSV directories.
2. It queries available annual-report year labels and validates the attachment host.
3. A background job downloads the selected PDF or ZIP into a resumable workspace.
4. Local extraction creates page records, text/table blocks and chunks.
5. Missing embeddings are requested; existing endpoint/model/text vectors are reused.
6. A complete merged corpus/index is staged, then published under the service lock.
7. The dashboard's two-second status poll opens Q&A after the job becomes ready.

Jobs move through `queued`, `running`, `ready` or `failed`; stage messages are descriptive, not exact percentages. First-time loading can take a few minutes. One preparation worker runs at a time. Existing published reports remain searchable while a snapshot is built.

Failed download/extraction/embedding does not publish a partial new corpus. Commit failures attempt to restore previous published records/index. Downloaded/raw/cache files can remain for retry. After backend restart, unfinished jobs are marked retryable; the user must load the report again. This is a local job manager, not a distributed queue: avoid running a separate CLI corpus mutation concurrently with preparation.

Company metadata is cached for one day and report metadata for six hours. Successful stale metadata can be reused during exchange outages. NSE/BSE access-denied responses are not bypassed. An available year in metadata does not guarantee that its historical attachment is still downloadable.

ZIP handling retains the original archive and reads one identifiable annual-report PDF without extracting member paths to disk. Ambiguous archives and PDF members over 200 MB are rejected.

## Algorithms and local/API responsibilities

```text
Streamlit → FastAPI → NSE company/year discovery
                         ↓
                 download PDF / ZIP
                         ↓
              local extraction → chunks
                         ↓
          cached document embeddings → local vector matrix

Question → optional follow-up rewrite → company/year filters
                         ↓
            BM25 + dense cosine search → rank fusion
                         ↓
                  selected evidence
                         ↓
                 Gemini cited answer
```

| Component | Implementation | Runs where? |
| --- | --- | --- |
| Company/year discovery and downloads | HTTPX against exchange data; BeautifulSoup for optional saved-HTML link discovery | Local backend + exchange HTTP requests |
| Normal PDF extraction | pdfplumber, conservative column-gutter detection and detected tables | Local |
| Chunking | Text up to 2,200 characters with up to two overlapping lines; table chunks up to 3,500 characters | Local |
| Keyword retrieval | BM25, `k1=1.5`, `b=0.75`; postings built in memory | Local |
| Document/query embeddings | `gemini-embedding-001`; current live index uses 3,072 dimensions | Google API for uncached inputs |
| Embedding cache / vector search | SQLite cache and normalized NumPy matrix; exact cosine similarity | Local |
| Hybrid retrieval | Reciprocal rank fusion of normally 50 candidates per ranking, constant 60 | Local |
| Follow-up reference resolution | `gemini-3.8-flash` rewrites the question within the selected filing scope | Google API after the first generated-answer turn |
| Answer generation | `gemini-3.8-flash` receives the question and selected cited evidence | Google API |
| Optional difficult-page transcription | Explicitly send one locally rendered page image to the model | Google API |
| Benchmark scoring and page-image browsing | Source-quote matching/metrics; local PDF rendering | Local |

BM25 normalizes numeric grouping commas so `1,62,990` and `162990` can match. Hybrid fusion adds `1 / (60 + rank)` contributions; it does not add incompatible raw scores. Method scores are relevance rankings, not confidence probabilities. There is no reranker or hosted vector database.

Raw PDFs stay local in this workflow. Embedding requests send chunk/query text; answer requests send the question and selected passages. Follow-ups send at most six recent user/answer messages, each bounded to 4,000 characters, to resolve references. Conversation is not trusted financial evidence: each answer retrieves fresh passages and starts a fresh citation numbering scheme. Passage-only search does not use history. Memory lasts for the Streamlit session, not a persistent user account.

## Source dates, parsing and citations

NSE often supplies report-year labels rather than exact fiscal boundaries. Automatically discovered records preserve `report_year_label` and mark convention-derived selection dates with `period_dates_inferred=true`. Treat these date values as report selection/filter keys; check the PDF for exact fiscal dates, especially for non-March year ends. Official dissemination timestamps are recorded when available. The original three benchmark sources retain unknown publication dates.

`as_of` filters exclude unknown publication dates and later filings. This is different from filtering the reporting year.

Original PDFs, raw layout text, text-region/table coordinates, extraction methods and source URLs are retained. Quality flags cover selected low-text, encoding and table-extraction problems. They do not prove reading-order or numerical correctness. Borderless/multi-page tables, scanned pages and unusual layouts can need review. Units and standalone/consolidated scope are not silently inferred. Initial table rows are header candidates, not verified semantic headers.

Answers use a 24,000-character evidence budget and keep at most one chunk per document/page/block-type combination. Citation numbers are checked against supplied sources; factual support, completeness, numerical correctness and abstention are not automatically certified. Archived ZIP sources link to the exchange attachment; use local page browsing to inspect their extracted PDFs.

## API usage

The complete request schemas and interactive examples are available at `/docs`.

| Method/path | Purpose |
| --- | --- |
| `GET /health` | Backend reachability only; no remote-model health check. |
| `GET /companies` | Searchable NSE directory, independent of prepared filings. |
| `GET /companies/{symbol}/reports` | Available report-year choices and manifest preparation markers. |
| `POST /prepare` | Start/reuse preparation with `company_id` and the returned `period_end` key. |
| `GET /preparations/{job_id}` | Job status/stage polling. |
| `GET /corpus` | Prepared reports, counts, corpus hash and model/index configuration status. |
| `POST /search` | Retrieve passages without an answer or history rewriting. |
| `POST /ask` | Optional follow-up resolution, retrieval, then a cited answer. |
| `GET /documents/{id}/pages/{number}` | Extracted page text and flags. |
| `GET /documents/{id}/pages/{number}/image` | Locally rendered PNG. |
| `GET /documents/{id}/pdf` | Locally stored original/extracted PDF. |
| `GET /benchmarks` | Saved evaluation reports and corpus compatibility. |

Example preparation request, using a key discovered from `/companies/TCS/reports`:

```json
{"company_id": "TCS", "period_end": "2025-03-31"}
```

Poll the returned `job_id` before asking about an unprepared filing. A manifest's prepared marker alone is not a full index-health guarantee; `/corpus` exposes readiness separately.

Example Q&A request:

```json
{
  "question": "What was the consolidated operating margin?",
  "company_ids": ["INFY"],
  "period_end": "2025-03-31",
  "mode": "hybrid",
  "k": 10
}
```

For a follow-up, add `history` entries with `role` (`user`/`assistant`) and `content`. `/ask` requires exactly one available company/year when history is nonempty. Its response includes both original `question` and resolved `retrieval_question`. Questions are limited to 4,000 characters and `k` to 1–20. Omitting company/year filters allows a standalone API/CLI search across the prepared corpus; the dashboard keeps its session scoped.

## CLI and optional utilities

Seed the fixed pilot corpus if you want to reproduce its benchmark:

```bash
uv run filings download
uv run filings process
uv run filings index
uv run filings stats
```

Download/processing/BM25 do not need model credentials. Dense indexing and generated answers do. `search` and `ask` default to BM25; select hybrid explicitly:

```bash
uv run filings search 'net interest income' --company HDFCBANK --mode bm25
uv run filings ask 'What was Infosys consolidated operating margin in FY2025?' --company INFY --mode hybrid
uv run filings page INFY 77 --render data/qa/infosys-77.png
```

Source manifests can include allowed NSE or BSE attachments: `uv run filings download --config your-sources.json`. Downloading reuses intact cached files and retains prior versions when a replacement is fetched. It does not remotely recheck an unchanged cached URL.

`process --max-pages 10` creates a partial active corpus, not an additive sample. Run full `process` and rebuild the index to restore complete coverage. Corpus/model changes invalidate the dense index; use `filings index` to rebuild through the cache.

Optional difficult-page transcription:

```bash
uv run filings vision-page HDFCBANK 211 --output data/vision_proposals.jsonl
# Inspect/correct the proposal against the source image, then set reviewed=true.
uv run filings process --repairs data/vision_proposals.jsonl
uv run filings index
```

Proposals start unreviewed and do not automatically enter retrieval. The vision command sends one page image to the model; normal extraction does not.

Other helpers:

```bash
uv run filings discover saved_exchange_page.html --base-url https://www.bseindia.com/corporates/ann
uv run python scripts/api_smoke.py
```

Discovery lists allowed PDF links from saved/rendered HTML; no Selenium driver is bundled. The smoke script makes live chat/embedding requests. `scripts/build_pilot_benchmark.py` overwrites the curated benchmark when executed/imported, so use it only when rebuilding those definitions.

## Retrieval benchmark

```bash
uv run filings evaluate --modes bm25
uv run filings evaluate --modes bm25 dense hybrid --output experiments/comparison
```

The bootstrap development set has 20 questions: 18 answerable and two outside its corpus period. Labels were curated/source-checked by the coding agent and use three inspected evidence pages; they are not independent human annotations or an untouched test set.

Evaluation measures evidence-group recall, complete coverage, first relevant reciprocal rank and a novelty-aware DCG variant at 5/10/20. The DCG variant is not standard graded nDCG. Repeated overlapping hits do not replace missing evidence groups. Unanswerable queries are logged but excluded from recall averages. Other valid unannotated passages may be missed by quote matching.

See [benchmark methodology](benchmark/README.md) and [recorded results](BENCHMARK_RESULTS.md). Saved runs include corpus/benchmark hashes and retrieval settings. Adding a report changes the corpus; historical scores should not be presented as current. Latency includes cache/first-load effects and is not a controlled serving comparison. No generated-answer accuracy benchmark is implemented.

## Files and code documentation

[docs/CODE_GUIDE.md](docs/CODE_GUIDE.md) contains the reading order, module responsibilities, generated-data map, and a complete inventory of Python functions/classes/methods. Their source docstrings describe inputs, outputs, important errors and side effects.

| Area | Files |
| --- | --- |
| Application entry points | [streamlit_app.py](streamlit_app.py), [scripts/run_dashboard.py](scripts/run_dashboard.py), [cli.py](src/filings_rag/cli.py) |
| Web interface/backend | [dashboard.py](src/filings_rag/dashboard.py), [api.py](src/filings_rag/api.py) |
| Discovery/preparation | [catalog.py](src/filings_rag/catalog.py), [preparation.py](src/filings_rag/preparation.py), [ingestion.py](src/filings_rag/ingestion.py) |
| Document processing | [processing.py](src/filings_rag/processing.py), [vision.py](src/filings_rag/vision.py) |
| Search/models/answers | [retrieval.py](src/filings_rag/retrieval.py), [providers.py](src/filings_rag/providers.py), [generation.py](src/filings_rag/generation.py) |
| Evaluation/storage/package | [evaluation.py](src/filings_rag/evaluation.py), [storage.py](src/filings_rag/storage.py), [__init__.py](src/filings_rag/__init__.py) |
| Utility scripts | [api_smoke.py](scripts/api_smoke.py), [build_pilot_benchmark.py](scripts/build_pilot_benchmark.py) |
| Tests | [test_pipeline.py](tests/test_pipeline.py), [test_web.py](tests/test_web.py), [test_catalog.py](tests/test_catalog.py) |
| Configuration | [pyproject.toml](pyproject.toml), [uv.lock](uv.lock), [.env.example](.env.example), [.streamlit/config.toml](.streamlit/config.toml), [configs/pilot.json](configs/pilot.json), [.gitignore](.gitignore) |
| Benchmark artifacts/docs | [pilot.jsonl](benchmark/pilot.jsonl), [benchmark README](benchmark/README.md), [results](BENCHMARK_RESULTS.md) |

`data/` contains generated source files, caches, job records and indexes. `experiments/` contains generated runs/answers. These directories, `.env`, and local environments are gitignored.

## Running services separately and checking the project

```bash
# Alternative launcher ports
uv run python scripts/run_dashboard.py --api-port 8001 --ui-port 8502

# Or use two terminals
uv run uvicorn filings_rag.api:app --host 127.0.0.1 --port 8000
uv run streamlit run streamlit_app.py --server.address 127.0.0.1 --server.port 8501

# Offline checks
uv run pytest -q
uv run ruff check src tests scripts streamlit_app.py
```

`FILINGS_API_URL` selects the dashboard's API origin (default `http://127.0.0.1:8000`); the combined launcher sets it automatically for its selected port. `FILINGS_DATA_DIR` selects backend data storage. Saved experiments still default to the project `experiments/` directory.

Automated tests use synthetic files, mocked exchange/model responses, FastAPI's test client and Streamlit AppTest. Live checks separately exercised directory/report discovery, a new TCS report through preparation and cited Q&A, historical ZIP extraction, follow-ups and local page rendering. Browser visual inspection was unavailable because computer-use permissions were off. Tests do not certify general financial-answer or PDF accuracy.

Known boundaries: NSE-only automatic discovery, website-dependent availability, heuristic parsing, inferred date conventions for exchange year labels, a small development benchmark and incomplete cross-company retrieval. The dashboard is a local single-process prototype; additional production features are outside this project's scope.

## References

- [NSE listed-company directories](https://www.nseindia.com/static/market-data/securities-available-for-trading)
- [NSE annual reports](https://www.nseindia.com/companies-listing/corporate-filings-annual-reports)
- [BSE corporate filings](https://www.bseindia.com/corporates)
- [pdfplumber](https://github.com/jsvine/pdfplumber)
- [Google compatible API](https://ai.google.dev/gemini-api/docs/openai), [Gemini 3.8 Flash](https://ai.google.dev/gemini-api/docs/models/gemini-3.8-flash)
- [Streamlit chat input](https://docs.streamlit.io/develop/api-reference/chat/st.chat_input), [fragments](https://docs.streamlit.io/develop/api-reference/execution-flow/st.fragment), [AppTest](https://docs.streamlit.io/develop/api-reference/app-testing/st.testing.v1.apptest)
- [FastAPI lifespan](https://fastapi.tiangolo.com/advanced/events/)
