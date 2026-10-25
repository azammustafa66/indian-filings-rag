# Code guide

This guide maps every Python file and every named function, method and class to its responsibility. It is based on the current source; detailed argument/return/error/side-effect descriptions live in the source docstrings. The README covers installation and use.

## Reading order and request flow

1. Start with `streamlit_app.py` and `scripts/run_dashboard.py` to understand how the two local services start.
2. Read `dashboard.py` and `api.py` for UI requests, validation and company/year session scope.
3. Follow `catalog.py` → `preparation.py` → `ingestion.py` → `processing.py` for a new annual report.
4. Follow `providers.py` and `retrieval.py` for cached vectors, BM25, cosine similarity and rank fusion.
5. Read `generation.py` for follow-up rewriting and evidence-grounded answers; `evaluation.py` for retrieval scoring.
6. Use `storage.py`, `vision.py`, the CLI/utilities and tests as supporting references.

```text
Search company/year
  dashboard.session_picker → GET /companies and /companies/{symbol}/reports
  → ExchangeCatalog → public NSE metadata / local cache

Load missing report
  POST /prepare → PreparationManager.start → background run
  → download → process → staged merged corpus → build_dense(shared cache)
  → publish under FilingService lock → status ready → activate UI session

Ask / follow up
  POST /ask → validate scope → optional contextualize_question
  → FilingService.search → Retriever.search → generation.answer
  → response text + numbered evidence + local session history
```

A source can be listed in the exchange catalog without being prepared. A manifest preparation marker is not a full vector-index health check. The UI uses corpus readiness and polls background jobs separately.

## Data concepts

| Concept | Meaning |
| --- | --- |
| Company ID | NSE symbol used to resolve the selected issuer. |
| Filing record | Exchange source metadata, attachment URL, report-year key and local provenance. |
| Document ID | SHA-256 of the PDF bytes; an original ZIP has a separate hash/path. |
| Page | One-based position within the PDF, including cover letters; a PDF page can contain a printed two-page spread. |
| Block | Extracted text region or table with coordinates and provenance. |
| Chunk | A character-bounded piece of block content that retrieval can rank. |
| Hit | `rank`, method-specific `score`, and the retrieved `chunk`; not a confidence probability. |
| Embedding | Model-generated numeric representation; returned vectors and the search matrix are unit-normalized. |
| Evidence group | One required piece of support with alternative acceptable source quotes. Comparisons can require multiple groups. |
| Preparation job | Persisted identifier/scope/status/stage for a background load, not a user chat session. |
| Active Q&A session | Browser-session company/year and conversation state; changing filings clears previous context. |

NSE report labels do not independently prove exact fiscal boundaries. Automatically derived date keys are marked `period_dates_inferred`; prefer `report_year_label` for context and the actual PDF for exact dates. Publication timestamps, reporting periods and selection keys are different concepts.

## Where work happens

| Local | External requests |
| --- | --- |
| PDF/ZIP storage, hashing, parsing, table/column heuristics, chunks | Public NSE directory/report metadata and exchange attachment downloads |
| BM25, vector storage, exact cosine search, reciprocal rank fusion | Uncached Gemini Embedding inputs |
| Cache reuse, API validation, session state, source-image rendering, benchmark metrics | Gemini follow-up rewrite, generated answer, explicit page-image transcription |

The frontend never reads the model key. The backend loads `.env` at startup. Conversation history resolves references; only fresh retrieved evidence supports each answer. Citation ID checks do not establish financial correctness.

## Configuration and non-Python files

| File | Responsibility |
| --- | --- |
| [README.md](../README.md) | Current installation, UI/API/CLI workflows and limitations. |
| [pyproject.toml](../pyproject.toml) | Dependencies, package build, console entry point and Ruff settings. |
| [uv.lock](../uv.lock) | Resolved dependency versions. |
| [.env.example](../.env.example) | Empty credential fields and compatible Gemini endpoint/model defaults. Actual `.env` is local and gitignored. |
| [.streamlit/config.toml](../.streamlit/config.toml) | UI theme, disabled usage telemetry and rerun-on-save setting. |
| [.gitignore](../.gitignore) | Excludes generated data, secrets, environments and experiments. |
| [Dockerfile](../Dockerfile) | Installs locked runtime dependencies, starts both services, and checks their health. |
| [.dockerignore](../.dockerignore) | Excludes secrets, local corpus, caches, and development files from image builds. |
| [configs/pilot.json](../configs/pilot.json) | Optional fixed three-report seed sources for the original benchmark; not the searchable company universe. |
| [benchmark/pilot.jsonl](../benchmark/pilot.jsonl) | Development questions, reference answers, filters and source-span evidence. |
| [benchmark/README.md](../benchmark/README.md) | Label provenance, metric definitions and benchmark limits. |
| [BENCHMARK_RESULTS.md](../BENCHMARK_RESULTS.md) | Historical comparison on the original corpus, with its scope caveat. |

## Python module and symbol inventory

Names below are qualified by their containing class/function. This distinguishes, for example, `FilingService.search` from the nested route handler `create_app.search`. Nested test callbacks and property methods are included. Short lambdas are described after the inventory; they have no function body in which to place a normal docstring.


### `streamlit_app.py`

[Open source](../streamlit_app.py)

Streamlit entry point; the dashboard communicates with the separate FastAPI service.

No named functions or classes are defined in this file. Its module documentation above explains its entry-point/import behavior.

### `scripts/run_dashboard.py`

[Open source](../scripts/run_dashboard.py)

Start FastAPI and Streamlit together locally or in Docker. The optional `--ui-host`
controls dashboard binding; the API stays on loopback. Ctrl+C and Docker SIGTERM
both reach the child-process cleanup path.

| Symbol | Type | Responsibility |
| --- | --- | --- |
| `main` | Function | Launch both services with shared port settings and clean them up on exit. |
| `main.stop_on_signal` | Function | Convert SIGTERM into the same cleanup path used for Ctrl+C. |

### `src/filings_rag/__init__.py`

[Open source](../src/filings_rag/__init__.py)

Indian listed-company filings research package.

Streamlit provides the interface, FastAPI coordinates on-demand report preparation
and scoped Q&A, and the CLI exposes individual pipeline steps. Importing this
package alone does not download filings, read credentials or call a model.

No named functions or classes are defined in this file. Its module documentation above explains its entry-point/import behavior.

### `src/filings_rag/api.py`

[Open source](../src/filings_rag/api.py)

FastAPI endpoints for discovery, preparation, scoped Q&A and local source browsing.

Run with uvicorn filings_rag.api:app. Lifespan setup loads the project .env and
creates a shared corpus service, NSE catalog and background preparation manager.
GET /health only checks backend reachability. All credentials remain on the
backend; response models expose no keys. Tests can inject isolated directories
and disable .env loading. This module creates the app object at import but does
not start a server or perform model calls merely because it is imported.

| Symbol | Type | Responsibility |
| --- | --- | --- |
| `ChatTurn` | Class | Validate one prior user/assistant message for follow-up reference resolution. |
| `QuestionRequest` | Class | Validate shared /search and /ask inputs and their fixed source filters. |
| `PrepareRequest` | Class | Validate the issuer and ISO year-selection key supplied to /prepare. |
| `pipeline_errors` | Function | Translate known operational errors to HTTP responses without upstream payloads. |
| `public_hit` | Function | Keep searchable evidence/provenance while omitting duplicate manifest payloads. |
| `FilingService` | Class | Reuse corpus state and serve queries/documents within one local API process. |
| `FilingService.__init__` | Method | Store data locations and model settings without calling any model API. |
| `FilingService.refresh` | Method | Refresh manifests, page lookup and retriever after generated files change. |
| `FilingService.overview` | Method | Return prepared documents, coverage counts and configuration/index status. |
| `FilingService.search` | Method | Run existing retrieval with validated filters and return full internal hits. |
| `FilingService.session_scope` | Method | Validate the one-company/year scope needed before rewriting a follow-up. |
| `FilingService.page` | Method | Look up an extracted page by corpus document ID and one-based PDF position. |
| `FilingService.pdf_path` | Method | Resolve only a known manifest PDF contained within the configured data directory. |
| `FilingService.benchmarks` | Method | Read saved evaluation reports; never rerun model calls from the dashboard. |
| `FilingService.close` | Method | Release HTTP clients when the API process shuts down. |
| `create_app` | Function | Build the local FastAPI application with injectable data/report locations. |
| `create_app.lifespan` | Function | Load backend settings and create shared service/catalog/job-manager state. |
| `create_app.health` | Function | Confirm the backend is reachable; this does not check remote model availability. |
| `create_app.corpus` | Function | List available reports, coverage and model/index configuration status. |
| `create_app.companies` | Function | Return cached/refreshable NSE issuer choices, independent of prepared reports. |
| `create_app.reports` | Function | Return source-year choices for an exact NSE symbol, including prepared markers. |
| `create_app.prepare` | Function | Return a ready/in-flight job or enqueue the requested report's preparation. |
| `create_app.preparation` | Function | Return persisted in-memory job status and phase text for UI polling. |
| `create_app.search` | Function | Retrieve passages without requesting a generated answer. |
| `create_app.ask` | Function | Generate an answer from freshly retrieved, source-filtered evidence. |
| `create_app.page` | Function | Return extracted page text and quality flags for the document browser. |
| `create_app.page_image` | Function | Render a known source page locally as PNG; no vision model is called. |
| `create_app.pdf` | Function | Download the local original PDF using a manifest-backed identifier. |
| `create_app.benchmarks` | Function | Return existing benchmark reports and whether they match the active corpus. |

### `src/filings_rag/dashboard.py`

[Open source](../src/filings_rag/dashboard.py)

Streamlit interface for searchable company/year selection and filing Q&A.

The main-page selector reads the exchange directory through FastAPI, loads missing
reports via background jobs and opens a company/year-scoped conversation. A
fragment polls preparation status every two seconds. Completed responses and
page previews live in Streamlit session state, so widget reruns do not regenerate
answers. All discovery, retrieval and model operations go through the backend.

| Symbol | Type | Responsibility |
| --- | --- | --- |
| `BackendError` | Class | An actionable backend connectivity or request error suitable for the dashboard. |
| `api_json` | Function | Call the local JSON API and translate transport/HTTP failures to BackendError. |
| `cached_json` | Function | Cache read-only metadata/report responses for 30 seconds. |
| `financial_year_label` | Function | Show March-year-end reports as FY YYYY–YY, otherwise retain the exact year end. |
| `activate_session` | Function | Open a prepared issuer/year session and clear prior conversation/preview state. |
| `session_examples` | Function | Build starter questions for the selected company/report rather than other filings. |
| `conversation_history` | Function | Return at most six bounded prior messages for follow-up reference resolution. |
| `preparation_status` | Function | Poll one pending preparation job every two seconds as a Streamlit fragment. |
| `session_picker` | Function | Render exchange-wide type-to-search company and available-year controls. |
| `sidebar` | Function | Return retrieval options combined with the fixed active session scope. |
| `evidence_card` | Function | Show one passage with company, page, rank and an exchange PDF link. |
| `render_reply` | Function | Render a saved answer or search result with correctly numbered context passages. |
| `question_view` | Function | Render prior responses and submit one new question with the active filing scope. |
| `document_view` | Function | Browse local page text/images for the active session, or all prepared reports. |
| `benchmark_view` | Function | Chart saved retrieval reports without running evaluation or calling a model. |
| `main` | Function | Assemble the dashboard, including operation from an initially empty corpus. |

### `src/filings_rag/catalog.py`

[Open source](../src/filings_rag/catalog.py)

Discover NSE main-board/SME companies and annual-report year choices.

Public exchange CSVs and JSON metadata supply company identities and attachment
URLs; a model never invents them. Successful responses are cached locally for
one day (companies) or six hours (reports), with stale-cache reuse on refresh
failure. BSE-only automated discovery is not implemented. Exact fiscal boundaries
are not verified from NSE year labels, so derived selection dates are marked.

| Symbol | Type | Responsibility |
| --- | --- | --- |
| `published_time` | Function | Convert an NSE dissemination timestamp to an Indian-time ISO string. |
| `ExchangeCatalog` | Class | Discover companies and report choices using cached official NSE responses. |
| `ExchangeCatalog.__init__` | Method | Initialize cache paths, a reusable exchange client and the refresh lock. |
| `ExchangeCatalog._json` | Method | Load cached metadata or refresh it through a zero-argument loader. |
| `ExchangeCatalog.companies` | Method | Return company_id/name records searchable independently of prepared filings. |
| `ExchangeCatalog.companies.load` | Function | Read the exchange CSV schemas, which differ between main-board and SME files. |
| `ExchangeCatalog.reports` | Method | Discover report-year sources for an exact directory symbol. |
| `ExchangeCatalog.reports.load` | Function | Normalize the exchange year labels and attachment URLs without inventing availability. |
| `ExchangeCatalog.close` | Method | Release the exchange HTTP client on backend shutdown. |

### `src/filings_rag/preparation.py`

[Open source](../src/filings_rag/preparation.py)

Prepare an exchange-selected report in a single background worker.

Jobs download/extract one report in a resumable workspace, merge a staged corpus,
reuse the shared embedding cache and publish under the API service lock. Job
states are queued, running, ready and failed; phase descriptions are not precise
completion percentages. Published records/index have rollback copies during
commit. Raw/cache files may remain after failures for reuse on a user retry.

| Symbol | Type | Responsibility |
| --- | --- | --- |
| `PreparationManager` | Class | Queue resumable annual-report preparation separately from request handling. |
| `PreparationManager.__init__` | Method | Create the worker/locks and recover persisted preparation records. |
| `PreparationManager.update` | Method | Merge status fields into a known job and persist all records with updated_at. |
| `PreparationManager.get` | Method | Return a shallow copy of a known job record for status polling. |
| `PreparationManager.start` | Method | Reuse searchable company/year data or enqueue one exchange-validated report. |
| `PreparationManager.stage` | Method | Mark a job running with an informative phase message. |
| `PreparationManager.run` | Method | Prepare a selected source and record either ready or failed status. |
| `PreparationManager.publish` | Method | Publish staged records/index while holding the shared service lock. |
| `PreparationManager.close` | Method | Request shutdown at the next stage boundary and cancel queued jobs. |

### `src/filings_rag/ingestion.py`

[Open source](../src/filings_rag/ingestion.py)

Download allowed NSE/BSE PDF or ZIP attachments and retain source provenance.

This module consumes source records from a curated configuration or ExchangeCatalog.
Downloads use content hashes and a resumable manifest. Archive PDFs are selected
without extracting member paths; ambiguous archives are rejected. HTML link
parsing supports saved/rendered pages but does not run Selenium or exchange
pagination itself. Discovery and background coordination live in separate modules.

| Symbol | Type | Responsibility |
| --- | --- | --- |
| `exchange_url` | Function | Validate and return an HTTPS URL on an allowed NSE/BSE host. |
| `attachment_links` | Function | Return sorted, unique exchange PDF links found in saved or rendered HTML. |
| `fetch` | Function | Fetch attachment bytes while validating every redirect destination. |
| `download` | Function | Download source-configured exchange PDFs/ZIPs and checkpoint their manifest. |
| `pdf_from_archive` | Function | Read a single identifiable annual-report PDF from an NSE ZIP without extracting paths. |

### `src/filings_rag/processing.py`

[Open source](../src/filings_rag/processing.py)

Turn varied PDF layouts into page records, evidence blocks and search chunks.

Original files and raw layout text remain available alongside conservative
column extraction, detected tables and optional reviewed vision repairs. Quality
flags identify selected extraction problems, not complete correctness.

| Symbol | Type | Responsibility |
| --- | --- | --- |
| `text_regions` | Function | Extract one full-page text region or two regions separated by a central gutter. |
| `quality_flags` | Function | Return heuristic labels for low text, suspect glyphs or failed table extraction. |
| `segments` | Function | Split text into character-bounded chunks with optional overlapping lines. |
| `table_segments` | Function | Serialize table rows into bounded chunks with repeated initial-row context. |
| `process` | Function | Extract downloaded PDFs and rebuild the active page, block and chunk files. |

### `src/filings_rag/providers.py`

[Open source](../src/filings_rag/providers.py)

Call compatible model APIs, cache embedding responses and return normalized vectors.

Settings come from environment variables loaded by the CLI. Gemini uses
Google-scoped credentials and its OpenAI-compatible endpoint. This module
handles HTTP retries, response checks and a persistent SQLite embedding cache.
Cached values retain the response vectors; embed() normalizes them when returning
the float32 array used by local search.

| Symbol | Type | Responsibility |
| --- | --- | --- |
| `Provider` | Class | HTTP adapter for chat or embedding requests using environment configuration. |
| `Provider.__init__` | Method | Resolve endpoint, model and credentials and prepare the HTTP client. |
| `Provider.identity` | Property | Return the endpoint/model pair used for cache and index compatibility checks. |
| `Provider.request` | Method | POST a model request and return the decoded JSON response. |
| `Provider.embed` | Method | Return unit-normalized float32 vectors in the same order as input texts. |
| `Provider.chat` | Method | Send text or multimodal messages and return the first nonempty answer. |

### `src/filings_rag/retrieval.py`

[Open source](../src/filings_rag/retrieval.py)

Rank evidence chunks with BM25, dense cosine similarity or reciprocal rank fusion.

BM25 postings/statistics are built locally in memory. Dense document vectors and
compatibility metadata are persisted; query vectors use the same provider/model
and shared text cache. API calls are only needed for uncached embeddings. Hybrid
fusion uses rank positions, not direct addition of incompatible score scales.

| Symbol | Type | Responsibility |
| --- | --- | --- |
| `tokenize` | Function | Lowercase text and extract word/decimal tokens for BM25. |
| `embedding_text` | Function | Build the shared BM25/document-embedding representation for a chunk. |
| `build_dense` | Function | Embed the active chunk corpus and persist vectors plus compatibility metadata. |
| `Retriever` | Class | Search one processed corpus with BM25, dense cosine or fused rankings. |
| `Retriever.__init__` | Method | Load root/chunks.jsonl and build BM25 postings, lengths and corpus fingerprint. |
| `Retriever._dense` | Method | Load and memoize dense vectors after checking corpus, provider, order and shape. |
| `Retriever.search` | Method | Return ranked hit dictionaries containing rank, score and the source chunk. |

### `src/filings_rag/generation.py`

[Open source](../src/filings_rag/generation.py)

Resolve follow-ups and generate answers grounded in newly retrieved evidence.

The first question needs no rewrite. For later turns, Gemini uses bounded prior
conversation only to resolve references, never as verified financial evidence.
The answer model receives fresh passages and numbered page references. Citation
IDs are validated; claim support, numerical correctness and abstention are not
automatically certified. CLI questions remain independent unless callers use the
separate contextualize_question helper.

| Symbol | Type | Responsibility |
| --- | --- | --- |
| `contextualize_question` | Function | Rewrite a follow-up as a standalone query within the selected filing scope. |
| `answer` | Function | Generate an evidence-grounded answer from ranked retrieval hits. |

### `src/filings_rag/vision.py`

[Open source](../src/filings_rag/vision.py)

Transcribe one PDF page through a vision API and save a review proposal.

This optional path helps with difficult layouts or scanned content. Proposals
start unreviewed and do not automatically replace text in the search corpus.

| Symbol | Type | Responsibility |
| --- | --- | --- |
| `extract_page` | Function | Render one filing page, transcribe it with the model and save a review proposal. |

### `src/filings_rag/evaluation.py`

[Open source](../src/filings_rag/evaluation.py)

Evaluate retrieval against source quotes grouped by required evidence.

A group contains alternative acceptable passages; comparison questions can
require multiple groups. This evaluates retrieval, not generated-answer accuracy
or abstention. The novelty-aware DCG variant is not standard graded nDCG.

| Symbol | Type | Responsibility |
| --- | --- | --- |
| `normalize` | Function | Normalize case, whitespace and table serialization separators for quote matching. |
| `matches` | Function | Check that a chunk contains the gold quote on the same document and PDF page. |
| `score` | Function | Score the first k hits against all required evidence groups for one question. |
| `validate_queries` | Function | Reject duplicate query IDs, unverified labels and invalid source evidence. |
| `evaluate` | Function | Run retrieval modes on a benchmark split and save metrics plus ranked outputs. |

### `src/filings_rag/storage.py`

[Open source](../src/filings_rag/storage.py)

Shared SHA-256 hashing and JSON Lines file helpers.

JSONL stores one record per line. Hashes identify source PDFs and detect changes
to the corpus or benchmark; temporary-file replacement limits partial writes.

| Symbol | Type | Responsibility |
| --- | --- | --- |
| `digest` | Function | Return a hexadecimal SHA-256 hash of bytes or UTF-8-encoded text. |
| `read_jsonl` | Function | Read nonblank JSONL records, returning an empty list if the file is absent. |
| `write_jsonl` | Function | Write records as Unicode-preserving JSONL through a temporary file. |
| `fingerprint` | Function | Hash records with sorted object keys while preserving their list order. |

### `src/filings_rag/cli.py`

[Open source](../src/filings_rag/cli.py)

Command-line entry point for the filings RAG pipeline.

The filings console command loads .env, parses arguments, dispatches work to the
relevant module and prints results. Importing this module does not run commands.

| Symbol | Type | Responsibility |
| --- | --- | --- |
| `emit` | Function | Print a value as indented JSON while retaining readable Unicode characters. |
| `main` | Function | Load .env, parse command-line arguments and dispatch the selected operation. |

### `scripts/api_smoke.py`

[Open source](../scripts/api_smoke.py)

Run a small live chat and embedding API connection check.

Execute from the repository root after configuring .env. Running this script
makes model API calls, prints a short response and vector dimensions, and writes
embeddings to data/embedding_cache.sqlite. Credentials are not printed.

No named functions or classes are defined in this file. Its module documentation above explains its entry-point/import behavior.

### `scripts/build_pilot_benchmark.py`

[Open source](../scripts/build_pilot_benchmark.py)

Materialize the curated pilot questions against downloaded source pages.

Run from the repository root after processing. Top-level code resolves source
hashes, checks annotated phrases and writes benchmark/pilot.jsonl. It does not
independently review the question labels; importing this script also runs it.

| Symbol | Type | Responsibility |
| --- | --- | --- |
| `evidence` | Function | Resolve a company/page quote to a source-hashed evidence record. |

### `tests/test_pipeline.py`

[Open source](../tests/test_pipeline.py)

Offline regression tests for retrieval, evidence scoring and API handling.

Temporary files isolate test data and mocked HTTP responses avoid live model
calls. These checks do not measure real-model answer quality or PDF accuracy.

| Symbol | Type | Responsibility |
| --- | --- | --- |
| `chunk` | Function | Build a minimal synthetic chunk record for isolated retrieval/evaluation tests. |
| `test_bm25_numbers_filters_and_unknown_dates` | Function | Check grouped-number matching, company/period filters and publication-date exclusions. |
| `test_evidence_groups_ignore_duplicate_chunks_and_require_both_companies` | Function | Ensure repeated hits cannot substitute for a second required company’s evidence. |
| `test_benchmark_rejects_unreviewed_or_fabricated_quotes` | Function | Ensure draft labels and quotes absent from the source page are rejected. |
| `test_exchange_discovery_and_redirect_boundaries` | Function | Check relative PDF discovery and rejection of redirects leaving exchange hosts. |
| `test_chunk_limits_table_context_and_extraction_flags` | Function | Check chunk size limits, repeated table context and suspicious-text flags. |
| `test_embedding_cache_order_dense_hybrid_and_stale_index` | Function | Exercise cached embeddings, reordered API results, both vector modes and stale-index rejection. |
| `test_embedding_cache_order_dense_hybrid_and_stale_index.handler` | Function | Return deterministic reversed-index embeddings and record mock API calls. |
| `test_embedding_cache_order_dense_hybrid_and_stale_index.provider_init` | Function | Inject the mock HTTP transport into providers created during the test. |
| `test_generation_rejects_nonexistent_citations` | Function | Ensure a model response cannot cite an evidence number that was never supplied. |
| `test_vision_proposals_require_source_review` | Function | Ensure unreviewed vision output cannot be used as a processing repair. |
| `test_gemini_key_is_scoped_to_google_endpoint` | Function | Ensure the Gemini credential is not implicitly selected for a non-Google endpoint. |
| `test_google_embedding_default_zero_index_and_validation` | Function | Accept Google’s omitted zero index while still rejecting duplicate/missing batch indices. |

### `tests/test_web.py`

[Open source](../tests/test_web.py)

Offline integration tests for the API, Streamlit sessions and report preparation.

Exchange-shaped CSV/JSON responses, synthetic filing records and mocked model
stages isolate tests from real .env credentials and network services. These tests
exercise issuer/year choices beyond the prepared corpus, history scoping, job
publication/failure isolation and UI reruns. They do not establish real-model
answer quality or general PDF extraction accuracy.

| Symbol | Type | Responsibility |
| --- | --- | --- |
| `web_corpus` | Function | Create two source/page/chunk records and configure dummy model credentials. |
| `client` | Function | Serve the actual API in process without loading local .env credentials. |
| `client.exchange_response` | Function | Serve exchange-shaped company/year data, including an unprepared company, offline. |
| `fake_answer` | Function | Return a deterministic cited answer while preserving selected evidence IDs. |
| `test_api_overview_pages_and_reports` | Function | Check corpus discovery, page/PDF serving and saved-report provenance. |
| `test_api_search_and_answer_filters` | Function | Exercise real BM25 filtering and the answer endpoint without a model call. |
| `test_api_rejects_invalid_questions` | Function | Validate blanks, result limits, retrieval choices, company lists and dates. |
| `test_api_errors_and_empty_corpus` | Function | Return useful HTTP failures for upstream model errors and absent corpus data. |
| `test_api_errors_and_empty_corpus.unavailable` | Function | Simulate the sanitized provider error raised by the existing adapter. |
| `test_api_refreshes_changed_chunks` | Function | Refresh the cached retriever when a generated corpus file changes. |
| `test_api_follow_up_uses_history_with_fixed_company_and_year` | Function | Resolve references while preserving source scope and rejecting unavailable sessions. |
| `test_api_follow_up_uses_history_with_fixed_company_and_year.rewrite` | Function | Record conversation context and return a standalone retrieval question. |
| `test_streamlit_question_flow_and_reruns` | Function | Drive Streamlit through real API routes and ensure reruns do not repeat answers. |
| `test_streamlit_question_flow_and_reruns.request` | Function | Bridge dashboard requests to an in-process FastAPI test client. |
| `test_streamlit_backend_unavailable` | Function | Show a reconnect/startup hint rather than an exception when the API is offline. |
| `test_streamlit_backend_unavailable.offline` | Function | Simulate an unavailable local API without opening a network connection. |
| `test_streamlit_year_options_follow_the_chosen_company` | Function | Offer company-specific report years and drop a year unavailable for the new issuer. |
| `test_streamlit_year_options_follow_the_chosen_company.request` | Function | Use the isolated API's refreshed corpus for the year-selection widgets. |
| `test_directory_and_years_include_an_unprepared_company` | Function | Company search must include exchange issuers outside the current corpus. |
| `mocked_preparation` | Function | Simulate PDF extraction/indexing while exercising real background queue and publication logic. |
| `mocked_preparation.download` | Function | Write a tiny exchange attachment and its real manifest shape to the isolated workspace. |
| `mocked_preparation.process` | Function | Create one extracted page/chunk plus the document cache expected at publication. |
| `mocked_preparation.build` | Function | Record shared-cache use; model/vector math is covered by the existing retrieval tests. |
| `wait_for_preparation` | Function | Wait briefly for the isolated worker, failing explicitly instead of hanging the test suite. |
| `test_preparation_publishes_new_report_and_reuses_ready_data` | Function | Prepare a company absent from the corpus without losing existing report data. |
| `test_failed_preparation_preserves_existing_corpus` | Function | An extraction failure leaves the searchable published snapshot unchanged. |
| `test_failed_preparation_preserves_existing_corpus.fail` | Function | Simulate a report parser failure before snapshot publication. |
| `test_streamlit_loads_a_company_outside_the_prepared_list` | Function | A searchable unprepared issuer/year loads in the background and opens its Q&A session. |
| `test_streamlit_loads_a_company_outside_the_prepared_list.request` | Function | Bridge directory, preparation status and corpus refresh calls into the test API. |

### `tests/test_catalog.py`

[Open source](../tests/test_catalog.py)

Archive and discovery-cache edge cases without live exchange requests.

| Symbol | Type | Responsibility |
| --- | --- | --- |
| `archive_bytes` | Function | Build a tiny in-memory ZIP fixture; no filesystem extraction is performed. |
| `test_report_zip_selection_and_ambiguity` | Function | Accept one report PDF or an unambiguous annual-report name; reject ambiguous attachments. |
| `test_company_cache_retains_directory_during_exchange_outage` | Function | Use a previously successful company directory if refreshing the exchange is denied. |
| `test_company_cache_retains_directory_during_exchange_outage.handler` | Function | Serve both real CSV column conventions, then simulate exchange access denial. |

## Anonymous helpers, constants and import behavior

- `MODES` in `dashboard.py` maps retrieval identifiers to display labels. UI lambdas format company/year/document/evaluation labels; they do not retrieve evidence or invoke a model.
- `DIRECTORIES` and `REPORT_PAGE` in `catalog.py` identify the official discovery sources. Sorting callbacks order normalized records.
- `ALLOWED_HOSTS` in `ingestion.py` restricts downloads/redirects to exchange hosts. `PARSER_VERSION` in `processing.py` identifies extraction-cache settings and block IDs.
- `SYSTEM` in `generation.py` is the answer prompt. It requests citations and scope/units discipline but does not guarantee compliance.
- `I`, `H` and `M` in `scripts/build_pilot_benchmark.py` are lambda quote helpers pinned to Infosys PDF page 77, HDFC Bank page 211 and Maruti page 38. `specs` holds the curated development question definitions.
- Test lambdas return deterministic mock HTTP/model outputs; tests do not use real model credentials.
- `scripts/api_smoke.py` makes live calls at execution/import. `scripts/build_pilot_benchmark.py` rebuilds the benchmark at execution/import. `streamlit_app.py` invokes the UI at execution/import. Do not import these as passive utility modules.
- `api.py` creates its `app` object on import, but service setup happens during lifespan. `scripts/run_dashboard.py` and `cli.py` use `__main__` guards.

## Generated storage map

| Path | Writer/purpose |
| --- | --- |
| `data/raw/<hash>.pdf` | Downloader: original direct PDF or the selected PDF read from a ZIP. |
| `data/raw/<hash>.zip` | Downloader: retained original exchange archive. |
| `data/manifest.jsonl` | Source metadata, document hashes, local paths and retained versions. |
| `data/download_failures.jsonl` | Failures from the latest download run in that workspace. |
| `data/processed/<hash>/pages.jsonl`, `blocks.jsonl` | Per-document extraction caches and parser settings. |
| `data/pages.jsonl` | Published raw-layout and reading-order page records. |
| `data/blocks.jsonl`, `chunks.jsonl` | Published evidence blocks and searchable chunks. |
| `data/quality_issues.jsonl` | Selected extraction flags; not a complete correctness audit. |
| `data/embedding_cache.sqlite` | Provider endpoint/model/exact-text embedding responses, reused across snapshots. |
| `data/index/vectors.npy`, `metadata.json` | Normalized matrix plus corpus hash, provider, dimensions and chunk order. |
| `data/catalog/companies.json` | Successful NSE directory snapshot and save time. |
| `data/catalog/reports-SYMBOL.json` | Successful report-year metadata and save time. |
| `data/preparation_jobs.jsonl` | Persisted job statuses/stages. |
| `data/preparation/SYMBOL/PERIOD/` | Resumable source config, downloaded files and per-report extraction outputs. |
| `publication-*`, `rollback-*` inside a preparation workspace | Temporary staged merged corpus/index and commit backups. |
| `data/vision_proposals.jsonl` | Optional transcriptions with review status; not automatically searchable. |
| `data/qa/*.png` | Optional page renders for source inspection. |
| `experiments/<run>/report.json`, `retrieval_runs.jsonl` | Hashed evaluation summaries and ranked per-query evidence. |
| `experiments/.../*smoke*.json` | Explicitly saved live-check outputs; not user chat persistence. |

Paths shown are defaults. CLI `--data-dir` and backend `FILINGS_DATA_DIR` select data locations. Evaluation, vision and render outputs have their own options; changing a data directory alone does not relocate all those separate defaults. Background preparation expects one coordinated API worker; external CLI corpus mutations are not covered by the service lock.

## Useful Python documentation commands

```bash
uv run python -m pydoc filings_rag.catalog
uv run python -m pydoc filings_rag.preparation
uv run python -c 'from filings_rag.providers import Provider; help(Provider.embed)'
```

Use package modules with these commands, not scripts with import-time actions. Docstrings describe actual behavior and boundaries, not a claim that every exchange layout or model response is correct. Automated tests use mocks/synthetic data; separate live checks cover the recorded real-service paths.
