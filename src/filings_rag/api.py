"""FastAPI endpoints for discovery, preparation, scoped Q&A and local source browsing.

Run with uvicorn filings_rag.api:app. Lifespan setup loads the project .env and
creates a shared corpus service, NSE catalog and background preparation manager.
GET /health only checks backend reachability. All credentials remain on the
backend; response models expose no keys. Tests can inject isolated directories
and disable .env loading. This module creates the app object at import but does
not start a server or perform model calls merely because it is imported.
"""

import io
import json
import os
import threading
import time
from collections import Counter
from contextlib import asynccontextmanager, contextmanager
from datetime import date
from pathlib import Path
from typing import Annotated, Literal

import pdfplumber
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, Field, StringConstraints

from .catalog import ExchangeCatalog
from .generation import answer, contextualize_question
from .preparation import PreparationManager
from .providers import Provider
from .retrieval import Retriever
from .storage import read_jsonl

PROJECT_ROOT = Path(__file__).resolve().parents[2]
Question = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=4000)]


class ChatTurn(BaseModel):
    """Validate one prior user/assistant message for follow-up reference resolution.

    content is stripped and limited to 1–4,000 characters. Only those two roles
    are accepted. Historical messages are not treated as verified source evidence.
    """

    role: Literal['user', 'assistant']
    content: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=4000)]


class QuestionRequest(BaseModel):
    """Validate shared /search and /ask inputs and their fixed source filters.

    Questions are stripped and limited to 1–4,000 characters. mode selects BM25,
    dense or hybrid; k is 1–20. Optional company_ids, report-year date key and
    publication cutoff constrain retrieval. history accepts at most six ChatTurn
    messages and is used only by /ask, which then requires one company/year.
    """

    question: Question = Field(description='A filing question; follow-ups can include recent history.')
    mode: Literal['bm25', 'dense', 'hybrid'] = 'hybrid'
    company_ids: list[str] | None = Field(default=None, min_length=1, max_length=50)
    period_end: date | None = None
    as_of: date | None = None
    k: int = Field(default=10, ge=1, le=20)
    history: list[ChatTurn] = Field(
        default_factory=list,
        max_length=6,
        description='Up to six prior messages. Only /ask uses history; '
        'it requires one selected company and reporting year.',
    )


class PrepareRequest(BaseModel):
    """Validate the issuer and ISO year-selection key supplied to /prepare.

    company_id must match the allowed symbol pattern; actual directory membership
    and report availability are checked by the catalog/manager. period_end is the
    key returned by report discovery, not an independently verified fiscal date.
    No attachment URL can be supplied through this model.
    """

    company_id: Annotated[str, StringConstraints(pattern=r'^[A-Z0-9&._-]{1,40}$')]
    period_end: date


@contextmanager
def pipeline_errors():
    """Translate known operational errors to HTTP responses without upstream payloads."""
    try:
        yield
    except HTTPException:
        raise
    except RuntimeError as exc:
        raise HTTPException(502, str(exc)) from None
    except ValueError as exc:
        raise HTTPException(503, str(exc)) from None
    except OSError:
        raise HTTPException(
            503, 'The filing data is unavailable. Check the backend data directory.'
        ) from None


def public_hit(hit: dict) -> dict:
    """Keep searchable evidence/provenance while omitting duplicate manifest payloads."""
    fields = (
        'chunk_id',
        'document_id',
        'company_id',
        'company_name',
        'page',
        'period_end',
        'text',
        'source_url',
        'block_type',
        'flags',
        'bbox',
        'extraction_method',
        'report_year_label',
        'period_dates_inferred',
    )
    return {
        **{key: value for key, value in hit.items() if key != 'chunk'},
        'chunk': {key: hit['chunk'].get(key) for key in fields},
    }


class FilingService:
    """Reuse corpus state and serve queries/documents within one local API process.

    The service caches manifests/pages and a Retriever until relevant file stamps
    change. Its reentrant lock coordinates refresh, search and preparation commit.
    LLM/embedding settings are resolved once; model availability is not proved by
    configuration flags. Restart the backend after changing .env settings.
    """

    def __init__(self, root: Path, experiments: Path):
        """Store data locations and model settings without calling any model API."""
        self.root = root.resolve()
        self.experiments = experiments.resolve()
        self.lock = threading.RLock()
        self.stamp = None
        self.retriever = None
        self.pages = {}
        self.filings = []
        self.llm = Provider('llm')
        self.embedding = Provider('embedding')

    def refresh(self):
        """Refresh manifests, page lookup and retriever after generated files change.

        Compare existence, modification time and size of corpus/index files. Build
        replacement state before releasing the old retriever client. Callers hold
        self.lock. This reads local data only and performs no model requests.
        """
        paths = [
            self.root / name
            for name in (
                'manifest.jsonl',
                'pages.jsonl',
                'chunks.jsonl',
                'index/metadata.json',
                'index/vectors.npy',
            )
        ]
        stamp = tuple((p.stat().st_mtime_ns, p.stat().st_size) if p.exists() else None for p in paths)
        if stamp == self.stamp:
            return
        filings = [r for r in read_jsonl(self.root / 'manifest.jsonl') if not r.get('superseded')]
        pages = {(p['document_id'], p['page']): p for p in read_jsonl(self.root / 'pages.jsonl')}
        chunk_path = self.root / 'chunks.jsonl'
        retriever = Retriever(self.root) if chunk_path.exists() and chunk_path.stat().st_size else None
        if self.retriever:
            self.retriever.provider.client.close()
        self.filings, self.pages, self.retriever, self.stamp = filings, pages, retriever, stamp

    def overview(self) -> dict:
        """Return prepared documents, coverage counts and configuration/index status.

        Include per-document page/chunk counts, optional exchange year labels, corpus
        hash and nonsecret model IDs. Dense readiness compares existing file/provider
        metadata, not a live API call. It does not fully validate the vector matrix;
        Retriever._dense performs additional checks at search time.
        """
        with self.lock:
            self.refresh()
            documents = {}
            chunk_counts = Counter(c['document_id'] for c in self.retriever.chunks) if self.retriever else {}
            for filing in self.filings:
                doc = filing['document_id']
                if doc in documents:
                    continue
                pages = [p for (d, _), p in self.pages.items() if d == doc]
                documents[doc] = {
                    'document_id': doc,
                    'company_id': filing['company_id'],
                    'company_name': filing['company_name'],
                    'period_end': filing['period_end'],
                    'filing_type': filing['filing_type'],
                    'source_url': filing['attachment_url'],
                    'pages': len(pages),
                    'chunks': chunk_counts.get(doc, 0),
                    'total_pages': max((p['total_pages'] for p in pages), default=0),
                    'report_year_label': filing.get('report_year_label'),
                    'period_dates_inferred': filing.get('period_dates_inferred', False),
                }
            dense_ready = False
            metadata = self.root / 'index/metadata.json'
            if self.retriever and metadata.exists() and (self.root / 'index/vectors.npy').exists():
                meta = json.loads(metadata.read_text())
                dense_ready = (
                    meta.get('corpus_hash') == self.retriever.corpus_hash
                    and meta.get('provider') == self.retriever.provider.identity
                )
            embedding = self.retriever.provider if self.retriever else self.embedding
            return {
                'documents': list(documents.values()),
                'pages': len(self.pages),
                'chunks': len(self.retriever.chunks) if self.retriever else 0,
                'flagged_pages': sum(bool(p['flags']) for p in self.pages.values()),
                'corpus_hash': self.retriever.corpus_hash if self.retriever else None,
                'dense_ready': dense_ready,
                'embedding_configured': bool(embedding and embedding.key and embedding.model),
                'answer_configured': bool(self.llm.key and self.llm.model),
                'answer_model': self.llm.model,
                'embedding_model': embedding.model if embedding else None,
            }

    def search(self, payload: QuestionRequest) -> list[dict]:
        """Run existing retrieval with validated filters and return full internal hits."""
        with self.lock:
            self.refresh()
            if not self.retriever:
                raise HTTPException(
                    503, 'No processed filings. Run filings download and filings process first.'
                )
            known = {r['company_id'] for r in self.filings}
            if payload.company_ids and set(payload.company_ids) - known:
                raise HTTPException(422, 'One or more selected companies are not in this corpus.')
            return self.retriever.search(
                payload.question,
                mode=payload.mode,
                k=payload.k,
                companies=payload.company_ids,
                period_end=payload.period_end.isoformat() if payload.period_end else None,
                as_of=payload.as_of.isoformat() if payload.as_of else None,
            )

    def session_scope(self, payload: QuestionRequest) -> str:
        """Validate the one-company/year scope needed before rewriting a follow-up.

        Match an active manifest source and return a human-readable issuer/report-year
        label. Raise HTTPException(422) for missing, multiple or unavailable choices.
        This makes no model call and does not persist conversation state.
        """
        with self.lock:
            self.refresh()
            if not payload.company_ids or len(payload.company_ids) != 1 or not payload.period_end:
                raise HTTPException(422, 'Follow-up questions require one company and a reporting year.')
            filing = next(
                (
                    r
                    for r in self.filings
                    if r['company_id'] == payload.company_ids[0]
                    and r['period_end'] == payload.period_end.isoformat()
                ),
                None,
            )
            if not filing:
                raise HTTPException(422, 'This company and reporting year are not available.')
            year = filing.get('report_year_label') or f'report ending {filing["period_end"]}'
            return f'{filing["company_name"]} · {year}'

    def page(self, document_id: str, number: int) -> dict:
        """Look up an extracted page by corpus document ID and one-based PDF position."""
        with self.lock:
            self.refresh()
            page = self.pages.get((document_id, number))
            if not page:
                raise HTTPException(404, 'This page is not in the processed corpus.')
            return {key: page[key] for key in ('document_id', 'company_id', 'page', 'text', 'flags')}

    def pdf_path(self, document_id: str) -> Path:
        """Resolve only a known manifest PDF contained within the configured data directory."""
        with self.lock:
            self.refresh()
            record = next((r for r in self.filings if r['document_id'] == document_id), None)
            if not record:
                raise HTTPException(404, 'Unknown filing document.')
            path = (self.root / record['local_path']).resolve()
            if not path.is_relative_to(self.root) or not path.is_file():
                raise HTTPException(404, 'The original PDF is unavailable.')
            return path

    def benchmarks(self) -> dict:
        """Read saved evaluation reports; never rerun model calls from the dashboard."""
        with self.lock:
            self.refresh()
            runs = []
            for path in sorted(self.experiments.glob('*/report.json'), reverse=True):
                try:
                    report = json.loads(path.read_text())
                    runs.append(
                        {
                            'name': path.parent.name,
                            'report': report,
                            'matches_corpus': bool(
                                self.retriever and report.get('corpus_hash') == self.retriever.corpus_hash
                            ),
                        }
                    )
                except (OSError, ValueError):
                    continue
            runs.sort(
                key=lambda r: (len(r['report'].get('results', {})), r['report'].get('created_at', '')),
                reverse=True,
            )
            return {'runs': runs}

    def close(self):
        """Release HTTP clients when the API process shuts down."""
        with self.lock:
            self.llm.client.close()
            self.embedding.client.close()
            if self.retriever:
                self.retriever.provider.client.close()


def create_app(
    data_dir: Path | None = None, experiments_dir: Path | None = None, load_env: bool = True
) -> FastAPI:
    """Build the local FastAPI application with injectable data/report locations.

    Args:
        data_dir: Corpus directory; defaults to FILINGS_DATA_DIR or project data/.
        experiments_dir: Saved evaluation directory; defaults to project experiments/.
        load_env: Load the project .env during lifespan setup; disable in tests.

    Returns:
        An app with discovery, preparation, corpus/search/answer, document and
        benchmark routes. Creating it does not bind a port or call a model.

    Lifespan creates and closes the service, exchange catalog and background
    manager. Changing data_dir does not automatically relocate experiments_dir.
    """

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        """Load backend settings and create shared service/catalog/job-manager state.

        Yield control to the server, then request job shutdown and close clients.
        Tests use load_env=False to avoid reading the real project credential file.
        """
        if load_env:
            load_dotenv(PROJECT_ROOT / '.env')
        root = data_dir or Path(os.getenv('FILINGS_DATA_DIR', str(PROJECT_ROOT / 'data')))
        experiments = experiments_dir or PROJECT_ROOT / 'experiments'
        service = FilingService(root, experiments)
        application.state.service = service
        application.state.catalog = ExchangeCatalog(root)
        application.state.preparation = PreparationManager(service, application.state.catalog)
        try:
            yield
        finally:
            application.state.preparation.close()
            application.state.catalog.close()
            service.close()

    application = FastAPI(
        title='Indian Filings RAG API',
        version='0.1.0',
        lifespan=lifespan,
        description='Search Indian company filings and ask page-grounded questions.',
    )

    @application.get('/health')
    def health():
        """Confirm the backend is reachable; this does not check remote model availability."""
        return {'status': 'ok'}

    @application.get('/corpus')
    def corpus(request: Request):
        """List available reports, coverage and model/index configuration status."""
        with pipeline_errors():
            return request.app.state.service.overview()

    @application.get('/companies')
    def companies(request: Request):
        """Return cached/refreshable NSE issuer choices, independent of prepared reports.

        The coverage label is NSE main-board/SME. This GET may fetch public directory
        CSVs but never invokes a model or downloads filing PDFs.
        """
        with pipeline_errors():
            return {'companies': request.app.state.catalog.companies(), 'coverage': 'NSE main-board and SME'}

    @application.get('/companies/{symbol}/reports')
    def reports(symbol: str, request: Request):
        """Return source-year choices for an exact NSE symbol, including prepared markers.

        This GET can refresh public report metadata. It does not prepare the report
        or guarantee that an old attachment URL remains downloadable.
        """
        with pipeline_errors():
            return {'reports': request.app.state.catalog.reports(symbol)}

    @application.post('/prepare')
    def prepare(payload: PrepareRequest, request: Request):
        """Return a ready/in-flight job or enqueue the requested report's preparation.

        Directory/year errors become HTTP 422. Discovery access failures use the
        operational error mapping. Actual PDF/model work is handled by the worker.
        """
        with pipeline_errors():
            try:
                return request.app.state.preparation.start(payload.company_id, payload.period_end.isoformat())
            except ValueError as exc:
                raise HTTPException(422, str(exc)) from None

    @application.get('/preparations/{job_id}')
    def preparation(job_id: str, request: Request):
        """Return persisted in-memory job status and phase text for UI polling.

        Unknown identifiers return HTTP 404. Polling does not repeat preparation or
        call the model. The stage is descriptive rather than an exact completion rate.
        """
        try:
            return request.app.state.preparation.get(job_id)
        except KeyError:
            raise HTTPException(404, 'Unknown preparation job.') from None

    @application.post('/search')
    def search(payload: QuestionRequest, request: Request):
        """Retrieve passages without requesting a generated answer."""
        with pipeline_errors():
            started = time.perf_counter()
            hits = request.app.state.service.search(payload)
            return {
                'question': payload.question,
                'mode': payload.mode,
                'hits': [public_hit(hit) for hit in hits],
                'elapsed_ms': (time.perf_counter() - started) * 1000,
            }

    @application.post('/ask')
    def ask(payload: QuestionRequest, request: Request):
        """Generate an answer from freshly retrieved, source-filtered evidence.

        Validate model configuration, resolve optional follow-up history inside its
        fixed company/year scope, retrieve the resulting query and call answer().
        Return original and retrieval questions, answer/sources/hits and elapsed time.
        History is stateless request context; the API stores no per-user conversation.
        """
        with pipeline_errors():
            started = time.perf_counter()
            service = request.app.state.service
            if not service.llm.key or not service.llm.model:
                raise HTTPException(503, 'Configure the answer model and API key in the backend .env.')
            retrieval_question = payload.question
            if payload.history:
                scope = service.session_scope(payload)
                retrieval_question = contextualize_question(
                    payload.question, [turn.model_dump() for turn in payload.history], scope
                )
            hits = service.search(payload.model_copy(update={'question': retrieval_question}))
            result = answer(retrieval_question, hits)
            return {
                **result,
                'question': payload.question,
                'retrieval_question': retrieval_question,
                'mode': payload.mode,
                'hits': [public_hit(hit) for hit in hits],
                'elapsed_ms': (time.perf_counter() - started) * 1000,
            }

    @application.get('/documents/{document_id}/pages/{number}')
    def page(document_id: str, number: int, request: Request):
        """Return extracted page text and quality flags for the document browser."""
        with pipeline_errors():
            return request.app.state.service.page(document_id, number)

    @application.get('/documents/{document_id}/pages/{number}/image')
    def page_image(document_id: str, number: int, request: Request):
        """Render a known source page locally as PNG; no vision model is called."""
        with pipeline_errors():
            service = request.app.state.service
            service.page(document_id, number)
            with pdfplumber.open(service.pdf_path(document_id)) as pdf:
                image = pdf.pages[number - 1].to_image(resolution=110).original
                buffer = io.BytesIO()
                image.save(buffer, format='PNG')
            return Response(buffer.getvalue(), media_type='image/png')

    @application.get('/documents/{document_id}/pdf')
    def pdf(document_id: str, request: Request):
        """Download the local original PDF using a manifest-backed identifier."""
        with pipeline_errors():
            return FileResponse(
                request.app.state.service.pdf_path(document_id),
                media_type='application/pdf',
                filename=f'{document_id[:12]}.pdf',
            )

    @application.get('/benchmarks')
    def benchmarks(request: Request):
        """Return existing benchmark reports and whether they match the active corpus."""
        with pipeline_errors():
            return request.app.state.service.benchmarks()

    return application


app = create_app()
