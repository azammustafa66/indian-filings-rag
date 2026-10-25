"""Prepare an exchange-selected report in a single background worker.

Jobs download/extract one report in a resumable workspace, merge a staged corpus,
reuse the shared embedding cache and publish under the API service lock. Job
states are queued, running, ready and failed; phase descriptions are not precise
completion percentages. Published records/index have rollback copies during
commit. Raw/cache files may remain after failures for reuse on a user retry.
"""

import json
import logging
import shutil
import tempfile
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .ingestion import download
from .processing import process
from .providers import Provider
from .retrieval import build_dense
from .storage import read_jsonl, write_jsonl

logger = logging.getLogger(__name__)


class PreparationManager:
    """Queue resumable annual-report preparation separately from request handling.

    One executor worker prevents competing ingestion publications within this API
    process. A job-state lock protects status snapshots and their JSONL file;
    the FilingService lock protects final corpus/index publication. This is a
    local-demo job manager, not a distributed multi-process task queue.
    """

    def __init__(self, service, catalog):
        """Create the worker/locks and recover persisted preparation records.

        service supplies the corpus directory and publication lock; catalog resolves
        exchange source records. Nonterminal stored jobs become failed/retryable in
        memory after a restart. They do not automatically resume until the user loads
        the report again. Construction performs no exchange or model requests.
        """
        self.service, self.catalog = service, catalog
        self.root = service.root
        self.path = self.root / 'preparation_jobs.jsonl'
        self.lock = threading.RLock()
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='prepare-report')
        self.closing = threading.Event()
        self.jobs = {r['job_id']: r for r in read_jsonl(self.path)}
        for job in self.jobs.values():
            if job['status'] not in ('ready', 'failed'):
                job.update(
                    status='failed', stage='Interrupted by backend restart. Load this report to retry.'
                )

    def update(self, job_id: str, **values):
        """Merge status fields into a known job and persist all records with updated_at.

        Serialize writes under the job-state lock. This method returns None; unknown
        IDs raise KeyError and filesystem errors propagate.
        """
        with self.lock:
            self.jobs[job_id].update(values, updated_at=time.time())
            write_jsonl(self.path, list(self.jobs.values()))

    def get(self, job_id: str) -> dict:
        """Return a shallow copy of a known job record for status polling.

        No files, exchange endpoints or models are queried. Unknown IDs raise KeyError.
        """
        with self.lock:
            return dict(self.jobs[job_id])

    def start(self, company: str, period: str) -> dict:
        """Reuse searchable company/year data or enqueue one exchange-validated report.

        Args:
            company: Exact directory symbol selected by the client.
            period: ISO selection key returned by the report discovery endpoint.

        Returns:
            A job dictionary with job_id, issuer/year, status, stage and timestamps.
            Existing searchable chunks produce a ready result; an in-flight job for
            the same issuer/year is reused. Otherwise a new queued job is persisted.

        Discovery may make cached/public HTTP requests. Downloading, parsing and
        embedding happen asynchronously in run(), not in this request. Readiness
        here concerns searchable chunks; dense-index readiness is checked separately.
        Raises ValueError for an unavailable choice and propagates discovery failures.
        """
        with self.lock:
            ready = next(
                (
                    d
                    for d in self.service.overview()['documents']
                    if d['company_id'] == company and d['period_end'] == period and d['chunks']
                ),
                None,
            )
            if not ready:
                active = next(
                    (
                        j
                        for j in self.jobs.values()
                        if j['company_id'] == company
                        and j['period_end'] == period
                        and j['status'] not in ('ready', 'failed')
                    ),
                    None,
                )
                if active:
                    return dict(active)
            source = ready or next(
                (r for r in self.catalog.reports(company) if r['period_end'] == period), None
            )
            if not source:
                raise ValueError('No annual-report attachment is available for this company/year.')
            identifier = uuid.uuid4().hex
            job = {
                'job_id': identifier,
                'company_id': company,
                'company_name': source['company_name'],
                'period_end': period,
                'status': 'ready' if ready else 'queued',
                'stage': 'Report is already prepared.' if ready else 'Queued for preparation.',
                'created_at': time.time(),
                'updated_at': time.time(),
            }
            self.jobs[identifier] = job
            self.update(identifier)
            if not ready:
                self.executor.submit(self.run, identifier, source)
            return dict(job)

    def stage(self, job_id: str, message: str):
        """Mark a job running with an informative phase message.

        Raise RuntimeError if shutdown was requested, allowing the worker to stop
        between stages. Stage messages do not imply an exact percentage or ETA.
        """
        if self.closing.is_set():
            raise RuntimeError('Preparation was interrupted. Load the report again to resume.')
        self.update(job_id, status='running', stage=message)

    def run(self, job_id: str, source: dict):
        """Prepare a selected source and record either ready or failed status.

        Args:
            job_id: Previously persisted job identifier.
            source: Exchange-validated source record, not an arbitrary client URL.

        Download and extract into preparation/SYMBOL/PERIOD, stage merged manifests,
        pages/blocks/chunks, and build vectors with the shared global cache when an
        embedding provider is configured. Without one, publish keyword-search data
        and leave dense readiness to the usual index checks. Publish only after the
        stages succeed. On failure, log the exception and save a sanitized job message;
        exceptions are captured rather than returned to the initiating HTTP request.
        Downloaded files and document caches remain for a user-triggered retry.
        """
        try:
            working = self.root / 'preparation' / source['company_id'] / source['period_end']
            working.mkdir(parents=True, exist_ok=True)
            config = working / 'source.json'
            config.write_text(json.dumps([{k: v for k, v in source.items() if k != 'prepared'}]))
            self.stage(job_id, 'Downloading the annual report from NSE…')
            download(config, working)
            self.stage(job_id, 'Extracting PDF text, tables and source pages…')
            process(working)
            if not read_jsonl(working / 'chunks.jsonl'):
                raise ValueError('No searchable text was extracted from this report.')
            with tempfile.TemporaryDirectory(prefix='publication-', dir=working) as temp:
                snapshot = Path(temp)
                new_manifest = read_jsonl(working / 'manifest.jsonl')
                new_docs = {r['document_id'] for r in new_manifest}
                manifest = {r['filing_id']: r for r in read_jsonl(self.root / 'manifest.jsonl')}
                for row in new_manifest:
                    if any(
                        r['document_id'] == row['document_id'] and r['company_id'] != row['company_id']
                        for r in manifest.values()
                    ):
                        raise ValueError(
                            'The downloaded attachment conflicts with an existing company record.'
                        )
                    manifest[row['filing_id']] = row
                write_jsonl(snapshot / 'manifest.jsonl', list(manifest.values()))
                for name in ('pages.jsonl', 'blocks.jsonl', 'chunks.jsonl'):
                    old = [r for r in read_jsonl(self.root / name) if r['document_id'] not in new_docs]
                    write_jsonl(snapshot / name, [*old, *read_jsonl(working / name)])
                pages = read_jsonl(snapshot / 'pages.jsonl')
                write_jsonl(
                    snapshot / 'quality_issues.jsonl',
                    [
                        {k: p[k] for k in ('document_id', 'company_id', 'page', 'flags')}
                        for p in pages
                        if p['flags']
                    ],
                )
                provider = Provider('embedding')
                try:
                    configured = bool(provider.key and provider.model)
                finally:
                    provider.client.close()
                if configured:
                    self.stage(job_id, 'Creating embeddings; existing vectors are reused from cache…')
                    build_dense(snapshot, embedding_cache=self.root / 'embedding_cache.sqlite')
                self.stage(job_id, 'Publishing the prepared report…')
                self.publish(snapshot, working)
            self.update(job_id, status='ready', stage='Report ready. You can start asking questions.')
        except Exception as exc:
            logger.exception('Report preparation failed')
            # Expected pipeline errors are already sanitized; never reflect an arbitrary traceback.
            message = (
                str(exc)
                if isinstance(exc, (ValueError, RuntimeError))
                else 'Preparation failed. Retry loading the report.'
            )
            self.update(job_id, status='failed', stage=message)

    def publish(self, snapshot: Path, working: Path):
        """Publish staged records/index while holding the shared service lock.

        Args:
            snapshot: Temporary directory containing merged records and optional index.
            working: Report workspace containing raw files and extraction caches.

        Back up currently published files, copy new source/cache data, then replace
        records/index through temporary files. On a commit exception, attempt to
        restore previous published files and re-raise. Raw/cache copies can remain
        for reuse; they are not the active corpus until its records are published.
        This lock coordinates this service's requests, not external CLI processes.
        """
        names = ['manifest.jsonl', 'pages.jsonl', 'blocks.jsonl', 'chunks.jsonl', 'quality_issues.jsonl']
        names += (
            [str(p.relative_to(snapshot)) for p in (snapshot / 'index').glob('*')]
            if (snapshot / 'index').exists()
            else []
        )
        with self.service.lock, tempfile.TemporaryDirectory(prefix='rollback-', dir=working) as backup_dir:
            backup = Path(backup_dir)
            for name in names:
                current = self.root / name
                if current.exists():
                    (backup / name).parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(current, backup / name)
            try:
                (self.root / 'raw').mkdir(exist_ok=True)
                for path in (working / 'raw').iterdir():
                    if path.is_file():
                        shutil.copy2(path, self.root / 'raw' / path.name)
                for path in (working / 'processed').iterdir():
                    shutil.copytree(path, self.root / 'processed' / path.name, dirs_exist_ok=True)
                for name in names:
                    destination = self.root / name
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    pending = destination.with_suffix(destination.suffix + '.pending')
                    shutil.copy2(snapshot / name, pending)
                    pending.replace(destination)
            except Exception:
                for name in names:
                    previous = backup / name
                    if previous.exists():
                        shutil.copy2(previous, self.root / name)
                    elif (self.root / name).exists():
                        (self.root / name).unlink()
                raise

    def close(self):
        """Request shutdown at the next stage boundary and cancel queued jobs.

        The executor does not wait here and an active request/extraction stage is not
        instantly interrupted. A later backend restart marks unfinished jobs retryable.
        """
        self.closing.set()
        self.executor.shutdown(wait=False, cancel_futures=True)
