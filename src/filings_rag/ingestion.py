"""Download allowed NSE/BSE PDF or ZIP attachments and retain source provenance.

This module consumes source records from a curated configuration or ExchangeCatalog.
Downloads use content hashes and a resumable manifest. Archive PDFs are selected
without extracting member paths; ambiguous archives are rejected. HTML link
parsing supports saved/rendered pages but does not run Selenium or exchange
pagination itself. Discovery and background coordination live in separate modules.
"""

import io
import json
import re
import time
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urljoin, urlparse

import httpx
from bs4 import BeautifulSoup

from .storage import digest, read_jsonl, write_jsonl

ALLOWED_HOSTS = {'www.nseindia.com', 'nsearchives.nseindia.com', 'www.bseindia.com'}


def exchange_url(url: str) -> str:
    """Validate and return an HTTPS URL on an allowed NSE/BSE host.

    Raise ValueError for other schemes or hosts. Validation does not fetch the URL.
    """
    parsed = urlparse(url)
    if parsed.scheme != 'https' or parsed.hostname not in ALLOWED_HOSTS:
        raise ValueError(f'Expected an HTTPS NSE/BSE URL: {url}')
    return url


def attachment_links(html: str, base_url: str) -> list[str]:
    """Return sorted, unique exchange PDF links found in saved or rendered HTML.

    Resolve relative links against base_url and ignore off-exchange attachments.
    This only discovers links; company, period and completeness need verification.
    """
    exchange_url(base_url)
    links = set()
    for anchor in BeautifulSoup(html, 'html.parser').find_all('a', href=True):
        url = urljoin(base_url, anchor['href'])
        if urlparse(url).path.lower().endswith('.pdf'):
            try:
                links.add(exchange_url(url))
            except ValueError:
                continue
    return sorted(links)


def fetch(client: httpx.Client, url: str) -> bytes:
    """Fetch attachment bytes while validating every redirect destination.

    Args:
        client: Reusable HTTP client with automatic redirects disabled.
        url: HTTPS attachment URL on an allowed exchange host.

    Returns:
        The final response body as bytes. PDF validation happens in download.

    Raises:
        ValueError: A URL leaves the allowlist or the redirect limit is exceeded.
        httpx.HTTPError: A request fails after applicable retries.

    Transient status codes and transport errors receive up to three attempts.
    """
    for attempt in range(3):
        try:
            # Validate each redirect instead of letting a redirect leave the source allowlist.
            target = exchange_url(url)
            for _ in range(5):
                response = client.get(target)
                if response.is_redirect:
                    target = exchange_url(urljoin(target, response.headers['location']))
                    continue
                response.raise_for_status()
                return response.content
            raise ValueError('Too many redirects')
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code not in {429, 500, 502, 503, 504} or attempt == 2:
                raise
        except httpx.TransportError:
            if attempt == 2:
                raise
        time.sleep(2**attempt)
    raise RuntimeError('Download failed')


def download(config: Path, root: Path) -> list[dict]:
    """Download source-configured exchange PDFs/ZIPs and checkpoint their manifest.

    Args:
        config: JSON list of curated or catalog-validated source records.
        root: Workspace for raw attachments, manifest and failure records.

    Returns:
        All retained manifest records, including preserved earlier versions.

    Reuse matching intact cached PDFs, validate new content and name PDF files by
    SHA-256. For ZIPs, retain the archive and record its selected PDF member.
    Checkpoint successful files, continue collecting download failures, then raise
    RuntimeError if any failed. Cached unchanged URLs are not remotely rechecked;
    this downloader is not a change-monitoring service. Prior raw bytes remain.
    """
    sources = json.loads(config.read_text())
    existing = {r['filing_id']: r for r in read_jsonl(root / 'manifest.jsonl')}
    raw = root / 'raw'
    raw.mkdir(parents=True, exist_ok=True)
    failures = []
    with httpx.Client(timeout=120, headers={'User-Agent': 'IndianFilingsResearch/0.1'}) as client:
        for source in sources:
            filing_id = source['filing_id']
            old = existing.get(filing_id)
            if old and old['attachment_url'] == source['attachment_url']:
                path = root / old['local_path']
                if path.exists() and digest(path.read_bytes()) == old['content_hash']:
                    existing[filing_id] = {**old, **source}
                    print(f'cached {filing_id}', flush=True)
                    continue
            try:
                content = fetch(client, source['attachment_url'])
                archive = {}
                if zipfile.is_zipfile(io.BytesIO(content)):
                    archive_hash = digest(content)
                    (raw / f'{archive_hash}.zip').write_bytes(content)
                    content, member = pdf_from_archive(content)
                    archive = {'archive_member': member, 'original_local_path': f'raw/{archive_hash}.zip'}
                if b'%PDF-' not in content[:1024]:
                    raise ValueError('Attachment is not a PDF (possibly an access-block page)')
                sha = digest(content)
                path = raw / f'{sha}.pdf'
                if not path.exists():
                    temporary = path.with_suffix('.part')
                    temporary.write_bytes(content)
                    temporary.replace(path)
                row = {
                    **source,
                    **archive,
                    'document_id': sha,
                    'content_hash': sha,
                    'local_path': str(path.relative_to(root)),
                    'retrieved_at': datetime.now(UTC).isoformat(),
                }
                # Retain earlier versions, never replace the downloaded bytes.
                if old and old['content_hash'] != sha:
                    existing[f'{filing_id}@{old["content_hash"]}'] = {
                        **old,
                        'filing_id': f'{filing_id}@{old["content_hash"]}',
                        'superseded': True,
                    }
                    row['supersedes_document_id'] = old['document_id']
                existing[filing_id] = row
                write_jsonl(root / 'manifest.jsonl', list(existing.values()))
                print(f'downloaded {filing_id}: {len(content):,} bytes', flush=True)
            except (httpx.HTTPError, ValueError) as exc:
                failures.append({'filing_id': filing_id, 'error': str(exc)})
                print(f'failed {filing_id}: {exc}', flush=True)
            time.sleep(1)
    write_jsonl(root / 'manifest.jsonl', list(existing.values()))
    write_jsonl(root / 'download_failures.jsonl', failures)
    if failures:
        raise RuntimeError(f'{len(failures)} download(s) failed; see {root / "download_failures.jsonl"}')
    return list(existing.values())


def pdf_from_archive(content: bytes) -> tuple[bytes, str]:
    """Read a single identifiable annual-report PDF from an NSE ZIP without extracting paths.

    Reject ambiguous archives and oversized members instead of choosing a random
    attachment or allowing unbounded decompression. Original ZIP bytes are retained.
    """
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        members = [
            m
            for m in archive.infolist()
            if not m.is_dir()
            and m.filename.lower().endswith('.pdf')
            and not m.filename.startswith('__MACOSX/')
        ]
        if len(members) != 1:
            named = [
                m
                for m in members
                if re.search(r'annual|integrated|(?:^|[_ -])ar[_ -]', m.filename, re.IGNORECASE)
                and not re.search(r'notice|agm', m.filename, re.IGNORECASE)
            ]
            if len(named) != 1:
                raise ValueError('The exchange ZIP has multiple or no identifiable report PDFs.')
            members = named
        member = members[0]
        if member.file_size > 200 * 1024 * 1024:
            raise ValueError('The archived report exceeds the 200 MB extraction limit.')
        return archive.read(member), member.filename
