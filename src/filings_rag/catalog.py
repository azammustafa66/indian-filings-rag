"""Discover NSE main-board/SME companies and annual-report year choices.

Public exchange CSVs and JSON metadata supply company identities and attachment
URLs; a model never invents them. Successful responses are cached locally for
one day (companies) or six hours (reports), with stale-cache reuse on refresh
failure. BSE-only automated discovery is not implemented. Exact fiscal boundaries
are not verified from NSE year labels, so derived selection dates are marked.
"""

import csv
import io
import json
import re
import threading
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import httpx

from .ingestion import exchange_url
from .storage import read_jsonl

DIRECTORIES = {
    'equities': 'https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv',
    'sme': 'https://nsearchives.nseindia.com/emerge/corporates/content/SME_EQUITY_L.csv',
}
REPORT_PAGE = 'https://www.nseindia.com/companies-listing/corporate-filings-annual-reports'


def published_time(value: str | None) -> str | None:
    """Convert an NSE dissemination timestamp to an Indian-time ISO string.

    Args:
        value: Exchange text in DD-Mon-YYYY HH:MM:SS form, or an unknown value.

    Returns:
        A timezone-aware ISO timestamp, or None if the input is absent/invalid.
        A URL timestamp or cover-letter date is not substituted for dissemination.
    """
    try:
        return (
            datetime.strptime(value, '%d-%b-%Y %H:%M:%S').replace(tzinfo=ZoneInfo('Asia/Kolkata')).isoformat()
        )
    except (TypeError, ValueError):
        return None


class ExchangeCatalog:
    """Discover companies and report choices using cached official NSE responses.

    Attributes:
        root: Corpus directory containing the prepared-source manifest.
        cache: Directory for successful company/report metadata snapshots.
        client: HTTP client used for public exchange requests, injectable in tests.
        lock: Reentrant lock serializing cache refresh and replacement.

    This class makes no model calls and does not download the PDF attachments.
    Close it when the API service shuts down.
    """

    def __init__(self, root: Path, client=None):
        """Initialize cache paths, a reusable exchange client and the refresh lock.

        root is the corpus directory. client may be an httpx mock client. Construction
        alone does not fetch the company directory or contact an exchange.
        """
        self.root = root
        self.cache = root / 'catalog'
        self.client = client or httpx.Client(timeout=40, headers={'User-Agent': 'IndianFilingsResearch/0.1'})
        self.lock = threading.RLock()

    def _json(self, name: str, ttl: int, loader):
        """Load cached metadata or refresh it through a zero-argument loader.

        Args:
            name: Internal cache filename, not a user-supplied filesystem path.
            ttl: Freshness duration in seconds.
            loader: Callable returning JSON-serializable exchange metadata.

        Returns:
            Fresh cached data, successfully refreshed data, or previous cached data
            if the refresh raises an HTTP/normalization error.

        Writes successful data with a save timestamp using temporary-file replacement.
        Raises RuntimeError if refresh fails without a previous cache. Malformed
        existing cache JSON and filesystem errors propagate to the caller.
        """
        path = self.cache / name
        with self.lock:
            cached = json.loads(path.read_text()) if path.exists() else None
            if cached and time.time() - cached['saved_at'] < ttl:
                return cached['data']
            try:
                data = loader()
            except (httpx.HTTPError, ValueError) as exc:
                if cached:
                    return cached['data']
                raise RuntimeError('NSE discovery is unavailable right now. Please try again later.') from exc
            path.parent.mkdir(parents=True, exist_ok=True)
            temp = path.with_suffix('.tmp')
            temp.write_text(json.dumps({'saved_at': time.time(), 'data': data}, ensure_ascii=False))
            temp.replace(path)
            return data

    def companies(self) -> list[dict]:
        """Return company_id/name records searchable independently of prepared filings.

        Read main-board and SME CSV schemas, normalize column names, keep one record
        per symbol and sort by company name. Main-board entries take precedence on
        symbol collisions. Merge known manifest issuers not currently in the cached
        directory. Refresh directory metadata after one day; no PDF/model calls occur.
        """

        def load():
            """Read the exchange CSV schemas, which differ between main-board and SME files."""
            found = {}
            for segment, url in DIRECTORIES.items():
                response = self.client.get(url)
                response.raise_for_status()
                for raw in csv.DictReader(io.StringIO(response.content.decode('utf-8-sig'))):
                    row = {
                        key.strip().replace('_', ' '): value.strip()
                        for key, value in raw.items()
                        if key and isinstance(value, str)
                    }
                    symbol, name = row.get('SYMBOL'), row.get('NAME OF COMPANY')
                    if symbol and name:
                        found.setdefault(
                            symbol,
                            {
                                'company_id': symbol,
                                'company_name': name,
                                'isin': row.get('ISIN NUMBER'),
                                'segment': segment,
                            },
                        )
            if not found:
                raise ValueError('NSE company-directory format is not recognized')
            return sorted(found.values(), key=lambda r: r['company_name'].casefold())

        companies = {r['company_id']: r for r in self._json('companies.json', 86400, load)}
        for filing in read_jsonl(self.root / 'manifest.jsonl'):
            companies.setdefault(
                filing['company_id'],
                {
                    'company_id': filing['company_id'],
                    'company_name': filing['company_name'],
                    'segment': 'equities',
                },
            )
        return sorted(companies.values(), key=lambda r: r['company_name'].casefold())

    def reports(self, symbol: str) -> list[dict]:
        """Discover report-year sources for an exact directory symbol.

        Args:
            symbol: Validated NSE symbol selected from companies(), not free-form text.

        Returns:
            Source dictionaries sorted by descending period_end selection key, with
            exchange URLs/year labels and a prepared marker for manifest records.

        Try applicable main-board/SME report endpoints, validate attachment hosts,
        prefer the latest known dissemination per year and cache metadata for six
        hours. Manifest records override that year's remote choice for reuse; the
        prepared marker alone does not prove chunks or an index exist. Derived
        period dates are conventions, explicitly marked as inferred. An empty list
        means no usable report choices were returned, not proof none ever existed.

        Raises:
            ValueError: The symbol is invalid or absent from the directory.
            RuntimeError: Required discovery fails and no prior cache is available.
        """
        if not re.fullmatch(r'[A-Z0-9&._-]{1,40}', symbol):
            raise ValueError('Choose a valid NSE company symbol from search results.')
        company = next((r for r in self.companies() if r['company_id'] == symbol), None)
        if not company:
            raise ValueError('This company was not found in the NSE listed-company directory.')

        def load():
            """Normalize the exchange year labels and attachment URLs without inventing availability."""
            reports = {}
            for segment in dict.fromkeys([company['segment'], 'equities', 'sme']):
                response = self.client.get(
                    'https://www.nseindia.com/api/annual-reports', params={'index': segment, 'symbol': symbol}
                )
                response.raise_for_status()
                data = response.json().get('data', [])
                for raw in data:
                    try:
                        start, end = int(raw['fromYr']), int(raw['toYr'])
                        if not 1950 <= start <= end <= datetime.now().year + 1:
                            continue
                        url = exchange_url(raw.get('fileName') or raw['an_attchmntFile'])
                    except (KeyError, ValueError, TypeError):
                        continue
                    # NSE exposes year labels rather than exact fiscal dates. The
                    # date is a selection/filter key; mark its convention explicitly.
                    period = f'{end}-03-31' if start < end else f'{end}-12-31'
                    published = published_time(raw.get('disseminationDateTime'))
                    source = {
                        'filing_id': f'{symbol.lower()}-{start}-{end}-annual-nse',
                        'company_id': symbol,
                        'company_name': company['company_name'],
                        'exchange': 'NSE',
                        'filing_type': 'annual_report',
                        'period_start': f'{start}-04-01' if start < end else f'{start}-01-01',
                        'period_end': period,
                        'period_dates_inferred': True,
                        'report_year_label': f'FY {start}–{str(end)[-2:]}' if start < end else f'Year {end}',
                        'published_at': published,
                        'source_url': REPORT_PAGE,
                        'attachment_url': url,
                    }
                    if period not in reports or (published or '') > (reports[period]['published_at'] or ''):
                        reports[period] = source
                if reports:
                    break
            return list(reports.values())

        remote = self._json(f'reports-{symbol}.json', 21600, load)
        reports = {r['period_end']: {**r, 'prepared': False} for r in remote}
        for filing in read_jsonl(self.root / 'manifest.jsonl'):
            if filing['company_id'] == symbol and not filing.get('superseded'):
                reports[filing['period_end']] = {**filing, 'prepared': True}
        return sorted(reports.values(), key=lambda r: r['period_end'], reverse=True)

    def close(self):
        """Release the exchange HTTP client on backend shutdown."""
        self.client.close()
