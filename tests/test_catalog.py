"""Archive and discovery-cache edge cases without live exchange requests."""

import io
import json
import zipfile

import httpx
import pytest

from filings_rag.catalog import ExchangeCatalog
from filings_rag.ingestion import pdf_from_archive


def archive_bytes(files):
    """Build a tiny in-memory ZIP fixture; no filesystem extraction is performed."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w') as archive:
        for name, content in files.items():
            archive.writestr(name, content)
    return buffer.getvalue()


def test_report_zip_selection_and_ambiguity():
    """Accept one report PDF or an unambiguous annual-report name; reject ambiguous attachments."""
    content, name = pdf_from_archive(archive_bytes({'../../report.pdf': b'%PDF-test'}))
    assert content == b'%PDF-test' and name == '../../report.pdf'
    content, name = pdf_from_archive(
        archive_bytes({'notice.pdf': b'%PDF-notice', 'annual-report.pdf': b'%PDF-report'})
    )
    assert name == 'annual-report.pdf' and content == b'%PDF-report'
    with pytest.raises(ValueError, match='multiple or no'):
        pdf_from_archive(archive_bytes({'one.pdf': b'%PDF-a', 'two.pdf': b'%PDF-b'}))


def test_company_cache_retains_directory_during_exchange_outage(tmp_path):
    """Use a previously successful company directory if refreshing the exchange is denied."""
    requests = []

    def handler(request):
        """Serve both real CSV column conventions, then simulate exchange access denial."""
        requests.append(request)
        if len(requests) > 2:
            return httpx.Response(403)
        if 'emerge' in request.url.path:
            return httpx.Response(200, text='SYMBOL,NAME_OF_COMPANY,ISIN_NUMBER\nSMECO,SME Company,SME1\n')
        return httpx.Response(
            200, text='SYMBOL,NAME OF COMPANY,ISIN NUMBER\nTCS,Tata Consultancy Services,TCS1\n'
        )

    catalog = ExchangeCatalog(tmp_path, client=httpx.Client(transport=httpx.MockTransport(handler)))
    assert {r['company_id'] for r in catalog.companies()} == {'TCS', 'SMECO'}
    path = tmp_path / 'catalog/companies.json'
    cached = json.loads(path.read_text())
    cached['saved_at'] = 0
    path.write_text(json.dumps(cached))
    assert {r['company_id'] for r in catalog.companies()} == {'TCS', 'SMECO'}
    catalog.close()
