"""Offline regression tests for retrieval, evidence scoring and API handling.

Temporary files isolate test data and mocked HTTP responses avoid live model
calls. These checks do not measure real-model answer quality or PDF accuracy.
"""

import json

import httpx
import numpy as np
import pytest

from filings_rag.evaluation import matches, score, validate_queries
from filings_rag.ingestion import attachment_links, fetch
from filings_rag.processing import quality_flags, segments, table_segments
from filings_rag.providers import Provider
from filings_rag.retrieval import Retriever, build_dense
from filings_rag.storage import write_jsonl


def chunk(identifier, company, text, published='2025-06-01'):
    """Build a minimal synthetic chunk record for isolated retrieval/evaluation tests."""
    return {
        'chunk_id': identifier,
        'document_id': f'doc-{company}',
        'company_id': company,
        'company_name': company,
        'page': 1,
        'period_end': '2025-03-31',
        'published_at': published,
        'text': text,
    }


def test_bm25_numbers_filters_and_unknown_dates(tmp_path):
    """Check grouped-number matching, company/period filters and publication-date exclusions."""
    write_jsonl(
        tmp_path / 'chunks.jsonl',
        [
            chunk('a', 'INFY', 'Revenue was 1,62,990 crore; operating margin was 21.1 percent.'),
            chunk('b', 'BANK', 'Revenue was 162990 crore.', None),
            chunk('c', 'MARUTI', 'Vehicle exports grew strongly.', '2026-01-01'),
        ],
    )
    search = Retriever(tmp_path)
    assert search.search('162990', companies=['INFY'])[0]['chunk']['chunk_id'] == 'a'
    assert search.search('unmatchedzzzzz') == []
    assert search.search('revenue', companies=['BANK'], as_of='2025-12-31') == []
    assert search.search('exports', as_of='2025-12-31') == []
    assert search.search('revenue', period_end='2024-03-31') == []


def test_evidence_groups_ignore_duplicate_chunks_and_require_both_companies():
    """Ensure repeated hits cannot substitute for a second required company’s evidence."""
    a = chunk('a', 'INFY', 'Revenue was 162990 crore')
    b = chunk('b', 'MARUTI', 'Exports were 332585 vehicles')
    groups = [
        [{'document_id': a['document_id'], 'page': 1, 'quote': 'Revenue was 162990'}],
        [{'document_id': b['document_id'], 'page': 1, 'quote': 'Exports were 332585'}],
    ]
    result = score([{'chunk': a}, {'chunk': a}], groups, 5)
    assert result['evidence_recall@5'] == 0.5
    assert result['complete@5'] == 0
    result = score([{'chunk': a}, {'chunk': b}], groups, 5)
    assert result['complete@5'] == 1
    assert not matches({**a, 'page': 2}, groups[0][0])


def test_benchmark_rejects_unreviewed_or_fabricated_quotes(tmp_path):
    """Ensure draft labels and quotes absent from the source page are rejected."""
    write_jsonl(tmp_path / 'pages.jsonl', [{'document_id': 'doc', 'page': 1, 'text': 'actual evidence'}])
    query = {
        'query_id': 'q1',
        'answerable': True,
        'review_status': 'draft',
        'evidence_groups': [[{'document_id': 'doc', 'page': 1, 'quote': 'invented'}]],
    }
    with pytest.raises(ValueError, match='not been verified'):
        validate_queries([query], tmp_path)
    query['review_status'] = 'verified'
    with pytest.raises(ValueError, match='not found'):
        validate_queries([query], tmp_path)


def test_exchange_discovery_and_redirect_boundaries():
    """Check relative PDF discovery and rejection of redirects leaving exchange hosts."""
    html = '<a href="/a.PDF">A</a><a href="https://evil.example/b.pdf">B</a>'
    assert attachment_links(html, 'https://www.bseindia.com/corporates') == ['https://www.bseindia.com/a.PDF']
    client = httpx.Client(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(302, headers={'location': 'https://evil.example/b.pdf'})
        )
    )
    with pytest.raises(ValueError, match='Expected an HTTPS'):
        fetch(client, 'https://www.bseindia.com/a.pdf')


def test_chunk_limits_table_context_and_extraction_flags():
    """Check chunk size limits, repeated table context and suspicious-text flags."""
    text = '\n'.join('Standalone revenue 2025 INR crore ' * 30 for _ in range(10))
    pieces = segments(text, max_chars=1000)
    assert max(map(len, pieces)) <= 1000
    tables = table_segments([['Metric', 'FY2025'], ['Unit', 'INR crore']] + [['Revenue', '42']] * 500)
    assert len(tables) > 1
    assert all('INR crore' in t and len(t) <= 3500 for t in tables)
    assert 'low_text_or_image_page' in quality_flags('')
    assert 'encoding_suspect' in quality_flags('\ufffd' * 20 + 'A' * 100)


def test_embedding_cache_order_dense_hybrid_and_stale_index(tmp_path, monkeypatch):
    """Exercise cached embeddings, reordered API results, both vector modes and stale-index rejection."""
    monkeypatch.setenv('EMBEDDING_API_KEY', 'test-placeholder')
    calls = []

    def handler(request):
        """Return deterministic reversed-index embeddings and record mock API calls."""
        body = json.loads(request.content)
        calls.append(body)
        return httpx.Response(
            200,
            json={
                'data': [
                    {'index': i, 'embedding': [1, 0] if 'revenue' in text.lower() else [0, 1]}
                    for i, text in reversed(list(enumerate(body['input'])))
                ]
            },
        )

    transport = httpx.MockTransport(handler)
    original_init = Provider.__init__

    def provider_init(self, kind, cache_path=None, client=None):
        """Inject the mock HTTP transport into providers created during the test."""
        original_init(self, kind, cache_path, client=httpx.Client(transport=transport))

    monkeypatch.setattr(Provider, '__init__', provider_init)
    rows = [chunk('a', 'INFY', 'Revenue increased'), chunk('b', 'BANK', 'Deposits increased')]
    write_jsonl(tmp_path / 'chunks.jsonl', rows)
    build_dense(tmp_path)
    provider = Provider('embedding', tmp_path / 'embedding_cache.sqlite')
    vectors = provider.embed(['deposits', 'revenue', 'deposits'])
    assert np.array_equal(vectors[0], vectors[2])
    previous_calls = len(calls)
    provider.embed(['revenue', 'deposits'])
    assert len(calls) == previous_calls
    search = Retriever(tmp_path)
    assert search.search('revenue', mode='dense')[0]['chunk']['chunk_id'] == 'a'
    assert search.search('revenue', mode='hybrid')[0]['chunk']['chunk_id'] == 'a'
    rows[0]['text'] = 'Different evidence'
    write_jsonl(tmp_path / 'chunks.jsonl', rows)
    with pytest.raises(ValueError, match='stale'):
        Retriever(tmp_path).search('revenue', mode='dense')


def test_generation_rejects_nonexistent_citations(monkeypatch):
    """Ensure a model response cannot cite an evidence number that was never supplied."""
    from filings_rag.generation import answer

    monkeypatch.setattr(Provider, 'chat', lambda self, messages: 'Revenue increased [99].')
    c = {**chunk('a', 'INFY', 'Revenue increased'), 'block_type': 'text', 'source_url': 'https://example.com'}
    with pytest.raises(ValueError, match='invalid citation'):
        answer('Revenue?', [{'chunk': c}])


def test_vision_proposals_require_source_review(tmp_path):
    """Ensure unreviewed vision output cannot be used as a processing repair."""
    from filings_rag.processing import process

    write_jsonl(tmp_path / 'manifest.jsonl', [{'filing_id': 'test'}])
    proposals = tmp_path / 'proposals.jsonl'
    write_jsonl(proposals, [{'document_id': 'doc', 'page': 1, 'text': 'model output', 'reviewed': False}])
    with pytest.raises(ValueError, match='Review vision proposals'):
        process(tmp_path, repairs_path=proposals)


def test_gemini_key_is_scoped_to_google_endpoint(monkeypatch):
    """Ensure the Gemini credential is not implicitly selected for a non-Google endpoint."""
    for key in ('LLM_API_KEY', 'OPENAI_API_KEY', 'EMBEDDING_API_KEY'):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv('GEMINI_API_KEY', 'test-google-placeholder')
    monkeypatch.setenv('LLM_BASE_URL', 'https://generativelanguage.googleapis.com/v1beta/openai')
    monkeypatch.setenv('LLM_MODEL', 'gemini-3.8-flash')
    assert Provider('llm').key == 'test-google-placeholder'
    monkeypatch.setenv('LLM_BASE_URL', 'https://api.openai.com/v1')
    assert Provider('llm').key is None


def test_google_embedding_default_zero_index_and_validation(tmp_path, monkeypatch):
    """Accept Google’s omitted zero index while still rejecting duplicate/missing batch indices."""
    monkeypatch.setenv('GEMINI_API_KEY', 'test-google-placeholder')
    monkeypatch.setenv('EMBEDDING_BASE_URL', 'https://generativelanguage.googleapis.com/v1beta/openai')
    monkeypatch.setenv('EMBEDDING_MODEL', 'gemini-embedding-001')
    payload = {'data': [{'index': 1, 'embedding': [0, 1]}, {'embedding': [1, 0]}]}
    client = httpx.Client(transport=httpx.MockTransport(lambda req: httpx.Response(200, json=payload)))
    provider = Provider('embedding', tmp_path / 'cache.sqlite', client=client)
    assert np.array_equal(provider.embed(['first', 'second']), np.eye(2, dtype=np.float32))
    payload['data'] = [{'embedding': [1, 0]}, {'embedding': [0, 1]}]
    with pytest.raises(ValueError, match='incomplete or duplicate'):
        provider.embed(['third', 'fourth'])
