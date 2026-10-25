"""Offline integration tests for the API, Streamlit sessions and report preparation.

Exchange-shaped CSV/JSON responses, synthetic filing records and mocked model
stages isolate tests from real .env credentials and network services. These tests
exercise issuer/year choices beyond the prepared corpus, history scoping, job
publication/failure isolation and UI reruns. They do not establish real-model
answer quality or general PDF extraction accuracy.
"""

import json
import time
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient
from streamlit.testing.v1 import AppTest

from filings_rag import api, dashboard
from filings_rag import preparation as preparation_module
from filings_rag.storage import digest, fingerprint, read_jsonl, write_jsonl


@pytest.fixture
def web_corpus(tmp_path, monkeypatch):
    """Create two source/page/chunk records and configure dummy model credentials."""
    monkeypatch.setenv('LLM_API_KEY', 'test-secret-never-returned')
    monkeypatch.setenv('LLM_MODEL', 'test-answer-model')
    monkeypatch.setenv('LLM_BASE_URL', 'https://example.invalid/v1')
    root = tmp_path / 'data'
    root.mkdir()
    filings, pages, chunks = [], [], []
    for company, name, text in [
        ('INFY', 'Infosys Limited', 'Consolidated operating margin was 21.1 percent.'),
        ('HDFCBANK', 'HDFC Bank Limited', 'Net interest income grew by 13 percent.'),
    ]:
        content = b'%PDF-test-fixture-' + company.encode()
        doc = digest(content)
        (root / f'{doc}.pdf').write_bytes(content)
        filings.append(
            {
                'document_id': doc,
                'filing_id': company + '-fy2025-test',
                'company_id': company,
                'company_name': name,
                'period_end': '2025-03-31',
                'filing_type': 'annual_report',
                'local_path': f'{doc}.pdf',
                'attachment_url': 'https://nsearchives.nseindia.com/a.pdf',
            }
        )
        pages.append(
            {
                'document_id': doc,
                'company_id': company,
                'page': 1,
                'total_pages': 1,
                'text': text,
                'flags': [],
            }
        )
        chunks.append(
            {
                'chunk_id': doc + '-chunk',
                'document_id': doc,
                'company_id': company,
                'company_name': name,
                'page': 1,
                'text': text,
                'flags': [],
                'period_end': '2025-03-31',
                'published_at': '2025-06-01',
                'source_url': filings[-1]['attachment_url'],
                'block_type': 'text',
                'bbox': [0, 0, 600, 800],
                'extraction_method': 'test',
            }
        )
    for filename, records in [('manifest.jsonl', filings), ('pages.jsonl', pages), ('chunks.jsonl', chunks)]:
        write_jsonl(root / filename, records)
    experiments = tmp_path / 'experiments'
    (experiments / 'test-run').mkdir(parents=True)
    report = {
        'corpus_hash': fingerprint(chunks),
        'created_at': '2026-10-25',
        'results': {'bm25': {'metrics': {'evidence_recall@10': 0.8, 'complete@10': 0.7, 'mrr@10': 0.6}}},
    }
    (experiments / 'test-run/report.json').write_text(json.dumps(report))
    return root, experiments, filings


@pytest.fixture
def client(web_corpus):
    """Serve the actual API in process without loading local .env credentials."""
    root, experiments, _ = web_corpus
    with TestClient(api.create_app(root, experiments, load_env=False)) as connection:

        def exchange_response(request):
            """Serve exchange-shaped company/year data, including an unprepared company, offline."""
            if request.url.path.endswith('EQUITY_L.csv'):
                if 'emerge' in request.url.path:
                    return httpx.Response(200, text='SYMBOL,NAME_OF_COMPANY,SERIES\n')
                return httpx.Response(
                    200,
                    text='SYMBOL,NAME OF COMPANY,ISIN NUMBER\n'
                    'INFY,Infosys Limited,TEST1\nHDFCBANK,HDFC Bank Limited,TEST2\n'
                    'TCS,Tata Consultancy Services Limited,TEST3\n',
                )
            symbol = request.url.params.get('symbol')
            filings = [r for r in read_jsonl(root / 'manifest.jsonl') if r['company_id'] == symbol]
            if symbol == 'TCS':
                filings = [
                    {
                        'period_end': '2025-03-31',
                        'company_name': 'Tata Consultancy Services Limited',
                        'attachment_url': 'https://nsearchives.nseindia.com/annual_reports/tcs.pdf',
                    }
                ]
            return httpx.Response(
                200,
                json={
                    'data': [
                        {
                            'fromYr': str(int(r['period_end'][:4]) - 1),
                            'toYr': r['period_end'][:4],
                            'companyName': r['company_name'],
                            'fileName': r['attachment_url'],
                        }
                        for r in filings
                    ]
                },
            )

        connection.app.state.catalog.client.close()
        connection.app.state.catalog.client = httpx.Client(transport=httpx.MockTransport(exchange_response))
        yield connection


def fake_answer(question, hits):
    """Return a deterministic cited answer while preserving selected evidence IDs."""
    first = hits[0]['chunk']
    return {
        'answer': 'The margin was 21.1% [1].',
        'citation_support_verified': False,
        'sources': [
            {
                'citation': 1,
                'chunk_id': first['chunk_id'],
                'document_id': first['document_id'],
                'company_id': first['company_id'],
                'page': first['page'],
                'url': first['source_url'] + '#page=1',
            }
        ],
    }


def test_api_overview_pages_and_reports(client, web_corpus):
    """Check corpus discovery, page/PDF serving and saved-report provenance."""
    assert client.get('/health').json() == {'status': 'ok'}
    response = client.get('/corpus')
    data = response.json()
    assert data['pages'] == data['chunks'] == 2
    assert len(data['documents']) == 2
    assert 'test-secret-never-returned' not in response.text
    doc = web_corpus[2][0]['document_id']
    assert '21.1' in client.get(f'/documents/{doc}/pages/1').json()['text']
    assert client.get(f'/documents/{doc}/pages/2').status_code == 404
    assert client.get(f'/documents/{doc}/pdf').content.startswith(b'%PDF-test-fixture')
    assert client.get('/documents/unknown/pdf').status_code == 404
    assert client.get('/benchmarks').json()['runs'][0]['matches_corpus']


def test_api_search_and_answer_filters(client, monkeypatch):
    """Exercise real BM25 filtering and the answer endpoint without a model call."""
    monkeypatch.setattr(api, 'answer', fake_answer)
    body = {'question': '  operating margin  ', 'mode': 'bm25', 'company_ids': ['INFY']}
    search = client.post('/search', json=body).json()
    assert search['question'] == 'operating margin'
    assert [hit['chunk']['company_id'] for hit in search['hits']] == ['INFY']
    result = client.post('/ask', json=body).json()
    assert '[1]' in result['answer'] and result['sources'][0]['company_id'] == 'INFY'
    assert not result['citation_support_verified']
    assert not client.post('/search', json={**body, 'as_of': '2025-01-01'}).json()['hits']
    assert client.post('/search', json={**body, 'company_ids': ['UNKNOWN']}).status_code == 422


@pytest.mark.parametrize(
    'invalid',
    [
        {'question': '   '},
        {'question': 'margin', 'k': 21},
        {'question': 'margin', 'mode': 'wrong'},
        {'question': 'margin', 'company_ids': []},
        {'question': 'margin', 'period_end': 'not-a-date'},
        {'question': 'margin', 'history': [{'role': 'system', 'content': 'Invalid role'}]},
        {'question': 'margin', 'history': [{'role': 'user', 'content': 'Earlier'}] * 7},
    ],
)
def test_api_rejects_invalid_questions(client, invalid):
    """Validate blanks, result limits, retrieval choices, company lists and dates."""
    assert client.post('/ask', json=invalid).status_code == 422


def test_api_errors_and_empty_corpus(client, tmp_path, monkeypatch):
    """Return useful HTTP failures for upstream model errors and absent corpus data."""

    def unavailable(question, hits):
        """Simulate the sanitized provider error raised by the existing adapter."""
        raise RuntimeError('llm API returned HTTP 429')

    monkeypatch.setattr(api, 'answer', unavailable)
    response = client.post('/ask', json={'question': 'margin', 'mode': 'bm25'})
    assert response.status_code == 502 and '429' in response.json()['detail']
    with TestClient(api.create_app(tmp_path / 'empty', load_env=False)) as empty:
        assert empty.get('/corpus').json()['chunks'] == 0
        assert empty.post('/search', json={'question': 'margin', 'mode': 'bm25'}).status_code == 503


def test_api_refreshes_changed_chunks(client, web_corpus):
    """Refresh the cached retriever when a generated corpus file changes."""
    assert client.get('/corpus').json()['chunks'] == 2
    path = web_corpus[0] / 'chunks.jsonl'
    chunks = [json.loads(line) for line in path.read_text().splitlines()]
    chunks.append({**chunks[0], 'chunk_id': 'new-chunk', 'text': 'Dividend policy overview'})
    write_jsonl(path, chunks)
    assert client.get('/corpus').json()['chunks'] == 3
    assert client.post('/search', json={'question': 'dividend', 'mode': 'bm25'}).json()['hits']


def test_api_follow_up_uses_history_with_fixed_company_and_year(client, monkeypatch):
    """Resolve references while preserving source scope and rejecting unavailable sessions."""
    rewrites = []

    def rewrite(question, history, scope):
        """Record conversation context and return a standalone retrieval question."""
        rewrites.append((question, history, scope))
        return 'Infosys consolidated operating margin in FY2025 compared with the previous year'

    monkeypatch.setattr(api, 'contextualize_question', rewrite)
    monkeypatch.setattr(api, 'answer', fake_answer)
    body = {
        'question': 'And the previous year?',
        'mode': 'bm25',
        'company_ids': ['INFY'],
        'period_end': '2025-03-31',
        'history': [
            {'role': 'user', 'content': 'What was the operating margin?'},
            {'role': 'assistant', 'content': 'It was 21.1% [1].'},
        ],
    }
    response = client.post('/ask', json=body)
    assert response.is_success
    result = response.json()
    assert result['question'] == 'And the previous year?'
    assert 'operating margin' in result['retrieval_question']
    assert {hit['chunk']['company_id'] for hit in result['hits']} == {'INFY'}
    assert {hit['chunk']['period_end'] for hit in result['hits']} == {'2025-03-31'}
    assert rewrites[0][1] == body['history']
    assert 'Infosys Limited' in rewrites[0][2] and '2025-03-31' in rewrites[0][2]
    assert client.post('/ask', json={**body, 'period_end': None}).status_code == 422
    assert client.post('/ask', json={**body, 'period_end': '2024-03-31'}).status_code == 422
    assert len(rewrites) == 1


def test_streamlit_question_flow_and_reruns(client, monkeypatch):
    """Drive Streamlit through real API routes and ensure reruns do not repeat answers."""
    monkeypatch.setattr(api, 'answer', fake_answer)
    monkeypatch.setattr(api, 'contextualize_question', lambda question, history, scope: 'operating margin')
    calls = []

    def request(base, route, method='GET', body=None):
        """Bridge dashboard requests to an in-process FastAPI test client."""
        calls.append((method, route, body))
        response = client.request(method, route, json=body)
        assert response.is_success, response.text
        return response.json()

    monkeypatch.setattr(dashboard, 'api_json', request)
    dashboard.cached_json.clear()
    try:
        app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / 'streamlit_app.py')).run(timeout=20)
        assert not app.exception
        assert app.title[0].value == 'Explore company filings'
        assert app.metric[0].value == '2'
        assert not app.chat_input
        assert {'company_choice', 'year_choice'} <= {widget.key for widget in app.main.selectbox}
        assert 'company_choice' not in {widget.key for widget in app.sidebar.selectbox}
        assert app.main.button(key='start_session').disabled
        assert app.selectbox(key='year_choice').disabled
        app.selectbox(key='company_choice').set_value('INFY').run(timeout=20)
        assert app.selectbox(key='year_choice').options == ['FY 2024–25']
        app.selectbox(key='year_choice').set_value('2025-03-31').run(timeout=20)
        app.button(key='start_session').click().run(timeout=20)
        assert app.metric[0].value == '1'
        app.chat_input[0].set_value('What was the operating margin?').run(timeout=20)
        assert not app.exception
        assert len(app.chat_message) == 2
        assert sum(route == '/ask' for _, route, _ in calls) == 1
        assert any('21.1%' in markdown.value for markdown in app.markdown)
        app.slider[0].set_value(12).run(timeout=20)
        assert sum(route == '/ask' for _, route, _ in calls) == 1
        app.chat_input[0].set_value('And the previous year?').run(timeout=20)
        assert not app.exception
        requests = [body for _, route, body in calls if route == '/ask']
        assert requests[0]['history'] == []
        assert requests[1]['history'][0]['content'] == 'What was the operating margin?'
        assert all(
            body['company_ids'] == ['INFY'] and body['period_end'] == '2025-03-31' for body in requests
        )
        app.toggle[0].set_value(False).run(timeout=20)
        app.chat_input[0].set_value('operating margin').run(timeout=20)
        assert not app.exception
        assert sum(route == '/search' for _, route, _ in calls) == 1
        app.selectbox(key='company_choice').set_value('HDFCBANK').run(timeout=20)
        assert len(app.chat_message) == 6  # A pending choice does not silently change the active scope.
        app.selectbox(key='year_choice').set_value('2025-03-31').run(timeout=20)
        app.button(key='start_session').click().run(timeout=20)
        assert not app.chat_message
        app.toggle[0].set_value(True).run(timeout=20)
        app.chat_input[0].set_value('net interest income').run(timeout=20)
        assert not app.exception
        last = [body for _, route, body in calls if route == '/ask'][-1]
        assert last['company_ids'] == ['HDFCBANK'] and last['history'] == []
    finally:
        dashboard.cached_json.clear()


def test_streamlit_backend_unavailable(monkeypatch):
    """Show a reconnect/startup hint rather than an exception when the API is offline."""

    def offline(*args, **kwargs):
        """Simulate an unavailable local API without opening a network connection."""
        raise dashboard.BackendError('Cannot reach the filings API.')

    monkeypatch.setattr(dashboard, 'api_json', offline)
    dashboard.cached_json.clear()
    try:
        app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / 'streamlit_app.py')).run(timeout=20)
        assert not app.exception
        assert app.error and 'Cannot reach' in app.error[0].value
        assert 'run_dashboard.py' in app.code[0].value
    finally:
        dashboard.cached_json.clear()


def test_streamlit_year_options_follow_the_chosen_company(client, web_corpus, monkeypatch):
    """Offer company-specific report years and drop a year unavailable for the new issuer."""
    root = web_corpus[0]
    old_doc = digest(b'%PDF-older-infosys')
    (root / f'{old_doc}.pdf').write_bytes(b'%PDF-older-infosys')
    for filename in ('manifest.jsonl', 'pages.jsonl', 'chunks.jsonl'):
        records = read_jsonl(root / filename)
        old = {**records[0], 'document_id': old_doc}
        if filename != 'pages.jsonl':
            old['period_end'] = '2024-03-31'
        if filename == 'manifest.jsonl':
            old['local_path'] = f'{old_doc}.pdf'
        if filename == 'chunks.jsonl':
            old['chunk_id'] = old_doc + '-chunk'
        write_jsonl(root / filename, [*records, old])

    def request(base, route, method='GET', body=None):
        """Use the isolated API's refreshed corpus for the year-selection widgets."""
        response = client.request(method, route, json=body)
        assert response.is_success
        return response.json()

    monkeypatch.setattr(dashboard, 'api_json', request)
    dashboard.cached_json.clear()
    try:
        app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / 'streamlit_app.py')).run(timeout=20)
        app.selectbox(key='company_choice').set_value('INFY').run(timeout=20)
        assert app.selectbox(key='year_choice').options == ['FY 2024–25', 'FY 2023–24']
        app.selectbox(key='year_choice').set_value('2024-03-31').run(timeout=20)
        app.selectbox(key='company_choice').set_value('HDFCBANK').run(timeout=20)
        assert not app.exception
        assert app.selectbox(key='year_choice').options == ['FY 2024–25']
        assert app.selectbox(key='year_choice').value is None
        assert app.button(key='start_session').disabled
    finally:
        dashboard.cached_json.clear()


def test_directory_and_years_include_an_unprepared_company(client):
    """Company search must include exchange issuers outside the current corpus."""
    companies = client.get('/companies').json()['companies']
    assert any(c['company_id'] == 'TCS' for c in companies)
    assert all(c['company_id'] != 'TCS' for c in client.get('/corpus').json()['documents'])
    reports = client.get('/companies/TCS/reports').json()['reports']
    assert reports[0]['period_end'] == '2025-03-31' and not reports[0]['prepared']
    assert reports[0]['period_dates_inferred']
    assert client.post('/prepare', json={'company_id': 'TCS', 'period_end': '2024-03-31'}).status_code == 422
    assert client.get('/preparations/unknown').status_code == 404


@pytest.fixture
def mocked_preparation(web_corpus, monkeypatch):
    """Simulate PDF extraction/indexing while exercising real background queue and publication logic."""
    calls = []

    def download(config, root):
        """Write a tiny exchange attachment and its real manifest shape to the isolated workspace."""
        source = json.loads(config.read_text())[0]
        content = b'%PDF-new-company-test'
        doc = digest(content)
        (root / 'raw').mkdir(exist_ok=True)
        (root / 'raw' / f'{doc}.pdf').write_bytes(content)
        write_jsonl(
            root / 'manifest.jsonl',
            [{**source, 'document_id': doc, 'content_hash': doc, 'local_path': f'raw/{doc}.pdf'}],
        )
        calls.append('download')

    def process(root):
        """Create one extracted page/chunk plus the document cache expected at publication."""
        row = read_jsonl(root / 'manifest.jsonl')[0]
        page = {
            'document_id': row['document_id'],
            'company_id': row['company_id'],
            'page': 1,
            'total_pages': 1,
            'flags': [],
            'text': 'TCS annual revenue increased.',
        }
        chunk = {
            **read_jsonl(web_corpus[0] / 'chunks.jsonl')[0],
            **row,
            'chunk_id': row['document_id'] + '-chunk',
            'text': page['text'],
            'source_url': row['attachment_url'],
        }
        for name, records in [('pages.jsonl', [page]), ('blocks.jsonl', []), ('chunks.jsonl', [chunk])]:
            write_jsonl(root / name, records)
        write_jsonl(root / 'processed' / row['document_id'] / 'pages.jsonl', [page])
        write_jsonl(root / 'processed' / row['document_id'] / 'blocks.jsonl', [])

    def build(root, embedding_cache=None):
        """Record shared-cache use; model/vector math is covered by the existing retrieval tests."""
        assert embedding_cache == web_corpus[0] / 'embedding_cache.sqlite'
        calls.append('index')

    monkeypatch.setattr(preparation_module, 'download', download)
    monkeypatch.setattr(preparation_module, 'process', process)
    monkeypatch.setattr(preparation_module, 'build_dense', build)
    return calls


def wait_for_preparation(client, job_id):
    """Wait briefly for the isolated worker, failing explicitly instead of hanging the test suite."""
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        job = client.get('/preparations/' + job_id).json()
        if job['status'] in ('ready', 'failed'):
            return job
        time.sleep(0.01)
    raise AssertionError('Preparation worker did not finish')


def test_preparation_publishes_new_report_and_reuses_ready_data(client, mocked_preparation, web_corpus):
    """Prepare a company absent from the corpus without losing existing report data."""
    old = {c['chunk_id'] for c in read_jsonl(web_corpus[0] / 'chunks.jsonl')}
    body = {'company_id': 'TCS', 'period_end': '2025-03-31'}
    response = client.post('/prepare', json=body)
    assert response.is_success
    result = wait_for_preparation(client, response.json()['job_id'])
    assert result['status'] == 'ready', result
    chunks = read_jsonl(web_corpus[0] / 'chunks.jsonl')
    assert old < {c['chunk_id'] for c in chunks}
    assert any(c['company_id'] == 'TCS' for c in client.get('/corpus').json()['documents'])
    assert client.post(
        '/search',
        json={
            'question': 'annual revenue',
            'mode': 'bm25',
            'company_ids': ['TCS'],
            'period_end': '2025-03-31',
        },
    ).json()['hits']
    assert client.post('/prepare', json=body).json()['status'] == 'ready'
    assert mocked_preparation == ['download', 'index']


def test_failed_preparation_preserves_existing_corpus(client, mocked_preparation, web_corpus, monkeypatch):
    """An extraction failure leaves the searchable published snapshot unchanged."""
    original = (web_corpus[0] / 'chunks.jsonl').read_bytes()

    def fail(root):
        """Simulate a report parser failure before snapshot publication."""
        raise ValueError('The selected report could not be extracted.')

    monkeypatch.setattr(preparation_module, 'process', fail)
    job = client.post('/prepare', json={'company_id': 'TCS', 'period_end': '2025-03-31'}).json()
    assert wait_for_preparation(client, job['job_id'])['status'] == 'failed'
    assert (web_corpus[0] / 'chunks.jsonl').read_bytes() == original
    assert client.get('/corpus').json()['chunks'] == 2


def test_streamlit_loads_a_company_outside_the_prepared_list(client, mocked_preparation, monkeypatch):
    """A searchable unprepared issuer/year loads in the background and opens its Q&A session."""

    def request(base, route, method='GET', body=None):
        """Bridge directory, preparation status and corpus refresh calls into the test API."""
        response = client.request(method, route, json=body)
        assert response.is_success, response.text
        return response.json()

    monkeypatch.setattr(dashboard, 'api_json', request)
    dashboard.cached_json.clear()
    try:
        app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / 'streamlit_app.py')).run(timeout=20)
        app.selectbox(key='company_choice').set_value('TCS').run(timeout=20)
        app.selectbox(key='year_choice').set_value('2025-03-31').run(timeout=20)
        assert not app.button(key='start_session').disabled
        app.button(key='start_session').click().run(timeout=20)
        deadline = time.monotonic() + 5
        while 'active_session' not in app.session_state or not app.session_state['active_session']:
            assert time.monotonic() < deadline
            time.sleep(0.01)
            app.run(timeout=20)
        assert not app.exception and app.chat_input
        assert app.session_state['active_session']['company_id'] == 'TCS'
        assert app.metric[0].value == '1'
    finally:
        dashboard.cached_json.clear()
