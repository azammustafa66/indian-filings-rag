"""Streamlit interface for searchable company/year selection and filing Q&A.

The main-page selector reads the exchange directory through FastAPI, loads missing
reports via background jobs and opens a company/year-scoped conversation. A
fragment polls preparation status every two seconds. Completed responses and
page previews live in Streamlit session state, so widget reruns do not regenerate
answers. All discovery, retrieval and model operations go through the backend.
"""

import os
import re
from datetime import date

import httpx
import pandas as pd
import streamlit as st

MODES = {'hybrid': 'Hybrid · keywords + meaning', 'bm25': 'Keyword · BM25', 'dense': 'Semantic · embeddings'}


class BackendError(Exception):
    """An actionable backend connectivity or request error suitable for the dashboard."""


def api_json(base: str, route: str, method: str = 'GET', body: dict | None = None) -> dict:
    """Call the local JSON API and translate transport/HTTP failures to BackendError.

    base is the configured API origin; route is an internal endpoint path. method
    and body support explicit question/preparation submissions. Requests allow a
    long read timeout for model work and a short connection timeout. This helper
    never reads a model credential; it communicates only with FastAPI.
    """
    try:
        response = httpx.request(method, base + route, json=body, timeout=httpx.Timeout(300, connect=5))
    except httpx.TimeoutException:
        raise BackendError('The request timed out. Please try again with fewer passages.') from None
    except httpx.TransportError:
        raise BackendError('Cannot reach the filings API. Start the backend and reconnect.') from None
    if response.is_error:
        try:
            detail = response.json().get('detail')
        except ValueError:
            detail = None
        if isinstance(detail, list):
            detail = '; '.join(item.get('msg', 'Invalid input') for item in detail)
        raise BackendError(
            detail if isinstance(detail, str) else 'The backend could not complete this request.'
        )
    return response.json()


@st.cache_data(ttl=30, show_spinner=False)
def cached_json(base: str, route: str, corpus_hash: str = '') -> dict:
    """Cache read-only metadata/report responses for 30 seconds.

    base/route identify the request; corpus_hash participates in the cache key to
    invalidate dependent views after publication. Writes and model submissions
    must call api_json directly, not this cached helper.
    """
    return api_json(base, route)


def financial_year_label(period_end: str) -> str:
    """Show March-year-end reports as FY YYYY–YY, otherwise retain the exact year end."""
    end = date.fromisoformat(period_end)
    if (end.month, end.day) == (3, 31):
        return f'FY {end.year - 1}–{end.year % 100:02d}'
    return f'Year ended {end:%d %b %Y}'


def activate_session(documents: list[dict], company: str, period: str):
    """Open a prepared issuer/year session and clear prior conversation/preview state.

    Find the matching document in the latest overview, prefer its exchange year
    label, and store a fixed active_session. Clear messages, preview and document
    widget choice. Missing prepared data raises StopIteration; callers activate
    only after an overview refresh confirms publication.
    """
    filing = next(d for d in documents if d['company_id'] == company and d['period_end'] == period)
    st.session_state['active_session'] = {
        'company_id': company,
        'company_name': filing['company_name'],
        'period_end': period,
        'year_label': filing.get('report_year_label') or financial_year_label(period),
    }
    st.session_state['messages'] = []
    st.session_state.pop('preview', None)
    st.session_state.pop('document', None)


def session_examples(session: dict) -> dict[str, str]:
    """Build starter questions for the selected company/report rather than other filings."""
    name, year = session['company_name'], session['year_label']
    return {
        'Financial performance': f'What are the key financial results for {name} in {year}? State units and reporting basis.',
        'Business risks': f'What key business risks does {name} describe in its {year} annual report?',
        'Year-on-year changes': f'How did {name} describe its financial performance changing from the previous year in its {year} report?',
    }


def conversation_history(messages: list[dict]) -> list[dict]:
    """Return at most six bounded prior messages for follow-up reference resolution.

    Include user text and successful generated answers, omit errors/passage-only
    responses, truncate each content string to 4,000 characters and preserve
    chronological order. This does not include retrieved chunks or credentials.
    """
    history = []
    for message in messages:
        if message['role'] == 'user':
            content = message['text']
        else:
            content = message.get('result', {}).get('answer', '')
        if content.strip():
            history.append({'role': message['role'], 'content': content[:4000]})
    return history[-6:]


@st.fragment(run_every=2)
def preparation_status(base: str):
    """Poll one pending preparation job every two seconds as a Streamlit fragment.

    Read its ID from session state. A ready result schedules activation, clears
    metadata caches and reruns the app; a failed result records the message,
    clears pending state and reruns so retry controls become available. Other
    states show phase text. No model work is triggered by polling itself.
    """
    job_id = st.session_state.get('preparation_job')
    if not job_id:
        return
    try:
        job = api_json(base, '/preparations/' + job_id)
    except BackendError as exc:
        st.error(str(exc))
        return
    if job['status'] == 'ready':
        st.session_state['prepared_session'] = {
            'company_id': job['company_id'],
            'period_end': job['period_end'],
        }
        st.session_state.pop('preparation_job', None)
        cached_json.clear()
        st.rerun()
    elif job['status'] == 'failed':
        st.session_state['preparation_error'] = job['stage']
        st.session_state.pop('preparation_job', None)
        st.rerun()
    else:
        st.info(job['company_name'] + ' · ' + job['stage'])
        st.caption('First-time loading can take a few minutes. This page updates automatically.')


def session_picker(base: str, overview: dict) -> dict | None:
    """Render exchange-wide type-to-search company and available-year controls.

    Args:
        base: Backend origin for directory, report and preparation requests.
        overview: Latest prepared-corpus metadata, used to reuse ready reports.

    Returns:
        The active company/year session dictionary, or None before activation.

    Controls remain on the main page when the sidebar is collapsed. Directory
    and year failures can fall back to previously prepared choices with a visible
    notice. Ready data opens immediately; other choices call /prepare and display
    its polling fragment. A newly ready job is activated on the full-app rerun.
    Pending selector changes do not silently replace an existing conversation.
    """
    documents = overview['documents']
    pending = st.session_state.pop('prepared_session', None)
    if pending:
        activate_session(documents, pending['company_id'], pending['period_end'])
    active = st.session_state.get('active_session')
    with st.container(border=True):
        st.markdown('### Find a company and load its annual report')
        try:
            directory = cached_json(base, '/companies')['companies']
        except BackendError as exc:
            st.warning(str(exc) + ' Showing previously prepared companies only.')
            directory = list({d['company_id']: d for d in documents}.values())
        names = {d['company_id']: d['company_name'] for d in directory}
        company_column, year_column, action_column = st.columns([2, 1.5, 1.5], vertical_alignment='bottom')
        company = company_column.selectbox(
            'Company — type a name or NSE symbol',
            options=list(names),
            index=None,
            placeholder='Search e.g. Reliance, TCS, Tata Motors…',
            format_func=lambda key: names[key] + ' (' + key + ')',
            key='company_choice',
        )
        reports = []
        if company:
            try:
                reports = cached_json(
                    base, f'/companies/{company}/reports', overview.get('corpus_hash') or ''
                )['reports']
            except BackendError as exc:
                st.warning(str(exc))
                reports = [d for d in documents if d['company_id'] == company]
        years = {r['period_end']: r for r in reports}
        periods = sorted(years, reverse=True)
        if st.session_state.get('year_choice') not in periods:
            st.session_state.pop('year_choice', None)
        period = year_column.selectbox(
            'Financial year',
            periods,
            index=None,
            placeholder='Choose an available year',
            format_func=lambda key: years[key].get('report_year_label') or financial_year_label(key),
            disabled=not periods,
            key='year_choice',
        )
        prepared = any(
            d['company_id'] == company and d['period_end'] == period and d.get('chunks', 0) for d in documents
        )
        loading = bool(st.session_state.get('preparation_job'))
        if action_column.button(
            'Start Q&A session' if prepared else 'Load report & start Q&A',
            disabled=not company or not period or loading,
            type='primary',
            width='stretch',
            key='start_session',
        ):
            st.session_state.pop('preparation_error', None)
            if prepared:
                activate_session(documents, company, period)
                active = st.session_state['active_session']
            else:
                try:
                    job = api_json(base, '/prepare', 'POST', {'company_id': company, 'period_end': period})
                    st.session_state['preparation_job'] = job['job_id']
                except BackendError as exc:
                    st.error(str(exc))
        if company and not reports:
            st.info('NSE has no available annual-report attachments for this company right now.')
        if st.session_state.get('preparation_error'):
            st.error(st.session_state['preparation_error'])
            st.caption('Previously prepared reports are unchanged. Load this report again to retry.')
        preparation_status(base)
        if active:
            st.success(f'Active session: {active["company_name"]} · {active["year_label"]}')
        st.caption(
            'Search NSE main-board and SME companies. Available years come from the exchange. '
            'Reports are downloaded, parsed and embedded once, then reused. BSE-only issuers are not included.'
        )
    return active


def sidebar(overview: dict, active: dict | None) -> dict:
    """Return retrieval options combined with the fixed active session scope.

    Only optional search/model controls live in the sidebar. Without a compatible
    dense index/provider, offer BM25 only. Clear conversation resets messages but
    does not change the selected company/year. This performs no model requests.
    """
    with st.sidebar:
        st.markdown('## FILINGS / INDIA')
        st.caption('Search settings')
        with st.expander('Search options'):
            modes = list(MODES) if overview['dense_ready'] and overview['embedding_configured'] else ['bm25']
            mode = st.selectbox('Retrieval method', modes, format_func=lambda key: MODES[key], key='mode')
            if len(modes) == 1:
                st.caption('Semantic search needs a configured embedding model and a current index.')
            generate = st.toggle(
                'Generate a cited answer',
                value=overview['answer_configured'],
                disabled=not overview['answer_configured'],
                key='generate',
            )
            if not overview['answer_configured']:
                st.caption('Answer generation needs a configured backend model.')
            k = st.slider('Passages to retrieve', min_value=3, max_value=20, value=10, key='top_k')
            cutoff = st.date_input(
                'Only filings published by',
                value=None,
                key='as_of',
                help='Unknown publication dates are excluded when a cutoff is selected.',
            )
            st.caption('The pilot has no verified publication timestamps. A cutoff may return no evidence.')
        st.divider()
        st.caption('ANSWER MODEL')
        st.write(overview['answer_model'] or 'Not configured')
        st.caption('EMBEDDING MODEL')
        st.write(overview['embedding_model'] or 'Not configured')
        if st.button('Clear conversation', width='stretch'):
            st.session_state['messages'] = []
        st.caption('Connected to local FastAPI')
    return {
        'session': active,
        'company_ids': [active['company_id']] if active else [],
        'period_end': active['period_end'] if active else None,
        'mode': mode,
        'k': k,
        'as_of': cutoff.isoformat() if cutoff else None,
        'generate': generate,
    }


def evidence_card(hit: dict, number: int, cited: bool = False, expanded: bool = False):
    """Show one passage with company, page, rank and an exchange PDF link."""
    chunk = hit['chunk']
    label = f'[{number}] {chunk["company_name"]} · PDF page {chunk["page"]}'
    if cited:
        label += ' · cited'
    with st.expander(label, expanded=expanded):
        kind = (chunk.get('block_type') or 'text').title()
        label = chunk.get('report_year_label') or f'Period ending {chunk["period_end"]}'
        st.caption(f'{label} · {kind} · Retrieval rank {hit["rank"]}')
        if chunk.get('period_dates_inferred'):
            st.caption('Report year follows the exchange label; use the report for exact fiscal dates.')
        st.text(chunk['text'])
        url = chunk['source_url']
        st.link_button(
            'Open exchange attachment', url + (f'#page={chunk["page"]}' if '.pdf' in url.lower() else '')
        )
        if chunk.get('flags'):
            st.caption('Extraction review flags: ' + ', '.join(chunk['flags']))


def render_reply(result: dict):
    """Render a saved answer or search result with correctly numbered context passages."""
    hits = result.get('hits', [])
    is_answer = 'answer' in result
    if is_answer:
        st.markdown(result['answer'])
    elif hits:
        st.write(f'Found {len(hits)} relevant passages.')
    else:
        st.info('No evidence matched these filters. Try another company, period or publication cutoff.')
    st.caption(f'{MODES.get(result["mode"], result["mode"])} · {result.get("elapsed_ms", 0) / 1000:.1f}s')
    if not hits:
        return
    cited = {int(value) for value in re.findall(r'\[(\d+)\]', result.get('answer', ''))}
    if is_answer:
        st.caption(
            'Sources supplied to the answer model. Citation numbers are checked; claim support needs review.'
        )
        by_id = {hit['chunk']['chunk_id']: hit for hit in hits}
        for source in result.get('sources', []):
            hit = by_id.get(source['chunk_id'])
            if hit:
                evidence_card(hit, source['citation'], source['citation'] in cited)
    else:
        for number, hit in enumerate(hits, 1):
            evidence_card(hit, number, expanded=number == 1)


def question_view(base: str, settings: dict):
    """Render prior responses and submit one new question with the active filing scope.

    Before activation, display selection guidance rather than an enabled chat.
    Generated answers include bounded prior history; passage-only searches do not.
    Save the user's question and completed response/error in session state. Widget
    reruns redisplay these saved results instead of repeating an answer request.
    """
    session = settings['session']
    if not session:
        st.subheader('Choose a filing to begin')
        st.info('Select a company and financial year above, then click Start Q&A session.')
        return
    st.subheader(f'{session["company_name"]} · {session["year_label"]}')
    st.caption(
        'Answers use this selected report. Follow-up questions can refer to the conversation.'
        if settings['generate']
        else 'Passage-only search treats each question independently.'
    )
    messages = st.session_state.setdefault('messages', [])
    pending = None
    if not messages:
        st.write('Start with a question about this report:')
        for column, (label, question) in zip(st.columns(3), session_examples(session).items(), strict=True):
            if column.button(label, width='stretch', disabled=not settings['company_ids']):
                pending = question
    for message in messages:
        with st.chat_message(message['role']):
            if message['role'] == 'user':
                st.write(message['text'])
            elif 'error' in message:
                st.error(message['error'])
            else:
                st.caption(message['scope'])
                render_reply(message['result'])
    if not settings['company_ids']:
        st.info('Select at least one company in the sidebar to ask a question.')
    question = st.chat_input(
        'Ask about revenue, margins, risks or business performance…',
        disabled=not settings['company_ids'],
        key='question_input',
    )
    question = question or pending
    if question:
        history = conversation_history(messages) if settings['generate'] else []
        messages.append({'role': 'user', 'text': question})
        with st.chat_message('user'):
            st.write(question)
        with st.chat_message('assistant'):
            try:
                body = {key: settings[key] for key in ('company_ids', 'period_end', 'mode', 'k', 'as_of')}
                body['history'] = history
                body['question'] = question
                route = '/ask' if settings['generate'] else '/search'
                with st.spinner(
                    'Searching filings and preparing your answer…'
                    if settings['generate']
                    else 'Finding relevant passages…'
                ):
                    result = api_json(base, route, 'POST', body)
                scope = session['company_name'] + ' · ' + session['year_label']
                if settings['as_of']:
                    scope += ' · Published by ' + settings['as_of']
                messages.append({'role': 'assistant', 'result': result, 'scope': scope})
                st.caption(scope)
                render_reply(result)
            except BackendError as exc:
                messages.append({'role': 'assistant', 'error': str(exc)})
                st.error(str(exc))


def document_view(base: str, overview: dict, session: dict | None = None):
    """Browse local page text/images for the active session, or all prepared reports.

    A page form explicitly requests a preview; image rendering is local backend
    PDF work, not a vision API call. Keep the loaded preview in session state,
    label it with its actual document/page and show extraction review flags.
    """
    st.subheader('Browse the original evidence')
    docs = {
        doc['document_id']: doc
        for doc in overview['documents']
        if not session
        or (doc['company_id'] == session['company_id'] and doc['period_end'] == session['period_end'])
    }
    selected = st.selectbox(
        'Filing',
        list(docs),
        key='document',
        format_func=lambda key: f'{docs[key]["company_name"]} · Annual report · {docs[key]["period_end"]}',
    )
    doc = docs[selected]
    st.caption(f'{doc["pages"]} extracted PDF pages · Includes the filing cover letter')
    st.link_button('Open original exchange filing', doc['source_url'])
    if not doc['pages']:
        st.info('This filing has no extracted pages yet.')
        return
    with st.form('page_browser'):
        left, right = st.columns([1, 2])
        number = left.number_input('PDF page', min_value=1, max_value=max(1, doc['total_pages']), value=1)
        view = right.radio('View', ['Page image', 'Extracted text'], horizontal=True)
        submitted = st.form_submit_button('View page', type='primary')
    if submitted:
        try:
            route = f'/documents/{selected}/pages/{number}'
            page = api_json(base, route)
            preview = {'page': page, 'title': f'{doc["company_name"]} · PDF page {number}', 'view': view}
            if view == 'Page image':
                response = httpx.get(base + route + '/image', timeout=60)
                if response.is_error:
                    raise BackendError('This page could not be rendered. Try extracted text instead.')
                preview['image'] = response.content
            st.session_state['preview'] = preview
        except (BackendError, httpx.HTTPError) as exc:
            st.error(str(exc) if isinstance(exc, BackendError) else 'Could not load the source-page image.')
    preview = st.session_state.get('preview')
    if preview:
        st.markdown(f'#### {preview["title"]}')
        if preview['view'] == 'Page image':
            st.image(preview['image'], width='stretch')
        else:
            st.text(preview['page']['text'])
        if preview['page']['flags']:
            st.caption('Review flags: ' + ', '.join(preview['page']['flags']))


def benchmark_view(base: str, overview: dict):
    """Chart saved retrieval reports without running evaluation or calling a model.

    Mark corpus-hash mismatches as stale. Show question counts when present,
    recall/complete-coverage/MRR and the development-set limitations. These metrics
    measure evidence retrieval rather than generated-answer accuracy.
    """
    st.subheader('How well does retrieval find the evidence?')
    runs = cached_json(base, '/benchmarks', overview['corpus_hash'])['runs']
    if not runs:
        st.info('No evaluation results yet. Run the benchmark from the CLI to populate this view.')
        return
    selected = st.selectbox(
        'Evaluation run',
        range(len(runs)),
        format_func=lambda index: runs[index]['name'].replace('-', ' ').title(),
    )
    run = runs[selected]
    if not run['matches_corpus']:
        st.warning('This saved run used a different corpus. Its scores do not describe the current filings.')
    counts = next(iter(run['report'].get('results', {}).values()), {})
    if counts.get('queries') is not None:
        st.caption(f'{counts["queries"]} questions · {counts.get("answerable_queries", 0)} answerable')
    rows = []
    for mode, values in run['report'].get('results', {}).items():
        metrics = values.get('metrics', {})
        rows.append(
            {
                'Method': mode.upper(),
                'Evidence recall @10': metrics.get('evidence_recall@10', 0),
                'Complete evidence @10': metrics.get('complete@10', 0),
                'MRR @10': metrics.get('mrr@10', 0),
            }
        )
    if not rows:
        st.info('This report has no scored answerable questions.')
        return
    frame = pd.DataFrame(rows).set_index('Method')
    st.bar_chart(frame[['Evidence recall @10', 'Complete evidence @10']], color=['#2DD4BF', '#94A3B8'])
    st.dataframe(
        frame.style.format(
            {'Evidence recall @10': '{:.1%}', 'Complete evidence @10': '{:.1%}', 'MRR @10': '{:.3f}'}
        ),
        width='stretch',
    )
    st.caption(
        'Recall measures required evidence found. Complete evidence requires every evidence group. '
        'MRR rewards an earlier first relevant result.'
    )
    st.info(
        'Pilot development labels are source-checked and concentrated on a few pages. '
        'These scores measure retrieval, not answer accuracy or held-out test performance.'
    )


def main():
    """Assemble the dashboard, including operation from an initially empty corpus.

    Configure the page, read FastAPI metadata, render directory/year loading before
    corpus-dependent views, and attach the session/search controls and three tabs.
    If the backend is offline, show launch/reconnect guidance. No credentials are
    loaded by this frontend; API/model operations belong to the backend.
    """
    st.set_page_config(
        page_title='Indian Filings · Research workspace',
        page_icon='📑',
        layout='wide',
        initial_sidebar_state='collapsed',
    )
    st.markdown(
        """<style>
        .block-container {max-width: 1220px; padding-top: 2rem;}
        h1 {letter-spacing: -0.035em;}
        [data-testid="stMetric"] {background: #151E30; border: 1px solid #2A3850;
            border-radius: 12px; padding: 16px 20px;}
        [data-testid="stChatMessage"] {background: #151E30; border: 1px solid #2A3850; border-radius: 12px;}
        [data-testid="stSidebar"] {border-right: 1px solid #2A3850;}
        </style>""",
        unsafe_allow_html=True,
    )
    base = os.getenv('FILINGS_API_URL', 'http://127.0.0.1:8000').rstrip('/')
    try:
        overview = cached_json(base, '/corpus')
    except BackendError as exc:
        st.title('Indian company filings')
        st.error(str(exc))
        st.write('Start both services from the project folder:')
        st.code('uv run python scripts/run_dashboard.py', language='bash')
        if st.button('Reconnect'):
            cached_json.clear()
            st.rerun()
        return
    st.caption('RESEARCH WORKSPACE / INDIAN LISTED COMPANIES')
    st.title('Explore company filings')
    st.write('Select a company and financial year, then ask questions about that report.')
    session = session_picker(base, overview)
    settings = sidebar(overview, session)
    selected_docs = [
        d
        for d in overview['documents']
        if not session
        or (d['company_id'] == session['company_id'] and d['period_end'] == session['period_end'])
    ]
    columns = st.columns(3)
    columns[0].metric('Selected reports' if session else 'Available reports', len(selected_docs))
    columns[1].metric('PDF pages', f'{sum(d["pages"] for d in selected_docs):,}')
    columns[2].metric('Searchable passages', f'{sum(d.get("chunks", 0) for d in selected_docs):,}')
    if not overview['documents'] or not overview['chunks']:
        st.info('Choose a listed company and year above to prepare your first annual report.')
        return
    questions, documents, benchmarks = st.tabs(['Ask filings', 'Browse documents', 'Benchmark results'])
    with questions:
        question_view(base, settings)
    with documents:
        document_view(base, overview, session)
    with benchmarks:
        try:
            benchmark_view(base, overview)
        except BackendError as exc:
            st.error(str(exc))
