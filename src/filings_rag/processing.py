"""Turn varied PDF layouts into page records, evidence blocks and search chunks.

Original files and raw layout text remain available alongside conservative
column extraction, detected tables and optional reviewed vision repairs. Quality
flags identify selected extraction problems, not complete correctness.
"""

import re
from pathlib import Path

import pdfplumber

from .storage import digest, fingerprint, read_jsonl, write_jsonl

PARSER_VERSION = 'pdfplumber-layout-v2'


def text_regions(page, tables: list) -> list[dict]:
    """Extract one full-page text region or two regions separated by a central gutter.

    Use word positions to find a clear gap with substantial text on both sides.
    Skip split positions crossing a detected wide table. Return text blocks with
    PDF bounding boxes in left-to-right order; complex layouts can still fail.
    """
    words = page.extract_words()
    body = [w for w in words if page.height * 0.08 < w['top'] < page.height * 0.92]
    split = None
    margin = page.width * 0.007
    candidates = []
    for step in range(45, 56):
        x = page.width * step / 100
        if any(t.bbox[0] < x < t.bbox[2] and t.bbox[2] - t.bbox[0] > page.width * 0.55 for t in tables):
            continue
        crossing = sum(w['x0'] < x + margin and w['x1'] > x - margin for w in body)
        left = sum(w['x1'] <= x - margin for w in body)
        right = sum(w['x0'] >= x + margin for w in body)
        if crossing == 0 and min(left, right) >= 35:
            candidates.append((abs(step - 50), x))
    if candidates:
        split = min(candidates)[1]
    bounds = (
        [page.bbox] if split is None else [(0, 0, split, page.height), (split, 0, page.width, page.height)]
    )
    return [
        {
            'block_type': 'text',
            'bbox': list(box),
            'text': (page if split is None else page.crop(box)).extract_text(layout=True) or '',
            'table_rows': None,
        }
        for box in bounds
    ]


def quality_flags(text: str, table_errors: int = 0) -> list[str]:
    """Return heuristic labels for low text, suspect glyphs or failed table extraction.

    An empty result means these particular checks passed, not that the page is
    fully correct. table_errors reports a failure caught by the processing loop.
    """
    flags = []
    visible = [c for c in text if not c.isspace()]
    if len(visible) < 60:
        flags.append('low_text_or_image_page')
    if visible and sum(c == '\ufffd' or ord(c) < 32 for c in visible) / len(visible) > 0.01:
        flags.append('encoding_suspect')
    if '(cid:' in text:
        flags.append('unmapped_glyphs')
    if table_errors:
        flags.append('table_extraction_failed')
    return flags


def segments(text: str, max_chars: int = 2200, overlap_lines: int = 2) -> list[str]:
    """Split text into character-bounded chunks with optional overlapping lines.

    Normalize whitespace within lines, omit empty lines and split oversized
    lines. Keep up to overlap_lines preceding lines when the size budget allows.
    The limit is in characters, not model tokens; raw layout text is kept elsewhere.
    """
    lines = []
    for line in text.splitlines():
        # The immutable page text retains layout spacing; search text should not
        # spend its context budget on PDF indentation and trailing padding.
        line = ' '.join(line.split())
        if line.strip():
            lines.extend(line[i : i + max_chars] for i in range(0, len(line), max_chars))
    result, current = [], []
    for line in lines:
        if current and len('\n'.join([*current, line])) > max_chars:
            result.append('\n'.join(current))
            current = current[-overlap_lines:] if overlap_lines else []
            while current and len('\n'.join([*current, line])) > max_chars:
                current.pop(0)
        current.append(line)
    if current:
        result.append('\n'.join(current))
    return result


def table_segments(rows: list[list], max_chars: int = 3500) -> list[str]:
    # Repeat the first rows as context, without claiming they are confirmed table headers.
    """Serialize table rows into bounded chunks with repeated initial-row context.

    Rows contain cell values; None becomes an empty cell and cell line breaks
    become separators. The first two rows are repeated as header candidates, not
    verified semantic headers. Return an empty list for an empty table.
    """
    lines = [' | '.join(str(c or '').replace('\n', ' / ') for c in row) for row in rows]
    if not lines:
        return []
    prefix = 'First table rows (header candidates):\n' + '\n'.join(lines[:2])[:1000]
    body = segments('\n'.join(lines), max_chars=max_chars - len(prefix) - 1, overlap_lines=1)
    return [prefix + '\n' + item for item in body]


def process(root: Path, max_pages: int | None = None, repairs_path: Path | None = None) -> None:
    """Extract downloaded PDFs and rebuild the active page, block and chunk files.

    Args:
        root: Data directory containing manifest.jsonl and source PDFs.
        max_pages: Optional prefix length per report for a partial smoke run.
        repairs_path: Optional JSONL of reviewed page transcriptions.

    Reuse per-document extraction caches when parser settings and repair hashes
    match. Verify source hashes, combine duplicate source records, skip superseded
    filings and generate page-specific chunks with provenance. Write pages.jsonl,
    blocks.jsonl, chunks.jsonl and quality_issues.jsonl; return None. Existing active
    outputs are replaced, so a partial run also produces a partial search corpus.

    Raises:
        ValueError: Sources are missing/inconsistent, hashes differ, or supplied
            repairs are empty, unreviewed or duplicate a document/page pair.
    """
    manifest = read_jsonl(root / 'manifest.jsonl')
    if not manifest:
        raise ValueError('No downloaded filings. Run download first.')
    repairs = {}
    for repair in read_jsonl(repairs_path) if repairs_path else []:
        if repair.get('reviewed') is not True or not repair.get('text', '').strip():
            raise ValueError(
                'Review vision proposals against the source and set reviewed=true before processing'
            )
        key = (repair['document_id'], repair['page'])
        if key in repairs:
            raise ValueError('Duplicate repair for document/page')
        repairs[key] = repair
    records = {}
    for filing in manifest:
        if filing.get('superseded'):
            continue
        doc = filing['document_id']
        if doc in records:
            if records[doc]['company_id'] != filing['company_id']:
                raise ValueError('Identical bytes assigned to different companies; inspect manifest')
            records[doc]['sources'].append(filing)
        else:
            records[doc] = {**filing, 'sources': [filing]}
    all_pages, chunks, blocks = [], [], []
    for doc, record in records.items():
        if digest((root / record['local_path']).read_bytes()) != doc:
            raise ValueError(f'Source PDF hash mismatch: {record["company_id"]}')
        cache = root / 'processed' / doc
        cached = read_jsonl(cache / 'pages.jsonl')
        expected_settings = {'parser_version': PARSER_VERSION, 'max_pages': max_pages}
        doc_repairs = [r for (d, _), r in repairs.items() if d == doc]
        if doc_repairs:
            expected_settings['repairs_hash'] = fingerprint(doc_repairs)
        from_cache = (
            bool(cached)
            and (cache / 'blocks.jsonl').exists()
            and all(c.get('settings') == expected_settings for c in cached)
        )
        if from_cache:
            pages = cached
            doc_blocks = read_jsonl(cache / 'blocks.jsonl')
        else:
            pages, doc_blocks = [], []
            with pdfplumber.open(root / record['local_path']) as pdf:
                total_pages = len(pdf.pages)
                for index, page in enumerate(pdf.pages):
                    if max_pages is not None and index >= max_pages:
                        break
                    repair = repairs.get((doc, index + 1))
                    table_errors = 0
                    tables = []
                    try:
                        tables = [] if repair else page.find_tables()
                    except (ValueError, IndexError, TypeError):
                        table_errors = 1
                    raw_text = page.extract_text(layout=True) or ''
                    page_blocks = (
                        [
                            {
                                'block_type': 'text',
                                'bbox': list(page.bbox),
                                'text': repair['text'],
                                'table_rows': None,
                            }
                        ]
                        if repair
                        else text_regions(page, tables)
                    )
                    text = '\n\n'.join(b['text'] for b in page_blocks)
                    text_region_count = len(page_blocks)
                    try:
                        for table in tables:
                            rows = table.extract()
                            if len(rows) >= 2 and max(map(len, rows), default=0) >= 2:
                                page_blocks.append(
                                    {
                                        'block_type': 'table',
                                        'bbox': list(table.bbox),
                                        'text': '\n'.join(' | '.join(c or '' for c in r) for r in rows),
                                        'table_rows': rows,
                                    }
                                )
                    except (ValueError, IndexError, TypeError):
                        table_errors = 1
                    flags = quality_flags(text, table_errors)
                    pages.append(
                        {
                            'document_id': doc,
                            'company_id': record['company_id'],
                            'page': index + 1,
                            'total_pages': total_pages,
                            'text': text,
                            'raw_layout_text': raw_text,
                            'text_regions': text_region_count,
                            'flags': flags,
                            'settings': expected_settings,
                            'width': page.width,
                            'height': page.height,
                            'table_count': len(page_blocks) - text_region_count,
                        }
                    )
                    for position, block in enumerate(page_blocks):
                        block['block_id'] = digest(f'{PARSER_VERSION}:{doc}:{index + 1}:{position}')
                        doc_blocks.append(
                            {
                                **block,
                                'document_id': doc,
                                'page': index + 1,
                                'flags': flags,
                                'extraction_method': 'vision_api_reviewed' if repair else 'pdfplumber',
                            }
                        )
                    page.close()
                    if (index + 1) % 25 == 0:
                        print(f'{record["company_id"]}: {index + 1}/{total_pages} pages', flush=True)
            write_jsonl(cache / 'pages.jsonl', pages)
            write_jsonl(cache / 'blocks.jsonl', doc_blocks)
        all_pages.extend(pages)
        blocks.extend(doc_blocks)
        for block in doc_blocks:
            pieces = (
                table_segments(block['table_rows'])
                if block['block_type'] == 'table'
                else segments(block['text'])
            )
            for position, text in enumerate(pieces):
                if len(re.sub(r'\s', '', text)) < 30:
                    continue
                chunks.append(
                    {
                        'chunk_id': digest(f'{block["block_id"]}:{position}:{text}'),
                        'document_id': doc,
                        'block_id': block['block_id'],
                        'page': block['page'],
                        'company_id': record['company_id'],
                        'company_name': record['company_name'],
                        'filing_type': record['filing_type'],
                        'period_start': record['period_start'],
                        'period_end': record['period_end'],
                        **(
                            {
                                key: record[key]
                                for key in ('report_year_label', 'period_dates_inferred')
                                if key in record
                            }
                        ),
                        'published_at': record.get('published_at'),
                        'source_url': record['attachment_url'],
                        'sources': record['sources'],
                        'text': text,
                        'block_type': block['block_type'],
                        'bbox': block['bbox'],
                        'flags': block['flags'],
                        'extraction_method': block['extraction_method'],
                        # Page text is retained; no unverified section/basis/unit classification.
                        'section': None,
                        'statement_basis': None,
                        'unit': None,
                    }
                )
        print(f'processed {record["company_id"]}: {len(pages)} pages', flush=True)
    write_jsonl(root / 'pages.jsonl', all_pages)
    write_jsonl(root / 'blocks.jsonl', blocks)
    write_jsonl(root / 'chunks.jsonl', chunks)
    write_jsonl(
        root / 'quality_issues.jsonl',
        [{k: p[k] for k in ('document_id', 'company_id', 'page', 'flags')} for p in all_pages if p['flags']],
    )
    print(f'Saved {len(all_pages)} pages, {len(chunks)} chunks')
