"""Evaluate retrieval against source quotes grouped by required evidence.

A group contains alternative acceptable passages; comparison questions can
require multiple groups. This evaluates retrieval, not generated-answer accuracy
or abstention. The novelty-aware DCG variant is not standard graded nDCG.
"""

import json
import math
import re
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from .retrieval import Retriever
from .storage import digest, fingerprint, read_jsonl, write_jsonl


def normalize(text: str) -> str:
    # Table serialization inserts cell separators and wraps cell text with slashes.
    """Normalize case, whitespace and table serialization separators for quote matching.

    This is text normalization, not financial-number, currency or unit conversion.
    """
    return re.sub(r'[\s|]+', ' ', text.replace(' / ', ' ')).strip().casefold()


def matches(chunk: dict, evidence: dict) -> bool:
    """Check that a chunk contains the gold quote on the same document and PDF page."""
    return (
        chunk['document_id'] == evidence['document_id']
        and chunk['page'] == evidence['page']
        and normalize(evidence['quote']) in normalize(chunk['text'])
    )


def score(hits: list[dict], groups: list[list[dict]], k: int) -> dict:
    """Score the first k hits against all required evidence groups for one question.

    Each group contains alternative acceptable source passages. Count a group
    once even when overlapping chunks repeat its quote. Return evidence-group
    recall, complete-coverage indicator, first-hit reciprocal rank and a
    novelty-aware DCG variant. Raise ValueError for missing/empty groups.
    The DCG variant is not standard graded nDCG.
    """
    if not groups or any(not group for group in groups):
        raise ValueError('Answerable queries require nonempty gold evidence groups')
    covered = set()
    first, dcg = None, 0.0
    for rank, hit in enumerate(hits[:k], 1):
        found = {i for i, group in enumerate(groups) if any(matches(hit['chunk'], e) for e in group)}
        new = found - covered
        if found and first is None:
            first = rank
        if new:
            # Binary gain for new evidence; repeated overlapping chunks get no extra credit.
            dcg += 1 / math.log2(rank + 1)
        covered.update(found)
    # This is a novelty-aware measure, not standard graded chunk nDCG.
    ideal = sum(1 / math.log2(rank + 1) for rank in range(1, min(len(groups), k) + 1))
    return {
        f'evidence_recall@{k}': len(covered) / len(groups),
        f'complete@{k}': float(len(covered) == len(groups)),
        f'mrr@{k}': 1 / first if first else 0.0,
        f'novelty_dcg@{k}': dcg / ideal if ideal else 0.0,
    }


def validate_queries(queries: list[dict], root: Path) -> None:
    """Reject duplicate query IDs, unverified labels and invalid source evidence.

    Read root/pages.jsonl and confirm each nonblank gold quote occurs on its
    annotated page. Answerable queries need evidence groups; unanswerable ones
    must have none. Raise ValueError on failure; return None on success. A
    verified status here does not establish independent human annotation.
    """
    pages = {(p['document_id'], p['page']): p for p in read_jsonl(root / 'pages.jsonl')}
    ids = set()
    for query in queries:
        if query['query_id'] in ids:
            raise ValueError('Duplicate query ID')
        ids.add(query['query_id'])
        if query.get('review_status') != 'verified':
            raise ValueError(f'{query["query_id"]}: benchmark evidence has not been verified')
        groups = query.get('evidence_groups', [])
        if query['answerable'] and (not groups or any(not group for group in groups)):
            raise ValueError('Answerable query missing evidence')
        if not query['answerable'] and groups:
            raise ValueError('Unanswerable query has gold evidence')
        for group in groups:
            for evidence in group:
                page = pages.get((evidence['document_id'], evidence['page']))
                if not page or not normalize(evidence['quote']):
                    raise ValueError('Gold page or quote missing')
                if normalize(evidence['quote']) not in normalize(page['text']):
                    raise ValueError(f'Gold quote not found on page: {query["query_id"]}')


def evaluate(root: Path, benchmark: Path, output: Path, modes: list[str], split='dev') -> dict:
    """Run retrieval modes on a benchmark split and save metrics plus ranked outputs.

    Args:
        root: Processed corpus/index directory.
        benchmark: JSONL containing questions, filters and evidence groups.
        output: Directory for report.json and retrieval_runs.jsonl.
        modes: Retrieval modes to compare using the same questions and corpus.
        split: Benchmark partition, normally "dev" for this pilot.

    Validate gold evidence, retrieve 20 hits per question, and summarize scores
    at 5/10/20 overall and by category. Log unanswerable queries without including
    them in recall averages. Return the report with corpus/benchmark hashes,
    provider identity and retrieval settings. Dense/hybrid runs can call the
    embedding API for uncached questions; no answer generation is evaluated.
    Latency includes cache and first-load effects, so it is not a controlled
    serving comparison.
    """
    queries = [q for q in read_jsonl(benchmark) if q['split'] == split]
    if not queries:
        raise ValueError(f'No {split} benchmark questions')
    validate_queries(queries, root)
    retriever = Retriever(root)
    rows, summary = [], {}
    for mode in modes:
        mode_rows = []
        for query in queries:
            start = time.perf_counter()
            hits = retriever.search(
                query['question'],
                mode=mode,
                k=20,
                companies=query.get('company_ids'),
                period_end=query.get('period_end'),
                as_of=query.get('as_of'),
            )
            latency = (time.perf_counter() - start) * 1000
            metrics = {}
            if query['answerable']:
                for k in (5, 10, 20):
                    metrics.update(score(hits, query['evidence_groups'], k))
            row = {
                'query_id': query['query_id'],
                'category': query['category'],
                'mode': mode,
                'answerable': query['answerable'],
                'latency_ms': latency,
                'metrics': metrics,
                'hits': [
                    {
                        'chunk_id': h['chunk']['chunk_id'],
                        'document_id': h['chunk']['document_id'],
                        'page': h['chunk']['page'],
                        'rank': h['rank'],
                        'score': h['score'],
                    }
                    for h in hits
                ],
            }
            rows.append(row)
            mode_rows.append(row)
        scored = [r for r in mode_rows if r['metrics']]
        summary[mode] = {
            'queries': len(mode_rows),
            'answerable_queries': len(scored),
            'unanswerable_queries': len(mode_rows) - len(scored),
            'metrics': {
                key: float(np.mean([r['metrics'][key] for r in scored])) for key in scored[0]['metrics']
            }
            if scored
            else {},
            'by_category': {
                category: {
                    key: float(np.mean([r['metrics'][key] for r in scored if r['category'] == category]))
                    for key in scored[0]['metrics']
                }
                for category in sorted({r['category'] for r in scored})
            }
            if scored
            else {},
            'latency_ms_p50': float(np.median([r['latency_ms'] for r in mode_rows])),
            'latency_ms_p95': float(np.percentile([r['latency_ms'] for r in mode_rows], 95)),
        }
    report = {
        'created_at': datetime.now(UTC).isoformat(),
        'split': split,
        'corpus_hash': retriever.corpus_hash,
        'benchmark_hash': fingerprint(queries),
        'benchmark_file_hash': digest(benchmark.read_bytes()),
        'embedding_provider': retriever.provider.identity if any(m != 'bm25' for m in modes) else None,
        'retrieval_config': {'candidate_k': 50, 'rrf_constant': 60, 'bm25_k1': 1.5, 'bm25_b': 0.75},
        'note': 'Evidence-group recall; novelty DCG is not standard nDCG. '
        'No answer-generation or abstention accuracy is measured. Latency includes embedding '
        'cache effects and first-query vector loading; not a serving benchmark.',
        'results': summary,
    }
    output.mkdir(parents=True, exist_ok=True)
    write_jsonl(output / 'retrieval_runs.jsonl', rows)
    (output / 'report.json').write_text(json.dumps(report, indent=2))
    return report
