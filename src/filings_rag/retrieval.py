"""Rank evidence chunks with BM25, dense cosine similarity or reciprocal rank fusion.

BM25 postings/statistics are built locally in memory. Dense document vectors and
compatibility metadata are persisted; query vectors use the same provider/model
and shared text cache. API calls are only needed for uncached embeddings. Hybrid
fusion uses rank positions, not direct addition of incompatible score scales.
"""

import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from .providers import Provider
from .storage import fingerprint, read_jsonl


def tokenize(text: str) -> list[str]:
    # Keep Indian digit groupings equivalent to ungrouped numeric queries.
    """Lowercase text and extract word/decimal tokens for BM25.

    Remove digit-grouping commas so 1,62,990 and 162990 match the same token.
    """
    return re.findall(r'\w+(?:\.\d+)?', re.sub(r'(?<=\d),(?=\d)', '', text.casefold()))


def embedding_text(chunk: dict) -> str:
    """Build the shared BM25/document-embedding representation for a chunk.

    Prefix company name and its exchange report-year label when available;
    otherwise use the stored period end. Do not present inferred fiscal boundaries
    as verified dates. Identical text/provider inputs can reuse cached embeddings.
    """
    if chunk.get('report_year_label'):
        return f'{chunk["company_name"]} | exchange report {chunk["report_year_label"]}\n{chunk["text"]}'
    return f'{chunk["company_name"]} | period ending {chunk["period_end"]}\n{chunk["text"]}'


def build_dense(root: Path, embedding_cache: Path | None = None) -> dict:
    """Embed the active chunk corpus and persist vectors plus compatibility metadata.

    Args:
        root: Directory containing chunks.jsonl and receiving index/ output files.
        embedding_cache: Optional shared SQLite cache, useful for a staged corpus.
            Defaults to root/embedding_cache.sqlite.

    Returns:
        A summary with chunk count, vector dimensions and nonsecret provider identity.

    Reuse existing endpoint/model/text embeddings; request only missing vectors.
    Write normalized vectors.npy and metadata.json with corpus hash and chunk order.
    Raise ValueError for an empty corpus and propagate provider/filesystem failures.
    The caller coordinates publication if root is a temporary snapshot.
    """
    chunks = read_jsonl(root / 'chunks.jsonl')
    if not chunks:
        raise ValueError('No chunks. Run process first.')
    provider = Provider('embedding', embedding_cache or root / 'embedding_cache.sqlite')
    vectors = provider.embed([embedding_text(c) for c in chunks], progress=True)
    metadata = {
        'corpus_hash': fingerprint(chunks),
        'provider': provider.identity,
        'chunk_ids': [c['chunk_id'] for c in chunks],
        'dimensions': vectors.shape[1],
    }
    folder = root / 'index'
    folder.mkdir(exist_ok=True)
    with (folder / 'vectors.tmp').open('wb') as output:
        np.save(output, vectors, allow_pickle=False)
    (folder / 'vectors.tmp').replace(folder / 'vectors.npy')
    (folder / 'metadata.json').write_text(json.dumps(metadata, indent=2))
    return {'chunks': len(chunks), 'dimensions': vectors.shape[1], 'provider': provider.identity}


class Retriever:
    """Search one processed corpus with BM25, dense cosine or fused rankings.

    Construction builds BM25 postings in memory. Dense vectors load only when a
    dense/hybrid query requires them; BM25 queries need no model API. Filters
    restrict eligible results before ranking, while BM25 corpus statistics remain
    based on the full corpus.
    """

    def __init__(self, root: Path):
        """Load root/chunks.jsonl and build BM25 postings, lengths and corpus fingerprint.

        Prepare the embedding provider and leave dense vectors unloaded. Raise
        ValueError when no processed chunks exist. Construction makes no API calls.
        """
        self.root = root
        self.chunks = read_jsonl(root / 'chunks.jsonl')
        if not self.chunks:
            raise ValueError('No chunks. Run process first.')
        self.corpus_hash = fingerprint(self.chunks)
        self.lengths = []
        self.postings = defaultdict(list)
        for index, chunk in enumerate(self.chunks):
            tokens = tokenize(embedding_text(chunk))
            self.lengths.append(len(tokens))
            for token, count in Counter(tokens).items():
                self.postings[token].append((index, count))
        self.average_length = sum(self.lengths) / len(self.lengths)
        self.provider = Provider('embedding', root / 'embedding_cache.sqlite')
        self.vectors = None

    def _dense(self):
        """Load and memoize dense vectors after checking corpus, provider, order and shape.

        Return the vector matrix. Raise ValueError for a missing, stale or mismatched
        index instead of silently using embeddings for a different corpus/model.
        """
        if self.vectors is None:
            metadata_path = self.root / 'index' / 'metadata.json'
            if not metadata_path.exists():
                raise ValueError('Dense index missing. Configure embedding API and run index first.')
            meta = json.loads(metadata_path.read_text())
            if meta['corpus_hash'] != self.corpus_hash or meta['provider'] != self.provider.identity:
                raise ValueError('Dense index is stale or uses another model/endpoint. Run index again.')
            if meta['chunk_ids'] != [c['chunk_id'] for c in self.chunks]:
                raise ValueError('Dense index chunk order mismatch')
            self.vectors = np.load(self.root / 'index' / 'vectors.npy', allow_pickle=False)
            if self.vectors.shape != (len(self.chunks), meta['dimensions']):
                raise ValueError('Dense index shape mismatch')
        return self.vectors

    def search(
        self, query: str, mode='bm25', k=10, companies=None, period_end=None, as_of=None, candidate_k=50
    ) -> list[dict]:
        """Return ranked hit dictionaries containing rank, score and the source chunk.

        Args:
            query: Nonblank question or keyword string.
            mode: "bm25", "dense" or "hybrid".
            k: Maximum number of returned hits.
            companies: Optional collection of company IDs to include.
            period_end: Optional exact reporting-period end in YYYY-MM-DD form.
            as_of: Optional publication-date cutoff; unknown dates are excluded.
            candidate_k: Candidate depth per ranking before fusion (at least k).

        BM25 uses positive lexical scores; dense search embeds the query and ranks
        normalized vector dot products. Hybrid merges both lists with reciprocal
        rank fusion using constant 60. Scores are method-specific, not calibrated
        confidence values. Return [] if filters leave no eligible chunks.

        Raises:
            ValueError: Query/settings are invalid or the dense index is incompatible.
            RuntimeError: A required embedding API call fails.
        """
        if mode not in {'bm25', 'dense', 'hybrid'} or k < 1 or candidate_k < 1:
            raise ValueError('Invalid search mode or candidate count')
        if not query.strip():
            raise ValueError('Query cannot be empty')
        eligible = []
        for i, chunk in enumerate(self.chunks):
            if companies and chunk['company_id'] not in companies:
                continue
            if period_end and chunk['period_end'] != period_end:
                continue
            # Unknown publication dates are excluded from point-in-time searches.
            if as_of and (not chunk.get('published_at') or chunk['published_at'][:10] > as_of):
                continue
            eligible.append(i)
        if not eligible:
            return []
        sparse = defaultdict(float)
        if mode in {'bm25', 'hybrid'}:
            count = len(self.chunks)
            for token in set(tokenize(query)):
                postings = self.postings.get(token, [])
                idf = math.log(1 + (count - len(postings) + 0.5) / (len(postings) + 0.5))
                for index, tf in postings:
                    denom = tf + 1.5 * (0.25 + 0.75 * self.lengths[index] / self.average_length)
                    sparse[index] += idf * tf * 2.5 / denom
        depth = max(k, candidate_k)
        sparse_rank = sorted(
            (i for i in eligible if sparse[i] > 0), key=lambda i: (-sparse[i], self.chunks[i]['chunk_id'])
        )[:depth]
        dense, dense_rank = {}, []
        if mode in {'dense', 'hybrid'}:
            vectors = self._dense()
            query_vector = self.provider.embed([query])[0]
            if vectors.shape[1] != len(query_vector):
                raise ValueError('Query embedding dimensions differ from index')
            values = vectors[eligible] @ query_vector
            dense = dict(zip(eligible, values.tolist(), strict=True))
            dense_rank = sorted(eligible, key=lambda i: (-dense[i], self.chunks[i]['chunk_id']))[:depth]
        if mode == 'hybrid':
            scores = defaultdict(float)
            for ranking in (sparse_rank, dense_rank):
                for rank, index in enumerate(ranking, 1):
                    scores[index] += 1 / (60 + rank)
            ranking = sorted(scores, key=lambda i: (-scores[i], self.chunks[i]['chunk_id']))
        else:
            ranking = sparse_rank if mode == 'bm25' else dense_rank
            scores = sparse if mode == 'bm25' else dense
        return [
            {'rank': rank, 'score': float(scores[index]), 'chunk': self.chunks[index]}
            for rank, index in enumerate(ranking[:k], 1)
        ]
