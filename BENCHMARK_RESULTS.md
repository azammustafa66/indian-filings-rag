# Live retrieval comparison

Gemini 3.8 Flash is configured for answers and vision transcription; `gemini-embedding-001` supplies retrieval vectors through Google’s compatible API endpoint. The index has 3,246 chunks and 3,072 dimensions. All modes used the same corpus and 20-question bootstrap development benchmark (18 answerable, two outside the corpus period).

| Mode | Evidence recall@10 | Complete coverage@10 | MRR@10 | Cross-company complete@10 |
| --- | ---: | ---: | ---: | ---: |
| bm25 | 83.3% | 83.3% | 0.487 | 50% |
| dense | 75.0% | 72.2% | 0.467 | 0% |
| hybrid | 86.1% | 83.3% | 0.564 | 0% |

Hybrid ranks some annotated evidence earlier and has higher average evidence-group recall, but complete coverage is tied with BM25 at 15/18. Neither dense nor hybrid retrieves complete annotated evidence for either of the two cross-company questions at top 10; BM25 completes one. This remains an unresolved retrieval weakness. No reranker or per-company retrieval balancing is applied.

These are source-checked, agent-curated development labels concentrated on three rendered evidence pages, not independent human judgments or held-out results. Other valid passages can remain unjudged. The two unanswerable queries are logged but excluded from recall averages. This comparison does not evaluate answer correctness or abstention. Do not treat the small score difference as statistically established.

The corpus and benchmark were unchanged from the BM25 baseline. No retrieval parameters were tuned using these results. Query vectors were embedded during the dense run and reused from cache during hybrid evaluation, so latency figures are not a fair serving comparison.

Live checks also exercised Gemini chat, page-image transcription and cited BM25/hybrid answers. Both answer checks matched the reference figure; this is a smoke check, not a generation benchmark. Ten offline regression tests and Ruff pass. The vision proposal remains unreviewed and was not merged into the benchmark corpus.

Full ranked outputs and hashes: `experiments/gemini-comparison/report.json` and `retrieval_runs.jsonl` (generated, gitignored).

Reproduce:

```bash
uv run filings index
uv run filings evaluate --modes bm25 dense hybrid --output experiments/gemini-comparison
```

Provider references: [Google compatibility API](https://ai.google.dev/gemini-api/docs/openai), [Gemini 3.8 Flash](https://ai.google.dev/gemini-api/docs/models/gemini-3.8-flash).

These results describe the original three-report corpus. Loading additional companies changes the corpus hash; the dashboard marks these saved runs as stale rather than presenting them as current scores.
