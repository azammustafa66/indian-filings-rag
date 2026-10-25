# Pilot benchmark

`pilot.jsonl` contains 20 bootstrap development questions: 18 answerable and two deliberately outside the corpus period. Evidence is tied to SHA-256 document IDs, PDF page positions and source quotes, not chunk IDs. Cross-company questions require both evidence groups. Alternative valid evidence can be added within a group.

These labels were curated by the coding agent, with visual inspection of the three source pages used for answerable questions. They are **not independently human-annotated**, comprehensive relevance judgments, or an untouched test set. Every question is `split: dev`. No model-quality or generalization claim should be made from this small set. `review_status: verified` means source-checked, not human-approved.

The 18 answerable questions deliberately reuse three inspected pages. This is a pipeline smoke benchmark; expand page coverage and add independently reviewed company/period holdouts before comparing retrieval quality seriously. Do not split paraphrases or questions about the same underlying evidence across dev/test.

The evaluator checks that every gold quote exists on its source page. It measures evidence-group recall and complete evidence coverage at 5/10/20, first relevant reciprocal rank, and a novelty-aware normalized DCG variant. **The DCG variant is not standard graded nDCG.** Repeated overlapping chunks earn no additional novelty credit. A retrieved chunk must contain the annotated quote to count; other valid passages may currently be unjudged.

Unanswerable queries are retained in retrieval run logs but excluded from recall averages. A retriever returning candidates does not establish answerability. Generation correctness, claim support, abstention, standard pooled qrels/nDCG, confidence intervals, and independent annotation remain future evaluation work.

Run:

```bash
uv run filings evaluate --modes bm25
# After API credentials and a dense index exist:
uv run filings evaluate --modes bm25 dense hybrid --output experiments/comparison
```

The report records corpus and benchmark hashes, retrieval configuration, embedding endpoint/model, per-category results, and run details. Latency includes query embedding/cache effects and first-query index loading; it is not a controlled serving benchmark. Source publication timestamps are unknown in the pilot: filenames and cover-letter dates are not treated as verified exchange dissemination times.
