# Indian Filings RAG

A Python project for evidence-grounded questions over Indian listed-company filings.
The implementation is organized into ingestion, retrieval, evaluation and application layers.

## Development setup

```bash
uv sync --python 3.12
```

Use `.env.example` for local model settings. Credentials, downloaded filings,
model caches and generated experiments are excluded from version control.
