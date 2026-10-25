"""Run a small live chat and embedding API connection check.

Execute from the repository root after configuring .env. Running this script
makes model API calls, prints a short response and vector dimensions, and writes
embeddings to data/embedding_cache.sqlite. Credentials are not printed.
"""

from pathlib import Path

import httpx
from dotenv import load_dotenv

from filings_rag.providers import Provider

load_dotenv()
try:
    llm = Provider('llm')
    print(f'Chat model: {llm.model}', flush=True)
    result = llm.chat([{'role': 'user', 'content': 'Reply with exactly: API connection works'}])
    print(f'Chat response: {result}', flush=True)
    embedding = Provider('embedding', Path('data/embedding_cache.sqlite'))
    vectors = embedding.embed(['Indian company annual report', 'Quarterly financial results'])
    print(f'Embedding model: {embedding.model}; shape: {vectors.shape}', flush=True)
except httpx.TransportError:
    raise SystemExit('Network transport failed; no credentials printed') from None
except (RuntimeError, ValueError) as exc:
    raise SystemExit(str(exc)) from None
