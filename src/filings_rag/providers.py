"""Call compatible model APIs, cache embedding responses and return normalized vectors.

Settings come from environment variables loaded by the CLI. Gemini uses
Google-scoped credentials and its OpenAI-compatible endpoint. This module
handles HTTP retries, response checks and a persistent SQLite embedding cache.
Cached values retain the response vectors; embed() normalizes them when returning
the float32 array used by local search.
"""

import json
import os
import sqlite3
import time
from pathlib import Path
from urllib.parse import urlparse

import httpx
import numpy as np

from .storage import digest


class Provider:
    """HTTP adapter for chat or embedding requests using environment configuration.

    Attributes:
        kind: Expected to be "embedding" or "llm"; selects the settings prefix.
        base: Compatible API endpoint without a trailing slash.
        key: Resolved credential; never include it in logs or cache identity.
        model: Configured model ID.
        client: HTTP client, optionally injected for offline tests.
        cache_path: SQLite path required by embed, unused by chat.

    The CLI loads .env before constructing this adapter. Construction alone does
    not make a model request.
    """

    def __init__(self, kind: str, cache_path: Path | None = None, client=None):
        """Resolve endpoint, model and credentials and prepare the HTTP client.

        kind selects EMBEDDING_* or LLM_* settings. GEMINI_API_KEY is considered
        only for Google’s endpoint. cache_path is needed for embeddings; client can
        be an httpx client with a mock transport for tests.
        """
        self.kind = kind
        prefix = 'EMBEDDING' if kind == 'embedding' else 'LLM'
        self.base = os.getenv(f'{prefix}_BASE_URL', 'https://api.openai.com/v1').rstrip('/')
        self.key = os.getenv(f'{prefix}_API_KEY') or os.getenv('LLM_API_KEY') or os.getenv('OPENAI_API_KEY')
        if urlparse(self.base).hostname == 'generativelanguage.googleapis.com':
            self.key = (
                os.getenv(f'{prefix}_API_KEY') or os.getenv('GEMINI_API_KEY') or os.getenv('LLM_API_KEY')
            )
        self.model = os.getenv(f'{prefix}_MODEL', 'text-embedding-3-small' if kind == 'embedding' else '')
        self.client = client or httpx.Client(timeout=120)
        self.cache_path = cache_path

    @property
    def identity(self):
        """Return the endpoint/model pair used for cache and index compatibility checks.

        Credentials are intentionally excluded. This property performs no network I/O.
        """
        return {'base_url': self.base, 'model': self.model}

    def request(self, route: str, body: dict) -> dict:
        """POST a model request and return the decoded JSON response.

        route is a path such as /embeddings; body supplies request fields. Add model
        and bearer authentication, retry selected HTTP failures up to four attempts,
        and raise a sanitized RuntimeError on HTTP or transport failures. Missing
        credentials/model raise ValueError. Transport failures are not retried here.
        """
        if not self.key or not self.model:
            raise ValueError(f'Configure {self.kind} API key and model in .env; see .env.example')
        for attempt in range(4):
            try:
                response = self.client.post(
                    self.base + route,
                    json={'model': self.model, **body},
                    headers={'Authorization': f'Bearer {self.key}'},
                )
            except httpx.TransportError as exc:
                raise RuntimeError(f'{self.kind} API network failure ({type(exc).__name__})') from None
            if response.status_code in {429, 500, 502, 503, 504} and attempt < 3:
                time.sleep(2**attempt)
                continue
            if response.is_error:
                # Avoid reflecting arbitrary upstream error payloads or credentials into logs.
                raise RuntimeError(f'{self.kind} API returned HTTP {response.status_code}')
            return response.json()
        raise RuntimeError('API retries exhausted')

    def embed(self, texts: list[str], progress: bool = False) -> np.ndarray:
        """Return unit-normalized float32 vectors in the same order as input texts.

        Args:
            texts: Nonempty, nonblank inputs; duplicates are allowed.
            progress: Print batch progress when building the document index.

        Returns:
            A NumPy array shaped (len(texts), embedding_dimensions).

        Reuse SQLite entries keyed by endpoint, model and exact text. Request only
        missing unique texts in batches of 32 and checkpoint every successful batch.
        Restore API result order using indices, including Google’s omitted index=0,
        and reject incomplete batches, nonfinite values and zero-length vectors.

        Raises:
            ValueError: Inputs/cache configuration or returned vectors are invalid.
            RuntimeError: A required model request fails.
        """
        if not texts or any(not text.strip() for text in texts):
            raise ValueError('Embeddings require nonempty text inputs')
        if self.cache_path is None:
            raise ValueError('An embedding cache path is required')
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        keys = [digest(json.dumps(self.identity, sort_keys=True) + '\n' + text) for text in texts]
        vectors = {}
        with sqlite3.connect(self.cache_path) as db:
            db.execute('CREATE TABLE IF NOT EXISTS embeddings (key TEXT PRIMARY KEY, vector TEXT)')
            for key in set(keys):
                row = db.execute('SELECT vector FROM embeddings WHERE key=?', (key,)).fetchone()
                if row:
                    vectors[key] = json.loads(row[0])
            missing = list(dict.fromkeys(key for key in keys if key not in vectors))
            by_key = dict(zip(keys, texts, strict=True))
            for start in range(0, len(missing), 32):
                batch = missing[start : start + 32]
                data = self.request(
                    '/embeddings', {'input': [by_key[k] for k in batch], 'encoding_format': 'float'}
                )['data']
                # Google's protobuf JSON can omit the default integer index=0.
                # Do not infer arbitrary positions: all other indices must still
                # uniquely cover the complete requested batch.
                if urlparse(self.base).hostname == 'generativelanguage.googleapis.com':
                    data = [{'index': 0, **item} for item in data]
                data = sorted(data, key=lambda item: item.get('index', -1))
                if [item.get('index') for item in data] != list(range(len(batch))):
                    raise ValueError('Embedding API returned incomplete or duplicate indices')
                for key, item in zip(batch, data, strict=True):
                    vector = item['embedding']
                    if not vector or not np.isfinite(np.asarray(vector, dtype=float)).all():
                        raise ValueError('Invalid embedding vector')
                    vectors[key] = vector
                    db.execute('INSERT OR REPLACE INTO embeddings VALUES (?, ?)', (key, json.dumps(vector)))
                db.commit()
                if progress:
                    print(f'Embedded {len(vectors)}/{len(set(keys))} unique chunks', flush=True)
        result = np.asarray([vectors[key] for key in keys], dtype=np.float32)
        norms = np.linalg.norm(result, axis=1, keepdims=True)
        if (norms == 0).any():
            raise ValueError('Embedding API returned a zero vector')
        return result / norms

    def chat(self, messages: list[dict]) -> str:
        """Send text or multimodal messages and return the first nonempty answer.

        Messages follow the compatible chat-completions contract. This method does
        not add retrieval context or validate citations; generation.answer does that.
        Raise ValueError for an absent/blank response and propagate request failures.
        """
        data = self.request('/chat/completions', {'messages': messages})
        text = data['choices'][0]['message'].get('content')
        if not isinstance(text, str) or not text.strip():
            raise ValueError('Model returned no text')
        return text
