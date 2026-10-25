"""Shared SHA-256 hashing and JSON Lines file helpers.

JSONL stores one record per line. Hashes identify source PDFs and detect changes
to the corpus or benchmark; temporary-file replacement limits partial writes.
"""

import hashlib
import json
from pathlib import Path


def digest(value: str | bytes) -> str:
    """Return a hexadecimal SHA-256 hash of bytes or UTF-8-encoded text."""
    return hashlib.sha256(value.encode() if isinstance(value, str) else value).hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    """Read nonblank JSONL records, returning an empty list if the file is absent.

    Malformed JSON raises a decoding error instead of being silently skipped.
    """
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    """Write records as Unicode-preserving JSONL through a temporary file.

    Create parent directories, then replace the destination after serialization.
    This overwrites the destination; it does not append or coordinate writers.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows))
    temporary.replace(path)


def fingerprint(rows: list[dict]) -> str:
    """Hash records with sorted object keys while preserving their list order.

    Used to detect corpus or benchmark changes, including changes in record order.
    """
    return digest(json.dumps(rows, sort_keys=True, ensure_ascii=False))
