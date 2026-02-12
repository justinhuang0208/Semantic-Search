from __future__ import annotations

import hashlib
import re
import unicodedata
from pathlib import Path


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def normalize_query_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).lower()
    normalized = normalized.replace("\u3000", " ")
    normalized = re.sub(r"[\u2010-\u2015]", "-", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def is_hidden_or_ignored(path: Path) -> bool:
    return any(part.startswith(".") for part in path.parts)


def snippet(text: str, max_len: int = 180) -> str:
    content = re.sub(r"\s+", " ", text).strip()
    if len(content) <= max_len:
        return content
    return content[: max_len - 3] + "..."
