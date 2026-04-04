from __future__ import annotations

import hashlib
import re
import unicodedata
from pathlib import Path


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def normalize_path_text(value: str | Path, *, absolute: bool = False) -> str:
    if not str(value).strip():
        return ""
    path = Path(value).expanduser()
    if absolute:
        path = path.resolve()
    text = path.as_posix()
    if not absolute and text.startswith("./"):
        text = text[2:]
    text = re.sub(r"/+", "/", text)
    if text == ".":
        return ""
    return text.rstrip("/")


def prefix_matches_path(path_text: str, prefix_text: str) -> bool:
    path_norm = normalize_path_text(path_text)
    prefix_norm = normalize_path_text(prefix_text)
    if not prefix_norm:
        return True
    if path_norm == prefix_norm:
        return True
    return path_norm.startswith(f"{prefix_norm}/")


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
