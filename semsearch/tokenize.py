from __future__ import annotations

import re
import unicodedata
from collections import Counter

_WORD_RE = re.compile(r"[a-z0-9_+\-.]+")
_CJK_SEQ_RE = re.compile(r"[\u4e00-\u9fff]+")
_ATOMIC_RE = re.compile(r"[\u4e00-\u9fff]|[A-Za-z0-9_+\-.]+")


def _normalize(text: str) -> str:
    return unicodedata.normalize("NFKC", text).lower()


def tokenize_for_bm25(text: str) -> list[str]:
    normalized = _normalize(text)
    tokens: list[str] = []
    tokens.extend(_WORD_RE.findall(normalized))

    for seq in _CJK_SEQ_RE.findall(normalized):
        if len(seq) == 1:
            tokens.append(seq)
            continue
        for i in range(len(seq) - 1):
            tokens.append(seq[i : i + 2])
    return tokens


def count_terms(text: str) -> Counter[str]:
    return Counter(tokenize_for_bm25(text))


def rough_token_count(text: str) -> int:
    return len(_ATOMIC_RE.findall(_normalize(text)))


def split_long_text_by_tokens(text: str, window: int = 800, overlap: int = 120) -> list[str]:
    atoms = _ATOMIC_RE.findall(_normalize(text))
    if not atoms:
        return []
    if len(atoms) <= window:
        return [text.strip()]

    chunks: list[str] = []
    start = 0
    step = max(window - overlap, 1)
    while start < len(atoms):
        end = min(start + window, len(atoms))
        window_atoms = atoms[start:end]
        chunks.append(" ".join(window_atoms).strip())
        if end == len(atoms):
            break
        start += step
    return chunks
