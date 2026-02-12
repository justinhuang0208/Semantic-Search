from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class DocumentRecord:
    doc_id: str
    title: str
    source_path: str
    tags: list[str]
    out_links: list[str]
    updated_at: str
    content_hash: str
    char_count: int


@dataclass(slots=True)
class ChunkDraft:
    chunk_id: str
    doc_id: str
    title: str
    source_path: str
    section_path: str
    chunk_type: str
    context_prefix: str
    text: str
    search_text: str
    token_count: int
    content_hash: str
    tags: list[str]
    out_links: list[str]
    updated_at: str


@dataclass(slots=True)
class SearchResult:
    chunk_rowid: int
    chunk_id: str
    doc_id: str
    title: str
    source_path: str
    section_path: str
    chunk_type: str
    text: str
    fusion_score: float
    vector_rank: int | None
    bm25_rank: int | None
