from __future__ import annotations

import math
from collections import defaultdict

from .models import SearchResult
from .storage import Storage
from .tokenize import tokenize_for_bm25


def bm25_search(storage: Storage, query: str, top_k: int, k1: float = 1.5, b: float = 0.75) -> list[tuple[int, float]]:
    terms = tokenize_for_bm25(query)
    if not terms:
        return []

    n_docs, avgdl = storage.bm25_globals()
    if n_docs <= 0 or avgdl <= 0:
        return []

    postings = storage.bm25_postings(terms)
    scores: dict[int, float] = defaultdict(float)

    for row in postings:
        chunk_id = int(row["chunk_rowid"])
        tf = float(row["tf"])
        df = float(row["df"])
        dl = float(row["dl"])

        idf = math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))
        denom = tf + k1 * (1.0 - b + b * (dl / avgdl))
        if denom <= 0:
            continue
        scores[chunk_id] += idf * (tf * (k1 + 1.0) / denom)

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return ranked[:top_k]


def reciprocal_rank_fusion(
    vector_results: list[tuple[int, float]],
    bm25_results: list[tuple[int, float]],
    k: int = 60,
) -> tuple[list[tuple[int, float]], dict[int, int], dict[int, int]]:
    fused: dict[int, float] = defaultdict(float)
    vector_rank: dict[int, int] = {}
    bm25_rank: dict[int, int] = {}

    for rank, (chunk_id, _score) in enumerate(vector_results, start=1):
        vector_rank[chunk_id] = rank
        fused[chunk_id] += 1.0 / (k + rank)

    for rank, (chunk_id, _score) in enumerate(bm25_results, start=1):
        bm25_rank[chunk_id] = rank
        fused[chunk_id] += 1.0 / (k + rank)

    ranked = sorted(fused.items(), key=lambda x: x[1], reverse=True)
    return ranked, vector_rank, bm25_rank


def rerank_with_doc_diversity(
    storage: Storage,
    fused: list[tuple[int, float]],
    vector_rank: dict[int, int],
    bm25_rank: dict[int, int],
    top_k: int,
) -> list[SearchResult]:
    chunk_ids = [chunk_id for chunk_id, _ in fused]
    rows = storage.chunks_by_ids(chunk_ids)

    ordered_results: list[tuple[int, float]] = [(cid, score) for cid, score in fused if cid in rows]

    unique_first: list[tuple[int, float]] = []
    remaining: list[tuple[int, float]] = []
    seen_docs: set[str] = set()

    for chunk_id, score in ordered_results:
        row = rows[chunk_id]
        doc_id = str(row["doc_id"])
        if doc_id not in seen_docs:
            unique_first.append((chunk_id, score))
            seen_docs.add(doc_id)
        else:
            remaining.append((chunk_id, score))

    final_ids = unique_first + remaining

    results: list[SearchResult] = []
    for chunk_id, score in final_ids[:top_k]:
        row = rows[chunk_id]
        results.append(
            SearchResult(
                chunk_rowid=chunk_id,
                chunk_id=str(row["chunk_id"]),
                doc_id=str(row["doc_id"]),
                title=str(row["title"]),
                source_path=str(row["source_path"]),
                section_path=str(row["section_path"]),
                chunk_type=str(row["chunk_type"]),
                text=str(row["text"]),
                fusion_score=float(score),
                vector_rank=vector_rank.get(chunk_id),
                bm25_rank=bm25_rank.get(chunk_id),
            )
        )
    return results
