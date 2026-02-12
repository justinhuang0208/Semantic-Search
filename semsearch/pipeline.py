import math
from dataclasses import dataclass
from pathlib import Path

import yaml

from .embeddings import OpenRouterEmbedder
from .markdown_ingest import parse_markdown
from .models import ChunkDraft, SearchResult
from .retrieval import bm25_search, reciprocal_rank_fusion, rerank_with_doc_diversity
from .storage import Storage
from .tokenize import count_terms
from .utils import is_hidden_or_ignored, normalize_query_text
from .vector_index import VectorIndex


@dataclass(slots=True)
class IngestStats:
    documents: int
    chunks: int
    embedding_dim: int
    source: str
    updated_documents: int
    deleted_documents: int
    new_embedding_hashes: int
    reused_embedding_hashes: int


@dataclass(slots=True)
class EvalStats:
    recall_at_5: float
    mrr_at_10: float
    ndcg_at_10: float
    queries: int


def _collect_markdown_files(source: Path) -> list[Path]:
    files = []
    for path in sorted(source.glob("*.md")):
        if is_hidden_or_ignored(path):
            continue
        files.append(path)
    return files


def ingest(
    source: Path,
    db_path: Path,
    faiss_path: Path,
    api_key: str,
    model: str,
    rebuild: bool,
) -> IngestStats:
    files = _collect_markdown_files(source)
    if not files:
        raise RuntimeError(f"No markdown files found in {source}")

    documents = []
    chunks: list[ChunkDraft] = []
    for path in files:
        doc, drafts = parse_markdown(path)
        documents.append(doc)
        chunks.extend(drafts)

    storage = Storage(db_path)
    storage.create_schema()
    if rebuild:
        storage.clear_for_rebuild()

    source_to_doc = {doc.source_path: doc for doc in documents}
    source_to_chunks: dict[str, list[ChunkDraft]] = {}
    for chunk in chunks:
        source_to_chunks.setdefault(chunk.source_path, []).append(chunk)

    existing_hash_by_source = storage.document_hashes_by_source()

    current_sources = set(source_to_doc.keys())
    existing_sources = set(existing_hash_by_source.keys())
    deleted_sources = sorted(existing_sources - current_sources)

    if rebuild:
        updated_sources = sorted(current_sources)
    else:
        updated_sources = sorted(
            source_path
            for source_path, doc in source_to_doc.items()
            if existing_hash_by_source.get(source_path) != doc.content_hash
        )

    to_delete_sources = sorted(set(updated_sources) | set(deleted_sources))
    storage.delete_documents_by_source(to_delete_sources)

    for source_path in updated_sources:
        storage.insert_document(source_to_doc[source_path])

    chunks_to_insert: list[ChunkDraft] = []
    for source_path in updated_sources:
        chunks_to_insert.extend(source_to_chunks.get(source_path, []))

    embedder = OpenRouterEmbedder(api_key=api_key, model=model)

    unique_hash_to_text: dict[str, str] = {}
    for chunk in chunks_to_insert:
        unique_hash_to_text.setdefault(chunk.content_hash, chunk.search_text)

    requested_hashes = list(unique_hash_to_text.keys())
    cached_vectors = storage.embedding_cache_by_hashes(model=model, content_hashes=requested_hashes)
    missing_hashes = [h for h in requested_hashes if h not in cached_vectors]
    if missing_hashes:
        texts = [unique_hash_to_text[h] for h in missing_hashes]
        embedded = embedder.embed_texts(texts, input_type="document")
        for content_hash, vector in zip(missing_hashes, embedded.vectors, strict=True):
            storage.upsert_embedding_cache(model=model, content_hash=content_hash, vector=vector)

    for chunk in chunks_to_insert:
        rowid = storage.insert_chunk(chunk)
        tf = dict(count_terms(chunk.search_text))
        storage.insert_bm25_terms(rowid, tf)

    missing_cache = storage.missing_embedding_hashes_with_text(model=model)
    missing_cache_hashes = [h for h in missing_cache.keys() if h not in missing_hashes]
    if missing_cache_hashes:
        texts = [missing_cache[h] for h in missing_cache_hashes]
        embedded = embedder.embed_texts(texts, input_type="document")
        for content_hash, vector in zip(missing_cache_hashes, embedded.vectors, strict=True):
            storage.upsert_embedding_cache(model=model, content_hash=content_hash, vector=vector)

    storage.clear_bm25_derived()
    term_df = storage.compute_term_df()
    token_lengths = storage.token_lengths()
    avgdl = sum(token_lengths) / len(token_lengths) if token_lengths else 0.0
    storage.finalize_bm25(doc_count=len(token_lengths), avgdl=avgdl, term_df=term_df)
    storage.commit()

    doc_count, chunk_count = storage.counts()
    ids, vectors = storage.all_chunk_vectors(model=model)
    if not vectors:
        storage.close()
        raise RuntimeError("No vectors available to build FAISS index.")
    if len(ids) != chunk_count:
        storage.close()
        raise RuntimeError(
            "Chunk/vector count mismatch. Try running ingest with --rebuild once to recover cache."
        )

    index = VectorIndex(faiss_path)
    if faiss_path.exists():
        faiss_path.unlink()
    index.build(vectors=vectors, ids=ids, dim=len(vectors[0]))

    storage.close()

    requested_hash_set = set(requested_hashes)
    missing_hash_set = set(missing_hashes)
    reused_hashes = len(requested_hash_set - missing_hash_set)
    new_hashes = len(missing_hash_set)
    if missing_cache_hashes:
        new_hashes += len(missing_cache_hashes)

    return IngestStats(
        documents=doc_count,
        chunks=chunk_count,
        embedding_dim=len(vectors[0]),
        source=str(source),
        updated_documents=len(updated_sources),
        deleted_documents=len(deleted_sources),
        new_embedding_hashes=new_hashes,
        reused_embedding_hashes=reused_hashes,
    )


def search(
    query: str,
    db_path: Path,
    faiss_path: Path,
    api_key: str,
    model: str,
    top_k: int,
    vector_top_k: int = 20,
    bm25_top_k: int = 20,
) -> list[SearchResult]:
    normalized_query = normalize_query_text(query)
    if not normalized_query:
        return []

    embedder = OpenRouterEmbedder(api_key=api_key, model=model)
    embedded = embedder.embed_texts([normalized_query], input_type="query")

    index = VectorIndex(faiss_path)
    vector_results = index.search(embedded.vectors[0], top_k=vector_top_k)

    storage = Storage(db_path)
    storage.create_schema()

    bm25_results = bm25_search(storage, normalized_query, top_k=bm25_top_k)
    fused, vector_rank, bm25_rank = reciprocal_rank_fusion(vector_results, bm25_results, k=60)
    results = rerank_with_doc_diversity(
        storage,
        fused=fused,
        vector_rank=vector_rank,
        bm25_rank=bm25_rank,
        top_k=top_k,
    )
    storage.close()
    return results


def evaluate(
    golden_path: Path,
    db_path: Path,
    faiss_path: Path,
    api_key: str,
    model: str,
) -> tuple[EvalStats, list[dict]]:
    data = yaml.safe_load(golden_path.read_text(encoding="utf-8"))
    queries = data.get("queries", [])
    if not queries:
        raise RuntimeError("No queries found in golden file.")

    details: list[dict] = []
    recall_values: list[float] = []
    rr_values: list[float] = []
    ndcg_values: list[float] = []

    for item in queries:
        query = str(item["query"])
        relevant_patterns = [str(x) for x in item.get("relevant_docs", [])]

        results = search(
            query=query,
            db_path=db_path,
            faiss_path=faiss_path,
            api_key=api_key,
            model=model,
            top_k=10,
            vector_top_k=20,
            bm25_top_k=20,
        )

        ranked_docs = [Path(r.source_path).name for r in results]
        relevant_set = set(relevant_patterns)

        matched_patterns_for_recall: set[str] = set()
        recall_hits = 0
        for doc in ranked_docs[:5]:
            for pattern in relevant_set:
                if pattern in doc and pattern not in matched_patterns_for_recall:
                    matched_patterns_for_recall.add(pattern)
                    recall_hits += 1

        recall_den = max(len(relevant_set), 1)
        recall = min(recall_hits / recall_den, 1.0)
        recall_values.append(recall)

        rr = 0.0
        for idx, doc in enumerate(ranked_docs[:10], start=1):
            if any(pattern in doc for pattern in relevant_set):
                rr = 1.0 / idx
                break
        rr_values.append(rr)

        matched_patterns_for_dcg: set[str] = set()
        dcg = 0.0
        for idx, doc in enumerate(ranked_docs[:10], start=1):
            hit = 0
            for pattern in relevant_set:
                if pattern in doc and pattern not in matched_patterns_for_dcg:
                    matched_patterns_for_dcg.add(pattern)
                    hit = 1
                    break
            if hit == 1:
                dcg += 1.0 / math.log2(idx + 1.0)
        ideal_hits = min(len(relevant_set), 10)
        idcg = sum(1.0 / math.log2(i + 1.0) for i in range(1, ideal_hits + 1))
        ndcg = dcg / idcg if idcg > 0 else 0.0
        ndcg_values.append(ndcg)

        details.append(
            {
                "query": query,
                "relevant_docs": relevant_patterns,
                "top_results": ranked_docs[:10],
                "recall_at_5": recall,
                "rr_at_10": rr,
                "ndcg_at_10": ndcg,
            }
        )

    stats = EvalStats(
        recall_at_5=sum(recall_values) / len(recall_values),
        mrr_at_10=sum(rr_values) / len(rr_values),
        ndcg_at_10=sum(ndcg_values) / len(ndcg_values),
        queries=len(queries),
    )
    return stats, details
