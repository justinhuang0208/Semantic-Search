from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import yaml

from .collections import DEFAULT_COLLECTIONS_PATH, CollectionConfig, CollectionRegistry
from .embeddings import resolve_embedder
from .markdown_ingest import parse_markdown
from .models import ChunkDraft, SearchResult
from .retrieval import bm25_search, reciprocal_rank_fusion, rerank_with_doc_diversity
from .storage import Storage
from .tokenize import count_terms
from .utils import is_hidden_or_ignored, normalize_query_text
from .vector_index import VectorIndex, VectorIndexError

SearchMode = Literal["fulltext", "vector", "hybrid"]


@dataclass(slots=True)
class IngestStats:
    documents: int
    chunks: int
    embedding_dim: int
    embedding_provider: str
    embedding_model: str
    source: str
    collection_id: str
    collection_name: str
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


def _collect_markdown_files(source: Path, mask: str) -> list[Path]:
    files: list[Path] = []
    for path in sorted(source.glob(mask)):
        if not path.is_file():
            continue
        if is_hidden_or_ignored(path):
            continue
        files.append(path)
    return files


def _ensure_embedding_profile_matches(
    storage: Storage,
    *,
    provider: str,
    model: str,
    cache_key: str,
) -> None:
    profile = storage.embedding_profile()
    if profile is None:
        raise RuntimeError(
            "Embedding profile is missing for this index. "
            "Run `semsearch ingest` once to initialize metadata."
        )

    expected = {
        "provider": provider,
        "model": model,
        "cache_key": cache_key,
    }
    mismatches = [
        f"{key}=expected({value}) actual({profile[key]})"
        for key, value in expected.items()
        if str(profile[key]) != value
    ]
    if mismatches:
        mismatch_text = ", ".join(mismatches)
        raise RuntimeError(
            "Embedding configuration does not match the indexed vectors: "
            f"{mismatch_text}. Run `semsearch ingest --rebuild` with the same embedding settings."
        )


def _resolve_ingest_collection(
    registry: CollectionRegistry,
    source: Path,
    collection: str | None,
) -> CollectionConfig:
    source_root = source.expanduser().resolve()
    if collection:
        collection_id, prefix = registry.resolve_context_target(collection)
        if prefix:
            raise RuntimeError(
                "Collection filters do not accept path prefixes. Use a collection name or id."
            )
        if collection_id is None:
            raise RuntimeError(f"Collection not found: {collection}")
        resolved = registry.find_collection(collection_id)
        resolved_root = resolved.root_path_resolved()
        if resolved_root != source_root:
            raise RuntimeError(
                f"Source path mismatch for collection {resolved.name}: "
                f"expected {resolved_root}, got {source_root}"
            )
        return resolved

    matched = registry.find_by_root(source_root)
    if matched is not None:
        return matched

    return registry.ensure_collection_for_source(source_root)


def _resolve_query_collections(
    registry: CollectionRegistry,
    collection: str | None,
) -> list[str] | None:
    if collection:
        collection_id, prefix = registry.resolve_context_target(collection)
        if prefix:
            raise RuntimeError(
                "Collection filters do not accept path prefixes. Use a collection name or id."
            )
        if collection_id is None:
            raise RuntimeError(f"Collection not found: {collection}")
        return [collection_id]
    if not registry.collections:
        return None
    return [item.collection_id for item in registry.default_collections()]


def ingest(
    source: Path,
    db_path: Path,
    faiss_path: Path,
    api_key: str | None,
    model: str,
    rebuild: bool,
    use_local_embedding: bool = False,
    collections_path: Path = DEFAULT_COLLECTIONS_PATH,
    collection: str | None = None,
) -> IngestStats:
    registry = CollectionRegistry.load(collections_path)
    collection_cfg = _resolve_ingest_collection(registry, source, collection)
    source_root = collection_cfg.root_path_resolved()

    files = _collect_markdown_files(source_root, collection_cfg.mask)
    if not files:
        raise RuntimeError(f"No markdown files found in {source_root}")

    documents = []
    chunks: list[ChunkDraft] = []
    for path in files:
        relative_path = path.relative_to(source_root).as_posix()
        context_text = registry.render_context_text(collection_cfg.collection_id, relative_path)
        doc, drafts = parse_markdown(
            path,
            collection_id=collection_cfg.collection_id,
            collection_name=collection_cfg.name,
            relative_path=relative_path,
            context_text=context_text,
        )
        documents.append(doc)
        chunks.extend(drafts)

    storage = Storage(db_path)
    storage.create_schema()
    if rebuild:
        storage.clear_for_rebuild()

    source_to_doc = {doc.doc_id: doc for doc in documents}
    source_to_chunks: dict[str, list[ChunkDraft]] = {}
    for chunk in chunks:
        source_to_chunks.setdefault(chunk.doc_id, []).append(chunk)

    existing_hash_by_doc = storage.document_hashes(collection_id=collection_cfg.collection_id)

    current_doc_ids = set(source_to_doc.keys())
    existing_doc_ids = set(existing_hash_by_doc.keys())
    deleted_doc_ids = sorted(existing_doc_ids - current_doc_ids)

    if rebuild:
        updated_doc_ids = sorted(current_doc_ids)
    else:
        updated_doc_ids = sorted(
            doc_id
            for doc_id, doc in source_to_doc.items()
            if existing_hash_by_doc.get(doc_id) != doc.document_hash
        )

    doc_ids_to_delete = sorted(set(updated_doc_ids) | set(deleted_doc_ids))
    storage.delete_documents_by_doc_ids(doc_ids_to_delete)

    for doc_id in updated_doc_ids:
        storage.insert_document(source_to_doc[doc_id])

    chunks_to_insert: list[ChunkDraft] = []
    for doc_id in updated_doc_ids:
        chunks_to_insert.extend(source_to_chunks.get(doc_id, []))

    runtime = resolve_embedder(
        use_local_embedding=use_local_embedding,
        model=model,
        api_key=api_key,
    )
    embedder = runtime.embedder
    cache_key = runtime.cache_key

    unique_hash_to_text: dict[str, str] = {}
    for chunk in chunks_to_insert:
        unique_hash_to_text.setdefault(chunk.embedding_hash, chunk.search_text)

    requested_hashes = list(unique_hash_to_text.keys())
    cached_vectors = storage.embedding_cache_by_hashes(
        model=cache_key,
        embedding_hashes=requested_hashes,
    )
    missing_hashes = [h for h in requested_hashes if h not in cached_vectors]
    if missing_hashes:
        texts = [unique_hash_to_text[h] for h in missing_hashes]
        embedded = embedder.embed_texts(texts, input_type="document")
        for embedding_hash, vector in zip(missing_hashes, embedded.vectors, strict=True):
            storage.upsert_embedding_cache(
                model=cache_key,
                embedding_hash=embedding_hash,
                vector=vector,
            )

    for chunk in chunks_to_insert:
        rowid = storage.insert_chunk(chunk)
        tf = dict(count_terms(chunk.search_text))
        storage.insert_bm25_terms(rowid, tf)

    missing_cache = storage.missing_embedding_hashes_with_text(model=cache_key)
    missing_cache_hashes = [h for h in missing_cache.keys() if h not in missing_hashes]
    if missing_cache_hashes:
        texts = [missing_cache[h] for h in missing_cache_hashes]
        embedded = embedder.embed_texts(texts, input_type="document")
        for embedding_hash, vector in zip(missing_cache_hashes, embedded.vectors, strict=True):
            storage.upsert_embedding_cache(
                model=cache_key,
                embedding_hash=embedding_hash,
                vector=vector,
            )

    storage.clear_bm25_derived()
    term_df = storage.compute_term_df()
    token_lengths = storage.token_lengths()
    avgdl = sum(token_lengths) / len(token_lengths) if token_lengths else 0.0
    storage.finalize_bm25(doc_count=len(token_lengths), avgdl=avgdl, term_df=term_df)
    storage.commit()

    doc_count, chunk_count = storage.counts()
    ids, vectors = storage.all_chunk_vectors(model=cache_key)
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
    storage.upsert_embedding_profile(
        provider=runtime.provider,
        model=runtime.model,
        cache_key=cache_key,
        dim=len(vectors[0]),
    )
    storage.commit()

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
        embedding_provider=runtime.provider,
        embedding_model=runtime.model,
        source=str(source_root),
        collection_id=collection_cfg.collection_id,
        collection_name=collection_cfg.name,
        updated_documents=len(updated_doc_ids),
        deleted_documents=len(deleted_doc_ids),
        new_embedding_hashes=new_hashes,
        reused_embedding_hashes=reused_hashes,
    )


def search(
    query: str,
    db_path: Path,
    faiss_path: Path,
    api_key: str | None = None,
    model: str | None = None,
    top_k: int = 8,
    vector_top_k: int = 20,
    bm25_top_k: int = 20,
    use_local_embedding: bool = False,
    collections_path: Path = DEFAULT_COLLECTIONS_PATH,
    collection: str | None = None,
    search_mode: SearchMode = "hybrid",
) -> list[SearchResult]:
    normalized_query = normalize_query_text(query)
    if not normalized_query:
        return []

    if search_mode not in {"hybrid", "fulltext", "vector"}:
        raise ValueError(f"Unsupported search mode: {search_mode}")

    registry = CollectionRegistry.load(collections_path)
    selected_collection_ids = _resolve_query_collections(registry, collection)
    collection_name_by_id = {
        item.collection_id: item.name for item in registry.collections
    }

    storage = Storage(db_path)
    try:
        storage.create_schema()
        runtime = None
        vector_results: list[tuple[int, float]] = []
        bm25_results: list[tuple[int, float]] = []

        if search_mode != "fulltext":
            runtime = resolve_embedder(
                use_local_embedding=use_local_embedding,
                model=model,
                api_key=api_key,
            )
            _ensure_embedding_profile_matches(
                storage,
                provider=runtime.provider,
                model=runtime.model,
                cache_key=runtime.cache_key,
            )

            embedded = runtime.embedder.embed_texts([normalized_query], input_type="query")

            index = VectorIndex(faiss_path)
            try:
                vector_results = index.search(embedded.vectors[0], top_k=vector_top_k)
            except VectorIndexError:
                ids, vectors = storage.all_chunk_vectors(model=runtime.cache_key)
                vector_results = index.search_in_memory(
                    embedded.vectors[0],
                    vectors=vectors,
                    ids=ids,
                    top_k=vector_top_k,
                )
            if selected_collection_ids is not None:
                allowed_collection_ids = set(selected_collection_ids)
                rows = storage.chunks_by_ids([chunk_id for chunk_id, _score in vector_results])
                vector_results = [
                    (chunk_id, score)
                    for chunk_id, score in vector_results
                    if chunk_id in rows and str(rows[chunk_id]["collection_id"]) in allowed_collection_ids
                ]

        if search_mode != "vector":
            bm25_results = bm25_search(
                storage,
                normalized_query,
                top_k=bm25_top_k,
                collection_ids=selected_collection_ids,
            )

        if search_mode == "fulltext":
            fused = bm25_results
            vector_rank = {}
            bm25_rank = {chunk_id: rank for rank, (chunk_id, _score) in enumerate(bm25_results, start=1)}
        elif search_mode == "vector":
            fused = vector_results
            vector_rank = {chunk_id: rank for rank, (chunk_id, _score) in enumerate(vector_results, start=1)}
            bm25_rank = {}
        else:
            fused, vector_rank, bm25_rank = reciprocal_rank_fusion(vector_results, bm25_results, k=60)
        results = rerank_with_doc_diversity(
            storage,
            fused=fused,
            vector_rank=vector_rank,
            bm25_rank=bm25_rank,
            top_k=top_k,
        )
        if search_mode == "fulltext":
            results = sorted(results, key=lambda item: item.fusion_score, reverse=True)
        for item in results:
            item.collection_name = collection_name_by_id.get(item.collection_id, item.collection_name)
        return results
    finally:
        storage.close()


def evaluate(
    golden_path: Path,
    db_path: Path,
    faiss_path: Path,
    api_key: str | None,
    model: str,
    use_local_embedding: bool = False,
    collections_path: Path = DEFAULT_COLLECTIONS_PATH,
    collection: str | None = None,
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
            use_local_embedding=use_local_embedding,
            top_k=10,
            vector_top_k=20,
            bm25_top_k=20,
            collections_path=collections_path,
            collection=collection,
            search_mode="hybrid",
        )

        ranked_docs = [Path(r.relative_path).name for r in results]
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
