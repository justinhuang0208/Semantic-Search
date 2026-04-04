from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .collections import DEFAULT_COLLECTIONS_PATH, CollectionRegistry, default_index_paths
from .embeddings import DEFAULT_OLLAMA_MODEL, DEFAULT_OPENROUTER_MODEL
from .models import SearchResult
from .pipeline import evaluate, ingest, search

DEFAULT_SOURCE = Path("1 - Cards")
DEFAULT_DB_PATH = Path("data_index/semsearch.db")
DEFAULT_FAISS_PATH = Path("data_index/semsearch.faiss")
SOURCE_ENV_VAR = "SEMSEARCH_SOURCE"
COLLECTIONS_ENV_VAR = "SEMSEARCH_COLLECTIONS"


def _resolve_model(model: str | None, use_local_embedding: bool) -> str:
    if model and model.strip():
        return model.strip()
    if use_local_embedding:
        return DEFAULT_OLLAMA_MODEL
    return DEFAULT_OPENROUTER_MODEL


def _api_key(use_local_embedding: bool) -> str | None:
    if use_local_embedding:
        return None
    api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set.")
    return api_key


def _default_source() -> str:
    configured = os.getenv(SOURCE_ENV_VAR, "").strip()
    if configured:
        return configured
    return str(DEFAULT_SOURCE)


def _default_collections_path() -> str:
    configured = os.getenv(COLLECTIONS_ENV_VAR, "").strip()
    if configured:
        return configured
    return str(DEFAULT_COLLECTIONS_PATH)


def _load_registry(path: str | Path) -> CollectionRegistry:
    return CollectionRegistry.load(Path(path))


def _add_collections_path_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--collections-path",
        default=_default_collections_path(),
        help=f"Collection registry path (default: {DEFAULT_COLLECTIONS_PATH} or ${COLLECTIONS_ENV_VAR})",
    )


def _resolve_collection_id(registry: CollectionRegistry, identifier: str) -> str:
    collection_id, prefix = registry.resolve_context_target(identifier)
    if prefix:
        raise RuntimeError(
            "Use a collection name or id here. Path prefixes belong to context commands."
        )
    if collection_id is None:
        raise RuntimeError(f"Collection not found: {identifier}")
    return collection_id


def _resolve_index_paths(
    args: argparse.Namespace,
    collection_cfg=None,
) -> tuple[Path, Path]:
    db_arg = getattr(args, "db_path", None)
    faiss_arg = getattr(args, "faiss_path", None)
    if collection_cfg is not None:
        collection_db, collection_faiss = collection_cfg.index_paths()
        db_path = Path(db_arg) if db_arg is not None else collection_db
        faiss_path = Path(faiss_arg) if faiss_arg is not None else collection_faiss
        return (db_path, faiss_path)
    db_default = Path(db_arg) if db_arg is not None else DEFAULT_DB_PATH
    faiss_default = Path(faiss_arg) if faiss_arg is not None else DEFAULT_FAISS_PATH
    return (db_default, faiss_default)


def cmd_ingest(args: argparse.Namespace) -> int:
    collections_path = Path(args.collections_path)
    registry = _load_registry(collections_path)
    source_value = getattr(args, "source", None)
    source = Path(source_value) if source_value else Path(_default_source())
    collection_arg = None
    collection_cfg = None
    if args.collection:
        collection_id = _resolve_collection_id(registry, args.collection)
        collection_cfg = registry.find_collection(collection_id)
        source = collection_cfg.root_path_resolved()
        collection_arg = collection_cfg.collection_id
    db_path, faiss_path = _resolve_index_paths(args, collection_cfg)
    if collection_cfg is not None:
        registry.update_collection_index_paths(
            collection_cfg.collection_id,
            db_path=db_path,
            faiss_path=faiss_path,
        )

    model = _resolve_model(args.model, args.use_local_embedding)
    stats = ingest(
        source=source,
        db_path=db_path,
        faiss_path=faiss_path,
        api_key=_api_key(args.use_local_embedding),
        model=model,
        use_local_embedding=args.use_local_embedding,
        rebuild=args.rebuild,
        collections_path=collections_path,
        collection=collection_arg,
    )
    print(
        "Ingest complete: "
        f"docs={stats.documents}, chunks={stats.chunks}, dim={stats.embedding_dim}"
    )
    print(f"Embedding: provider={stats.embedding_provider}, model={stats.embedding_model}")
    print(
        "Collection: "
        f"{stats.collection_name} ({stats.collection_id})"
    )
    print(
        "Delta: "
        f"updated_docs={stats.updated_documents}, "
        f"deleted_docs={stats.deleted_documents}, "
        f"new_embeddings={stats.new_embedding_hashes}, "
        f"reused_embeddings={stats.reused_embedding_hashes}"
    )
    print(f"Source: {stats.source}")
    print(f"SQLite: {db_path}")
    print(f"FAISS: {faiss_path}")
    return 0


def cmd_query(args: argparse.Namespace) -> int:
    collection_cfg = None
    if args.collection:
        registry = _load_registry(Path(args.collections_path))
        collection_cfg = registry.find_collection(_resolve_collection_id(registry, args.collection))
    db_path, faiss_path = _resolve_index_paths(args, collection_cfg)
    model = _resolve_model(args.model, args.use_local_embedding)
    results = search(
        query=args.query,
        db_path=db_path,
        faiss_path=faiss_path,
        api_key=_api_key(args.use_local_embedding),
        model=model,
        use_local_embedding=args.use_local_embedding,
        top_k=args.top_k,
        collections_path=Path(args.collections_path),
        collection=args.collection,
    )

    print(json.dumps(_query_payload(results, show_chunk_type=args.show_chunk_type), ensure_ascii=False, indent=2))
    return 0


def _query_payload(results: list[SearchResult], *, show_chunk_type: bool) -> list[dict[str, object]]:
    return [_serialize_query_result(item, show_chunk_type=show_chunk_type) for item in results]


def _serialize_query_result(item: SearchResult, *, show_chunk_type: bool) -> dict[str, object]:
    payload: dict[str, object] = {
        "doc_id": _docid_for_result(item),
        "score": round(item.fusion_score, 4),
        "file": _result_file_uri(item),
        "title": item.title,
        "snippet": _result_snippet(item.text),
    }
    if show_chunk_type:
        payload["chunk_type"] = item.chunk_type
    return payload


def _docid_for_result(item: SearchResult) -> str:
    return item.doc_id


def _result_file_uri(item: SearchResult) -> str:
    collection = item.collection_name.strip() or item.collection_id
    return f"qmd://{collection}/{item.relative_path}"


def _result_snippet(text: str) -> str:
    return text.strip()


def cmd_eval(args: argparse.Namespace) -> int:
    collection_cfg = None
    if args.collection:
        registry = _load_registry(Path(args.collections_path))
        collection_cfg = registry.find_collection(_resolve_collection_id(registry, args.collection))
    db_path, faiss_path = _resolve_index_paths(args, collection_cfg)
    model = _resolve_model(args.model, args.use_local_embedding)
    stats, details = evaluate(
        golden_path=Path(args.golden),
        db_path=db_path,
        faiss_path=faiss_path,
        api_key=_api_key(args.use_local_embedding),
        model=model,
        use_local_embedding=args.use_local_embedding,
        collections_path=Path(args.collections_path),
        collection=args.collection,
    )

    print(f"Queries: {stats.queries}")
    print(f"Recall@5: {stats.recall_at_5:.4f}")
    print(f"MRR@10: {stats.mrr_at_10:.4f}")
    print(f"nDCG@10: {stats.ndcg_at_10:.4f}")

    if args.verbose:
        print("\nDetails:")
        for item in details:
            print(f"- Query: {item['query']}")
            print(
                "  "
                f"Recall@5={item['recall_at_5']:.4f}, "
                f"RR@10={item['rr_at_10']:.4f}, "
                f"nDCG@10={item['ndcg_at_10']:.4f}"
            )
            for idx, doc in enumerate(item["top_results"], start=1):
                print(f"    {idx}. {doc}")
    return 0


def cmd_collection_add(args: argparse.Namespace) -> int:
    registry = _load_registry(args.collections_path)
    default_db_path, default_faiss_path = default_index_paths(args.name)
    collection = registry.add_collection(
        name=args.name,
        root_path=Path(args.root_path),
        db_path=getattr(args, "db_path", None) or default_db_path,
        faiss_path=getattr(args, "faiss_path", None) or default_faiss_path,
        mask=args.mask,
        include_by_default=args.include_by_default,
    )
    print(
        f"Added collection: {collection.name} ({collection.collection_id}) "
        f"root={collection.root_path} db={collection.db_path} faiss={collection.faiss_path} mask={collection.mask}"
    )
    return 0


def cmd_collection_list(args: argparse.Namespace) -> int:
    registry = _load_registry(args.collections_path)
    collections = registry.list_collections()
    if not collections:
        print("No collections.")
        return 0
    for item in collections:
        default_flag = "default" if item.include_by_default else "opt-in"
        print(
            f"- {item.name} ({item.collection_id}) "
            f"root={item.root_path} db={item.db_path} faiss={item.faiss_path} mask={item.mask} {default_flag}"
        )
    return 0


def cmd_collection_rename(args: argparse.Namespace) -> int:
    registry = _load_registry(args.collections_path)
    collection = registry.rename_collection(args.target, args.new_name)
    print(f"Renamed collection: {collection.collection_id} -> {collection.name}")
    return 0


def cmd_collection_remove(args: argparse.Namespace) -> int:
    registry = _load_registry(args.collections_path)
    removed = registry.remove_collection(args.target)
    print(f"Removed collection: {removed.name} ({removed.collection_id})")
    return 0


def _context_text_from_args(args: argparse.Namespace) -> str:
    if args.file is not None:
        return Path(args.file).read_text(encoding="utf-8")
    if args.text:
        return args.text
    raise RuntimeError("Context text is required.")


def cmd_context_add(args: argparse.Namespace) -> int:
    registry = _load_registry(args.collections_path)
    text = _context_text_from_args(args)
    entry = registry.add_context(
        target=args.target,
        path_prefix="",
        text=text,
    )
    if entry.collection_id is None:
        target_label = "/"
    else:
        target_label = registry.collection_uri(entry.collection_id, entry.path_prefix)
    print(f"Saved context: {target_label}")
    return 0


def cmd_context_list(args: argparse.Namespace) -> int:
    registry = _load_registry(args.collections_path)
    entries = registry.list_contexts(args.target)
    if not entries:
        print("No contexts.")
        return 0
    for entry in entries:
        if entry.collection_id is None:
            target_label = "/"
        else:
            target_label = registry.collection_uri(entry.collection_id, entry.path_prefix)
        print(f"- {target_label}: {snippet(entry.text, 120)}")
    return 0


def cmd_context_rm(args: argparse.Namespace) -> int:
    registry = _load_registry(args.collections_path)
    removed = registry.remove_context(args.target, "")
    if removed.collection_id is None:
        target_label = "/"
    else:
        target_label = registry.collection_uri(removed.collection_id, removed.path_prefix)
    print(f"Removed context: {target_label}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="semsearch", description="Semantic search for markdown cards")
    sub = parser.add_subparsers(dest="command", required=True)

    p_ingest = sub.add_parser("ingest", help="Ingest markdown cards and build indexes")
    p_ingest.add_argument(
        "--source",
        default=argparse.SUPPRESS,
        help=argparse.SUPPRESS,
    )
    p_ingest.add_argument("--db-path", default=argparse.SUPPRESS)
    p_ingest.add_argument("--faiss-path", default=argparse.SUPPRESS)
    p_ingest.add_argument(
        "--collections-path",
        default=_default_collections_path(),
        help=f"Collection registry path (default: {DEFAULT_COLLECTIONS_PATH} or ${COLLECTIONS_ENV_VAR})",
    )
    p_ingest.add_argument(
        "--collection",
        help="Collection name or id to ingest. Preferred path for new workflows.",
    )
    p_ingest.add_argument(
        "--model",
        help=(
            "Embedding model. Defaults to "
            f"{DEFAULT_OPENROUTER_MODEL} (OpenRouter) or {DEFAULT_OLLAMA_MODEL} (with --use-local-embedding)."
        ),
    )
    p_ingest.add_argument("--use-local-embedding", action="store_true")
    p_ingest.add_argument("--rebuild", action="store_true")
    p_ingest.set_defaults(func=cmd_ingest)

    p_query = sub.add_parser("query", help="Query indexed markdown chunks")
    p_query.add_argument("query")
    p_query.add_argument("--top-k", type=int, default=8)
    p_query.add_argument("--show-chunk-type", action="store_true")
    p_query.add_argument("--db-path", default=argparse.SUPPRESS)
    p_query.add_argument("--faiss-path", default=argparse.SUPPRESS)
    p_query.add_argument(
        "--collections-path",
        default=_default_collections_path(),
        help=f"Collection registry path (default: {DEFAULT_COLLECTIONS_PATH} or ${COLLECTIONS_ENV_VAR})",
    )
    p_query.add_argument(
        "--collection",
        help="Collection name or id to search. If omitted, include-by-default collections are used.",
    )
    p_query.add_argument(
        "--model",
        help=(
            "Embedding model. Defaults to "
            f"{DEFAULT_OPENROUTER_MODEL} (OpenRouter) or {DEFAULT_OLLAMA_MODEL} (with --use-local-embedding)."
        ),
    )
    p_query.add_argument("--use-local-embedding", action="store_true")
    p_query.set_defaults(func=cmd_query)

    p_eval = sub.add_parser("eval", help="Evaluate search quality with golden set")
    p_eval.add_argument("--golden", default="tests/golden_queries.yaml")
    p_eval.add_argument("--db-path", default=argparse.SUPPRESS)
    p_eval.add_argument("--faiss-path", default=argparse.SUPPRESS)
    p_eval.add_argument(
        "--collections-path",
        default=_default_collections_path(),
        help=f"Collection registry path (default: {DEFAULT_COLLECTIONS_PATH} or ${COLLECTIONS_ENV_VAR})",
    )
    p_eval.add_argument(
        "--collection",
        help="Collection name or id to evaluate. If omitted, include-by-default collections are used.",
    )
    p_eval.add_argument(
        "--model",
        help=(
            "Embedding model. Defaults to "
            f"{DEFAULT_OPENROUTER_MODEL} (OpenRouter) or {DEFAULT_OLLAMA_MODEL} (with --use-local-embedding)."
        ),
    )
    p_eval.add_argument("--use-local-embedding", action="store_true")
    p_eval.add_argument("--verbose", action="store_true")
    p_eval.set_defaults(func=cmd_eval)

    p_collection = sub.add_parser("collection", help="Manage collection registry")
    _add_collections_path_arg(p_collection)
    collection_sub = p_collection.add_subparsers(dest="collection_command", required=True)

    p_collection_add = collection_sub.add_parser("add", help="Add a collection")
    _add_collections_path_arg(p_collection_add)
    p_collection_add.add_argument("name")
    p_collection_add.add_argument("root_path")
    p_collection_add.add_argument("--db-path", default=argparse.SUPPRESS)
    p_collection_add.add_argument("--faiss-path", default=argparse.SUPPRESS)
    p_collection_add.add_argument("--mask", default="*.md")
    p_collection_add.add_argument(
        "--include-by-default",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    p_collection_add.set_defaults(func=cmd_collection_add)

    p_collection_list = collection_sub.add_parser("list", help="List collections")
    _add_collections_path_arg(p_collection_list)
    p_collection_list.set_defaults(func=cmd_collection_list)

    p_collection_rename = collection_sub.add_parser("rename", help="Rename a collection")
    _add_collections_path_arg(p_collection_rename)
    p_collection_rename.add_argument("target")
    p_collection_rename.add_argument("new_name")
    p_collection_rename.set_defaults(func=cmd_collection_rename)

    p_collection_remove = collection_sub.add_parser("remove", help="Remove a collection")
    _add_collections_path_arg(p_collection_remove)
    p_collection_remove.add_argument("target")
    p_collection_remove.set_defaults(func=cmd_collection_remove)

    p_context = sub.add_parser("context", help="Manage collection contexts")
    _add_collections_path_arg(p_context)
    context_sub = p_context.add_subparsers(dest="context_command", required=True)

    p_context_add = context_sub.add_parser("add", help="Add or replace a context")
    _add_collections_path_arg(p_context_add)
    p_context_add.add_argument("target", help='Use "/" for global context or collection://collection/path')
    text_group = p_context_add.add_mutually_exclusive_group(required=True)
    text_group.add_argument("--text")
    text_group.add_argument("--file")
    p_context_add.set_defaults(func=cmd_context_add)

    p_context_list = context_sub.add_parser("list", help="List contexts")
    _add_collections_path_arg(p_context_list)
    p_context_list.add_argument(
        "target",
        nargs="?",
        help='Optional target scope, for example "/", "notes", or collection://notes/path',
    )
    p_context_list.set_defaults(func=cmd_context_list)

    p_context_rm = context_sub.add_parser("rm", help="Remove a context")
    _add_collections_path_arg(p_context_rm)
    p_context_rm.add_argument("target", help='Use the same target syntax as context add')
    p_context_rm.set_defaults(func=cmd_context_rm)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
