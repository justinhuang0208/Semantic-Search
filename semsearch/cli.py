from __future__ import annotations

import argparse
import os
from pathlib import Path

from .pipeline import evaluate, ingest, search
from .utils import snippet

DEFAULT_SOURCE = Path("1 - Cards")
DEFAULT_DB_PATH = Path("data_index/semsearch.db")
DEFAULT_FAISS_PATH = Path("data_index/semsearch.faiss")
DEFAULT_MODEL = "google/gemini-embedding-001"


def _api_key() -> str:
    api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set.")
    return api_key


def cmd_ingest(args: argparse.Namespace) -> int:
    stats = ingest(
        source=Path(args.source),
        db_path=Path(args.db_path),
        faiss_path=Path(args.faiss_path),
        api_key=_api_key(),
        model=args.model,
        rebuild=args.rebuild,
    )
    print(f"Ingest complete: docs={stats.documents}, chunks={stats.chunks}, dim={stats.embedding_dim}")
    print(f"Source: {stats.source}")
    print(f"SQLite: {args.db_path}")
    print(f"FAISS: {args.faiss_path}")
    return 0


def cmd_query(args: argparse.Namespace) -> int:
    results = search(
        query=args.query,
        db_path=Path(args.db_path),
        faiss_path=Path(args.faiss_path),
        api_key=_api_key(),
        model=args.model,
        top_k=args.top_k,
    )

    if not results:
        print("No results.")
        return 0

    for rank, item in enumerate(results, start=1):
        meta = f"{item.source_path} | {item.section_path}"
        if args.show_chunk_type:
            meta += f" | type={item.chunk_type}"
        print(f"{rank}. score={item.fusion_score:.4f} | {meta}")
        print(f"   {snippet(item.text)}")
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    stats, details = evaluate(
        golden_path=Path(args.golden),
        db_path=Path(args.db_path),
        faiss_path=Path(args.faiss_path),
        api_key=_api_key(),
        model=args.model,
    )

    print(f"Queries: {stats.queries}")
    print(f"Recall@5: {stats.recall_at_5:.4f}")
    print(f"MRR@10: {stats.mrr_at_10:.4f}")
    print(f"nDCG@10: {stats.ndcg_at_10:.4f}")

    if args.verbose:
        print("\nDetails:")
        for item in details:
            print(f"- Query: {item['query']}")
            print(f"  Recall@5={item['recall_at_5']:.4f}, RR@10={item['rr_at_10']:.4f}, nDCG@10={item['ndcg_at_10']:.4f}")
            for idx, doc in enumerate(item["top_results"], start=1):
                print(f"    {idx}. {doc}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="semsearch", description="Semantic search for markdown cards")
    sub = parser.add_subparsers(dest="command", required=True)

    p_ingest = sub.add_parser("ingest", help="Ingest markdown cards and build indexes")
    p_ingest.add_argument("--source", default=str(DEFAULT_SOURCE))
    p_ingest.add_argument("--db-path", default=str(DEFAULT_DB_PATH))
    p_ingest.add_argument("--faiss-path", default=str(DEFAULT_FAISS_PATH))
    p_ingest.add_argument("--model", default=DEFAULT_MODEL)
    p_ingest.add_argument("--rebuild", action="store_true")
    p_ingest.set_defaults(func=cmd_ingest)

    p_query = sub.add_parser("query", help="Query indexed markdown chunks")
    p_query.add_argument("query")
    p_query.add_argument("--top-k", type=int, default=8)
    p_query.add_argument("--show-chunk-type", action="store_true")
    p_query.add_argument("--db-path", default=str(DEFAULT_DB_PATH))
    p_query.add_argument("--faiss-path", default=str(DEFAULT_FAISS_PATH))
    p_query.add_argument("--model", default=DEFAULT_MODEL)
    p_query.set_defaults(func=cmd_query)

    p_eval = sub.add_parser("eval", help="Evaluate search quality with golden set")
    p_eval.add_argument("--golden", default="tests/golden_queries.yaml")
    p_eval.add_argument("--db-path", default=str(DEFAULT_DB_PATH))
    p_eval.add_argument("--faiss-path", default=str(DEFAULT_FAISS_PATH))
    p_eval.add_argument("--model", default=DEFAULT_MODEL)
    p_eval.add_argument("--verbose", action="store_true")
    p_eval.set_defaults(func=cmd_eval)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
