from __future__ import annotations

import io
import json
import unittest
from contextlib import redirect_stdout
from unittest import mock

from semsearch.embeddings import DEFAULT_OLLAMA_MODEL, DEFAULT_OPENROUTER_MODEL
from semsearch.models import SearchResult

try:
    from semsearch.cli import _api_key, _resolve_index_paths, _resolve_model, build_parser
    from semsearch.collections import CollectionConfig

    CLI_IMPORTABLE = True
except ModuleNotFoundError:
    _api_key = None  # type: ignore[assignment]
    _resolve_index_paths = None  # type: ignore[assignment]
    _resolve_model = None  # type: ignore[assignment]
    CollectionConfig = None  # type: ignore[assignment]
    build_parser = None  # type: ignore[assignment]
    CLI_IMPORTABLE = False


@unittest.skipUnless(CLI_IMPORTABLE, "runtime dependencies for semsearch.cli are not installed")
class CliFlagsTests(unittest.TestCase):
    def test_resolve_model_defaults(self) -> None:
        assert _resolve_model is not None
        self.assertEqual(_resolve_model(None, use_local_embedding=False), DEFAULT_OPENROUTER_MODEL)
        self.assertEqual(_resolve_model(None, use_local_embedding=True), DEFAULT_OLLAMA_MODEL)

    def test_api_key_not_required_for_local_embedding(self) -> None:
        assert _api_key is not None
        with mock.patch.dict("os.environ", {}, clear=True):
            self.assertIsNone(_api_key(use_local_embedding=True))

    def test_api_key_required_for_openrouter_embedding(self) -> None:
        assert _api_key is not None
        with mock.patch.dict("os.environ", {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "OPENROUTER_API_KEY"):
                _api_key(use_local_embedding=False)

    def test_parser_routes_embedding_flags_by_command(self) -> None:
        assert build_parser is not None
        parser = build_parser()

        args_ingest = parser.parse_args(["ingest", "--use-local-embedding"])
        args_ingest_legacy = parser.parse_args(["ingest", "--source", "legacy"])
        args_search = parser.parse_args(["search", "hello"])
        args_vsearch = parser.parse_args(["vsearch", "hello", "--use-local-embedding"])
        args_query = parser.parse_args(
            [
                "query",
                "hello",
                "--use-local-embedding",
                "--use-reranker",
                "--reranker-model",
                "tomaarsen/Qwen3-Reranker-0.6B-seq-cls",
                "--rerank-top-k",
                "12",
                "--reranker-device",
                "cpu",
            ]
        )
        args_eval = parser.parse_args(["eval", "--use-local-embedding"])
        args_collection = parser.parse_args(["collection", "list"])
        args_context = parser.parse_args(["context", "add", "/", "--text", "hello"])
        with self.assertRaises(SystemExit):
            parser.parse_args(["search", "hello", "--use-local-embedding"])

        self.assertTrue(args_ingest.use_local_embedding)
        self.assertEqual(args_ingest_legacy.source, "legacy")
        self.assertEqual(args_search.command, "search")
        self.assertFalse(hasattr(args_search, "use_local_embedding"))
        self.assertFalse(hasattr(args_search, "model"))
        self.assertTrue(args_vsearch.use_local_embedding)
        self.assertTrue(args_query.use_local_embedding)
        self.assertTrue(args_query.use_reranker)
        self.assertEqual(args_query.reranker_model, "tomaarsen/Qwen3-Reranker-0.6B-seq-cls")
        self.assertEqual(args_query.rerank_top_k, 12)
        self.assertEqual(args_query.reranker_device, "cpu")
        self.assertTrue(args_eval.use_local_embedding)
        self.assertEqual(args_collection.command, "collection")
        self.assertEqual(args_context.command, "context")

    def test_index_paths_default_to_collection_paths(self) -> None:
        assert _resolve_index_paths is not None
        assert CollectionConfig is not None
        collection = CollectionConfig(
            collection_id="notes",
            name="notes",
            root_path="/tmp/notes",
            db_path="data_index/notes.db",
            faiss_path="data_index/notes.faiss",
        )
        args = mock.Mock()
        args.db_path = None
        args.faiss_path = None

        db_path, faiss_path = _resolve_index_paths(args, collection)

        self.assertEqual(str(db_path), "data_index/notes.db")
        self.assertEqual(str(faiss_path), "data_index/notes.faiss")

    def test_search_outputs_structured_json(self) -> None:
        assert build_parser is not None
        parser = build_parser()
        result = SearchResult(
            chunk_rowid=1,
            chunk_id="test::new.md::text::0",
            doc_id="test::new.md",
            collection_id="test",
            collection_name="test",
            title="new",
            source_path="/tmp/test/new.md",
            relative_path="new.md",
            section_path="new",
            chunk_type="text",
            text="hello world",
            fusion_score=0.87654,
            rerank_score=None,
            final_score=0.87654,
            vector_rank=1,
            bm25_rank=1,
        )
        args = parser.parse_args(["search", "hello"])

        with mock.patch("semsearch.cli.search", return_value=[result]):
            buffer = io.StringIO()
            with redirect_stdout(buffer):
                exit_code = args.func(args)

        self.assertEqual(exit_code, 0)
        self.assertEqual(
            json.loads(buffer.getvalue()),
            [
                {
                    "doc_id": "test::new.md",
                    "score": 0.8765,
                    "file": "qmd://test/new.md",
                    "title": "new",
                    "snippet": "hello world",
                }
            ],
        )

    def test_vsearch_outputs_structured_json(self) -> None:
        assert build_parser is not None
        parser = build_parser()
        result = SearchResult(
            chunk_rowid=1,
            chunk_id="test::new.md::text::0",
            doc_id="test::new.md",
            collection_id="test",
            collection_name="test",
            title="new",
            source_path="/tmp/test/new.md",
            relative_path="new.md",
            section_path="new",
            chunk_type="text",
            text="hello world",
            fusion_score=0.87654,
            rerank_score=None,
            final_score=0.87654,
            vector_rank=1,
            bm25_rank=1,
        )
        args = parser.parse_args(["vsearch", "hello", "--use-local-embedding"])

        with mock.patch("semsearch.cli.search", return_value=[result]):
            buffer = io.StringIO()
            with redirect_stdout(buffer):
                exit_code = args.func(args)

        self.assertEqual(exit_code, 0)
        self.assertEqual(
            json.loads(buffer.getvalue()),
            [
                {
                    "doc_id": "test::new.md",
                    "score": 0.8765,
                    "file": "qmd://test/new.md",
                    "title": "new",
                    "snippet": "hello world",
                }
            ],
        )

    def test_query_can_include_chunk_type_in_json(self) -> None:
        assert build_parser is not None
        parser = build_parser()
        result = SearchResult(
            chunk_rowid=1,
            chunk_id="test::new.md::code::0",
            doc_id="test::new.md",
            collection_id="test",
            collection_name="test",
            title="new",
            source_path="/tmp/test/new.md",
            relative_path="new.md",
            section_path="new",
            chunk_type="code",
            text="print('hello')",
            fusion_score=0.5,
            rerank_score=0.9321,
            final_score=0.9321,
            vector_rank=1,
            bm25_rank=2,
        )
        args = parser.parse_args(["query", "hello", "--use-local-embedding", "--show-chunk-type"])

        with mock.patch("semsearch.cli.search", return_value=[result]):
            buffer = io.StringIO()
            with redirect_stdout(buffer):
                exit_code = args.func(args)

        self.assertEqual(exit_code, 0)
        payload = json.loads(buffer.getvalue())
        self.assertEqual(payload[0]["chunk_type"], "code")

    def test_search_modes_route_to_pipeline(self) -> None:
        assert build_parser is not None
        parser = build_parser()
        result = SearchResult(
            chunk_rowid=1,
            chunk_id="test::new.md::text::0",
            doc_id="test::new.md",
            collection_id="test",
            collection_name="test",
            title="new",
            source_path="/tmp/test/new.md",
            relative_path="new.md",
            section_path="new",
            chunk_type="text",
            text="hello world",
            fusion_score=0.87654,
            rerank_score=None,
            final_score=0.87654,
            vector_rank=1,
            bm25_rank=1,
        )

        cases = [
            (["search", "hello"], "fulltext", None),
            (["vsearch", "hello", "--use-local-embedding"], "vector", None),
            (["query", "hello", "--use-local-embedding"], "hybrid", None),
        ]

        for argv, expected_mode, expected_api_key in cases:
            with self.subTest(argv=argv):
                args = parser.parse_args(argv)
                with mock.patch("semsearch.cli.search", return_value=[result]) as search_mock:
                    buffer = io.StringIO()
                    with redirect_stdout(buffer):
                        exit_code = args.func(args)

                self.assertEqual(exit_code, 0)
                search_mock.assert_called_once()
                self.assertEqual(search_mock.call_args.kwargs["search_mode"], expected_mode)
                self.assertEqual(search_mock.call_args.kwargs["api_key"], expected_api_key)

    def test_query_passes_reranker_options_to_pipeline(self) -> None:
        assert build_parser is not None
        parser = build_parser()
        args = parser.parse_args(
            [
                "query",
                "hello",
                "--use-local-embedding",
                "--use-reranker",
                "--reranker-model",
                "tomaarsen/Qwen3-Reranker-0.6B-seq-cls",
                "--rerank-top-k",
                "16",
                "--reranker-device",
                "mps",
            ]
        )

        with mock.patch("semsearch.cli.search", return_value=[]) as search_mock:
            buffer = io.StringIO()
            with redirect_stdout(buffer):
                exit_code = args.func(args)

        self.assertEqual(exit_code, 0)
        search_mock.assert_called_once()
        kwargs = search_mock.call_args.kwargs
        self.assertTrue(kwargs["use_reranker"])
        self.assertEqual(kwargs["reranker_model"], "tomaarsen/Qwen3-Reranker-0.6B-seq-cls")
        self.assertEqual(kwargs["rerank_top_k"], 16)
        self.assertEqual(kwargs["reranker_device"], "mps")


if __name__ == "__main__":
    unittest.main()
