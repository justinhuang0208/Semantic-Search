from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

from semsearch.embeddings import DEFAULT_OLLAMA_MODEL, DEFAULT_OPENROUTER_MODEL
from semsearch.models import SearchResult
from semsearch.collections import CollectionRegistry

try:
    from semsearch.cli import (
        _api_key,
        _reranker_api_key,
        _resolve_index_paths,
        _resolve_model,
        _write_status_output,
        build_parser,
    )
    from semsearch.collections import CollectionConfig
    from semsearch.pipeline import last_ingested_at_metadata_key
    from semsearch.storage import Storage

    CLI_IMPORTABLE = True
except ModuleNotFoundError:
    _api_key = None  # type: ignore[assignment]
    _reranker_api_key = None  # type: ignore[assignment]
    _resolve_index_paths = None  # type: ignore[assignment]
    _resolve_model = None  # type: ignore[assignment]
    _write_status_output = None  # type: ignore[assignment]
    CollectionConfig = None  # type: ignore[assignment]
    Storage = None  # type: ignore[assignment]
    last_ingested_at_metadata_key = None  # type: ignore[assignment]
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

    def test_reranker_api_key_required_for_openrouter_provider(self) -> None:
        assert _reranker_api_key is not None
        with mock.patch.dict("os.environ", {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "OPENROUTER_API_KEY"):
                _reranker_api_key(use_reranker=True, provider="openrouter")

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
                "--reranker-provider",
                "openrouter",
                "--use-reranker",
                "--reranker-model",
                "cohere/rerank-v3.5",
                "--rerank-top-k",
                "12",
                "--reranker-device",
                "cpu",
            ]
        )
        args_eval = parser.parse_args(["eval", "--use-local-embedding"])
        args_collection = parser.parse_args(["collection", "list"])
        args_context = parser.parse_args(["context", "add", "/", "--text", "hello"])
        args_status = parser.parse_args(["status"])
        with self.assertRaises(SystemExit):
            parser.parse_args(["search", "hello", "--use-local-embedding"])

        self.assertTrue(args_ingest.use_local_embedding)
        self.assertEqual(args_ingest_legacy.source, "legacy")
        self.assertEqual(args_search.command, "search")
        self.assertEqual(args_search.top_k, 20)
        self.assertFalse(hasattr(args_search, "use_local_embedding"))
        self.assertFalse(hasattr(args_search, "model"))
        self.assertTrue(args_vsearch.use_local_embedding)
        self.assertEqual(args_vsearch.top_k, 20)
        self.assertTrue(args_query.use_local_embedding)
        self.assertTrue(args_query.use_reranker)
        self.assertEqual(args_query.top_k, 20)
        self.assertEqual(args_query.reranker_provider, "openrouter")
        self.assertEqual(args_query.reranker_model, "cohere/rerank-v3.5")
        self.assertEqual(args_query.rerank_top_k, 12)
        self.assertEqual(args_query.reranker_device, "cpu")
        self.assertTrue(args_eval.use_local_embedding)
        self.assertEqual(args_collection.command, "collection")
        self.assertEqual(args_context.command, "context")
        self.assertEqual(args_status.command, "status")

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
                "--reranker-provider",
                "openrouter",
                "--use-reranker",
                "--reranker-model",
                "cohere/rerank-v3.5",
                "--rerank-top-k",
                "16",
                "--reranker-device",
                "mps",
            ]
        )

        with mock.patch("semsearch.cli.search", return_value=[]) as search_mock:
            with mock.patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key"}, clear=True):
                buffer = io.StringIO()
                with redirect_stdout(buffer):
                    exit_code = args.func(args)

        self.assertEqual(exit_code, 0)
        search_mock.assert_called_once()
        kwargs = search_mock.call_args.kwargs
        self.assertTrue(kwargs["use_reranker"])
        self.assertEqual(kwargs["reranker_provider"], "openrouter")
        self.assertEqual(kwargs["reranker_model"], "cohere/rerank-v3.5")
        self.assertEqual(kwargs["reranker_api_key"], "test-key")
        self.assertEqual(kwargs["rerank_top_k"], 16)
        self.assertEqual(kwargs["reranker_device"], "mps")

    def test_status_outputs_collection_counts_and_last_ingested_time(self) -> None:
        assert _write_status_output is not None
        assert Storage is not None
        assert last_ingested_at_metadata_key is not None

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            registry_path = tmp_path / "collections.yml"

            notes_db = tmp_path / "notes.db"
            archive_db = tmp_path / "archive.db"

            notes_storage = Storage(notes_db)
            notes_storage.create_schema()
            notes_storage.conn.execute(
                """
                INSERT INTO documents
                (doc_id, collection_id, collection_name, title, source_path, relative_path,
                 tags_json, out_links_json, updated_at, source_hash, context_hash, document_hash, char_count)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    "doc-1",
                    "notes-id",
                    "notes",
                    "Doc 1",
                    "/tmp/notes/doc-1.md",
                    "doc-1.md",
                    "[]",
                    "[]",
                    "2026-04-06T09:00:00+08:00",
                    "source-1",
                    "context-1",
                    "document-1",
                    10,
                ),
            )
            notes_storage.conn.executemany(
                """
                INSERT INTO chunks
                (chunk_id, doc_id, collection_id, collection_name, title, source_path, relative_path,
                 section_path, chunk_type, context_prefix, context_text, text, search_text, token_count,
                 embedding_hash, tags_json, out_links_json, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        "chunk-1",
                        "doc-1",
                        "notes-id",
                        "notes",
                        "Doc 1",
                        "/tmp/notes/doc-1.md",
                        "doc-1.md",
                        "Doc 1",
                        "text",
                        "",
                        "",
                        "hello",
                        "hello",
                        1,
                        "embed-1",
                        "[]",
                        "[]",
                        "2026-04-06T09:00:00+08:00",
                    ),
                    (
                        "chunk-2",
                        "doc-1",
                        "notes-id",
                        "notes",
                        "Doc 1",
                        "/tmp/notes/doc-1.md",
                        "doc-1.md",
                        "Doc 1",
                        "text",
                        "",
                        "",
                        "world",
                        "world",
                        1,
                        "embed-2",
                        "[]",
                        "[]",
                        "2026-04-06T09:00:00+08:00",
                    ),
                ],
            )
            notes_storage.set_metadata(
                last_ingested_at_metadata_key("notes-id"),
                "2026-04-06T10:11:12+08:00",
            )
            notes_storage.commit()
            notes_storage.close()

            collection_registry = CollectionRegistry.load(registry_path)
            notes_collection = collection_registry.add_collection(
                name="notes",
                root_path=tmp_path / "notes",
                db_path=notes_db,
                faiss_path=tmp_path / "notes.faiss",
            )
            notes_collection.collection_id = "notes-id"
            archive_collection = collection_registry.add_collection(
                name="archive",
                root_path=tmp_path / "archive",
                db_path=archive_db,
                faiss_path=tmp_path / "archive.faiss",
            )
            archive_collection.collection_id = "archive-id"
            collection_registry.save()

            args = mock.Mock()
            args.collections_path = str(registry_path)
            buffer = io.StringIO()

            exit_code = _write_status_output(args, buffer)

        self.assertEqual(exit_code, 0)
        output = buffer.getvalue()
        self.assertIn("Total: collections=2 embedded_files=1 chunks=2", output)
        self.assertIn("- notes (notes-id) embedded_files=1 chunks=2 last_ingested_at=2026-04-06T10:11:12+08:00", output)
        self.assertIn("- archive (archive-id) embedded_files=0 chunks=0 last_ingested_at=-", output)


if __name__ == "__main__":
    unittest.main()
