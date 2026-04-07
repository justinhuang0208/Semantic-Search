from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from semsearch.storage import Storage

try:
    from semsearch.vector_index import VectorIndex, VectorIndexError

    FAISS_AVAILABLE = True
except ModuleNotFoundError:
    VectorIndex = None  # type: ignore[assignment]
    VectorIndexError = RuntimeError  # type: ignore[assignment]
    FAISS_AVAILABLE = False


class StorageAndIndexTests(unittest.TestCase):
    def test_collection_counts_and_metadata_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "semsearch.db"
            storage = Storage(db_path)
            storage.create_schema()
            storage.conn.execute(
                """
                INSERT INTO documents
                (doc_id, collection_id, collection_name, title, source_path, relative_path,
                 tags_json, out_links_json, updated_at, source_hash, context_hash, document_hash, char_count)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    "doc-1",
                    "notes",
                    "notes",
                    "Doc",
                    "/tmp/notes/doc.md",
                    "doc.md",
                    "[]",
                    "[]",
                    "2026-04-06T12:00:00+08:00",
                    "source-hash",
                    "context-hash",
                    "document-hash",
                    10,
                ),
            )
            storage.conn.execute(
                """
                INSERT INTO chunks
                (chunk_id, doc_id, collection_id, collection_name, title, source_path, relative_path,
                 section_path, chunk_type, context_prefix, context_text, text, search_text, token_count,
                 embedding_hash, tags_json, out_links_json, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    "chunk-1",
                    "doc-1",
                    "notes",
                    "notes",
                    "Doc",
                    "/tmp/notes/doc.md",
                    "doc.md",
                    "Doc",
                    "text",
                    "",
                    "",
                    "hello",
                    "hello",
                    1,
                    "embed-1",
                    "[]",
                    "[]",
                    "2026-04-06T12:00:00+08:00",
                ),
            )
            storage.set_metadata(
                "collection_last_ingested_at:notes",
                "2026-04-06T12:34:56+08:00",
            )
            storage.commit()

            doc_count, chunk_count = storage.collection_counts("notes")
            metadata_value = storage.metadata_value("collection_last_ingested_at:notes")
            storage.close()

        self.assertEqual(doc_count, 1)
        self.assertEqual(chunk_count, 1)
        self.assertEqual(metadata_value, "2026-04-06T12:34:56+08:00")

    def test_embedding_profile_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "semsearch.db"
            storage = Storage(db_path)
            storage.create_schema()
            storage.upsert_embedding_profile(
                provider="openrouter",
                model="google/gemini-embedding-001",
                cache_key="google/gemini-embedding-001",
                dim=768,
            )
            storage.commit()

            profile = storage.embedding_profile()
            storage.close()

        self.assertIsNotNone(profile)
        assert profile is not None
        self.assertEqual(profile["provider"], "openrouter")
        self.assertEqual(profile["model"], "google/gemini-embedding-001")
        self.assertEqual(profile["cache_key"], "google/gemini-embedding-001")
        self.assertEqual(profile["dim"], 768)

    def test_clear_for_rebuild_removes_embedding_profile(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "semsearch.db"
            storage = Storage(db_path)
            storage.create_schema()
            storage.upsert_embedding_profile(
                provider="ollama",
                model="qwen3-embedding:0.6b",
                cache_key="ollama::qwen3-embedding:0.6b",
                dim=1024,
            )
            storage.commit()
            storage.clear_for_rebuild()
            profile = storage.embedding_profile()
            storage.close()

        self.assertIsNone(profile)

    @unittest.skipUnless(FAISS_AVAILABLE, "faiss is not installed")
    def test_vector_index_dimension_mismatch_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            index_path = Path(tmp) / "semsearch.faiss"
            index = VectorIndex(index_path)
            index.build(
                vectors=[np.asarray([1.0, 0.0, 0.0], dtype=np.float32)],
                ids=[1],
                dim=3,
            )

            with self.assertRaisesRegex(VectorIndexError, "dimension"):
                index.search(np.asarray([1.0, 0.0], dtype=np.float32), top_k=3)

    def test_native_faiss_index_without_runtime_raises_clear_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            index_path = Path(tmp) / "semsearch.faiss"
            index_path.write_bytes(b"IxM2\x00\x04\x00\x00fake-faiss")
            index = VectorIndex(index_path)

            with mock.patch("semsearch.vector_index.FAISS_AVAILABLE", False):
                with self.assertRaisesRegex(VectorIndexError, "does not have faiss installed"):
                    index.load()

    def test_search_in_memory_supports_numpy_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            index = VectorIndex(Path(tmp) / "unused.faiss")
            results = index.search_in_memory(
                np.asarray([1.0, 0.0], dtype=np.float32),
                vectors=[
                    np.asarray([1.0, 0.0], dtype=np.float32),
                    np.asarray([0.0, 1.0], dtype=np.float32),
                ],
                ids=[11, 22],
                top_k=2,
            )

        self.assertEqual(results[0][0], 11)
        self.assertGreater(results[0][1], results[1][1])


if __name__ == "__main__":
    unittest.main()
