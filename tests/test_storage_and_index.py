from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

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


if __name__ == "__main__":
    unittest.main()
