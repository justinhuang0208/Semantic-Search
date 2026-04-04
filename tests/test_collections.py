from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

from semsearch.collections import CollectionRegistry
from semsearch.embeddings import EmbeddingResponse
from semsearch.pipeline import ingest, search
from semsearch.vector_index import VectorIndexError


class FakeEmbedder:
    def embed_texts(self, texts: list[str], input_type: str | None) -> EmbeddingResponse:
        vectors: list[np.ndarray] = []
        for text in texts:
            lowered = text.lower()
            if "alpha" in lowered:
                vec = np.asarray([1.0, 0.0], dtype=np.float32)
            elif "beta" in lowered:
                vec = np.asarray([0.0, 1.0], dtype=np.float32)
            else:
                vec = np.asarray([0.5, 0.5], dtype=np.float32)
            vectors.append(vec)
        return EmbeddingResponse(vectors=vectors, dim=2)


class CollectionRegistryTests(unittest.TestCase):
    def test_context_matching_uses_global_collection_and_path_specific_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            registry_path = tmp_path / "collections.yml"
            root = tmp_path / "notes"
            root.mkdir()

            registry = CollectionRegistry.load(registry_path)
            collection = registry.add_collection(name="notes", root_path=root)
            registry.add_context(target="/", path_prefix="", text="global")
            registry.add_context(target="collection://notes", path_prefix="", text="collection")
            registry.add_context(target="collection://notes/api", path_prefix="", text="api")

            rendered = registry.render_context_text(collection.collection_id, "api/guide.md")
            self.assertEqual(rendered.split("\n\n"), ["global", "collection", "api"])
            self.assertEqual(
                registry.collection_uri(collection.collection_id, "api/guide.md"),
                "collection://notes/api/guide.md",
            )


class CollectionIngestTests(unittest.TestCase):
    def test_query_respects_collection_filter(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            root_a = tmp_path / "A"
            root_b = tmp_path / "B"
            root_a.mkdir()
            root_b.mkdir()

            (root_a / "topic.md").write_text("# Topic\nalpha banana\n", encoding="utf-8")
            (root_b / "topic.md").write_text("# Topic\nbeta carrot\n", encoding="utf-8")

            registry_path = tmp_path / "collections.yml"
            registry = CollectionRegistry.load(registry_path)
            collection_a = registry.add_collection(name="A", root_path=root_a)
            collection_b = registry.add_collection(name="B", root_path=root_b)

            db_path = tmp_path / "semsearch.db"
            faiss_path = tmp_path / "semsearch.faiss"
            runtime = SimpleNamespace(
                provider="stub",
                model="stub-model",
                cache_key="stub::stub-model",
                embedder=FakeEmbedder(),
            )

            with mock.patch("semsearch.pipeline.resolve_embedder", return_value=runtime):
                ingest(
                    source=root_a,
                    db_path=db_path,
                    faiss_path=faiss_path,
                    api_key=None,
                    model="stub-model",
                    rebuild=True,
                    use_local_embedding=True,
                    collections_path=registry_path,
                    collection=collection_a.collection_id,
                )
                ingest(
                    source=root_b,
                    db_path=db_path,
                    faiss_path=faiss_path,
                    api_key=None,
                    model="stub-model",
                    rebuild=False,
                    use_local_embedding=True,
                    collections_path=registry_path,
                    collection=collection_b.collection_id,
                )

                all_results = search(
                    query="topic",
                    db_path=db_path,
                    faiss_path=faiss_path,
                    api_key=None,
                    model="stub-model",
                    top_k=5,
                    use_local_embedding=True,
                    collections_path=registry_path,
                )
                filtered_results = search(
                    query="topic",
                    db_path=db_path,
                    faiss_path=faiss_path,
                    api_key=None,
                    model="stub-model",
                    top_k=5,
                    use_local_embedding=True,
                    collections_path=registry_path,
                    collection=collection_a.collection_id,
                )

            self.assertEqual({item.collection_id for item in all_results}, {collection_a.collection_id, collection_b.collection_id})
            self.assertEqual({item.collection_id for item in filtered_results}, {collection_a.collection_id})
            self.assertEqual(filtered_results[0].relative_path, "topic.md")

    def test_query_falls_back_to_in_memory_vectors_when_index_load_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            root = tmp_path / "notes"
            root.mkdir()
            (root / "topic.md").write_text("# Topic\nalpha banana\n", encoding="utf-8")

            registry_path = tmp_path / "collections.yml"
            registry = CollectionRegistry.load(registry_path)
            collection = registry.add_collection(name="notes", root_path=root)

            db_path = tmp_path / "semsearch.db"
            faiss_path = tmp_path / "semsearch.faiss"
            runtime = SimpleNamespace(
                provider="stub",
                model="stub-model",
                cache_key="stub::stub-model",
                embedder=FakeEmbedder(),
            )

            with mock.patch("semsearch.pipeline.resolve_embedder", return_value=runtime):
                ingest(
                    source=root,
                    db_path=db_path,
                    faiss_path=faiss_path,
                    api_key=None,
                    model="stub-model",
                    rebuild=True,
                    use_local_embedding=True,
                    collections_path=registry_path,
                    collection=collection.collection_id,
                )
                with mock.patch(
                    "semsearch.pipeline.VectorIndex.search",
                    side_effect=VectorIndexError("broken index"),
                ):
                    results = search(
                        query="alpha",
                        db_path=db_path,
                        faiss_path=faiss_path,
                        api_key=None,
                        model="stub-model",
                        top_k=5,
                        use_local_embedding=True,
                        collections_path=registry_path,
                        collection=collection.collection_id,
                    )

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].relative_path, "topic.md")


if __name__ == "__main__":
    unittest.main()
