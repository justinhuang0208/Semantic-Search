from __future__ import annotations

from pathlib import Path

import faiss
import numpy as np


class VectorIndexError(RuntimeError):
    pass


class VectorIndex:
    def __init__(self, index_path: Path) -> None:
        self.index_path = index_path

    def build(self, vectors: list[np.ndarray], ids: list[int], dim: int) -> None:
        if len(vectors) != len(ids):
            raise VectorIndexError("Vector count and id count mismatch.")
        if not vectors:
            raise VectorIndexError("No vectors to index.")

        matrix = np.vstack(vectors).astype(np.float32)
        id_array = np.asarray(ids, dtype=np.int64)

        base = faiss.IndexFlatIP(dim)
        index = faiss.IndexIDMap2(base)
        index.add_with_ids(matrix, id_array)

        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(self.index_path))

    def load(self) -> faiss.Index:
        if not self.index_path.exists():
            raise VectorIndexError(f"FAISS index not found: {self.index_path}")
        return faiss.read_index(str(self.index_path))

    def search(self, query_vec: np.ndarray, top_k: int) -> list[tuple[int, float]]:
        index = self.load()
        q = np.asarray(query_vec, dtype=np.float32).reshape(1, -1)
        scores, ids = index.search(q, top_k)
        results: list[tuple[int, float]] = []
        for chunk_id, score in zip(ids[0], scores[0], strict=False):
            if int(chunk_id) == -1:
                continue
            results.append((int(chunk_id), float(score)))
        return results
