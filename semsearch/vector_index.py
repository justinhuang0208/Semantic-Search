from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:  # pragma: no cover - optional dependency
    import faiss  # type: ignore[import-not-found]

    FAISS_AVAILABLE = True
except ModuleNotFoundError:  # pragma: no cover - exercised in this environment
    faiss = None
    FAISS_AVAILABLE = False


class VectorIndexError(RuntimeError):
    pass


_NUMPY_ARCHIVE_MAGIC = b"PK\x03\x04"


@dataclass(slots=True)
class _NumpyIndex:
    ids: np.ndarray
    vectors: np.ndarray
    dim: int


class VectorIndex:
    def __init__(self, index_path: Path) -> None:
        self.index_path = index_path

    def _is_numpy_archive(self) -> bool:
        with self.index_path.open("rb") as handle:
            return handle.read(4) == _NUMPY_ARCHIVE_MAGIC

    def _load_numpy_index(self) -> _NumpyIndex:
        with self.index_path.open("rb") as handle:
            data = np.load(handle, allow_pickle=False)
            try:
                return _NumpyIndex(
                    ids=np.asarray(data["ids"], dtype=np.int64),
                    vectors=np.asarray(data["vectors"], dtype=np.float32),
                    dim=int(np.asarray(data["dim"]).reshape(-1)[0]),
                )
            finally:
                data.close()

    def _search_loaded(self, loaded: faiss.Index | _NumpyIndex, query_vec: np.ndarray, top_k: int) -> list[tuple[int, float]]:
        q = np.asarray(query_vec, dtype=np.float32).reshape(1, -1)
        query_dim = int(q.shape[1])

        if isinstance(loaded, _NumpyIndex):
            index_dim = int(loaded.dim)
            if query_dim != index_dim:
                raise VectorIndexError(
                    f"Query vector dimension {query_dim} does not match FAISS index dimension "
                    f"{index_dim}. Rebuild index with matching embedding provider/model."
                )
            if loaded.vectors.size == 0:
                return []
            scores = loaded.vectors @ q[0]
            order = np.argsort(-scores)[:top_k]
            results: list[tuple[int, float]] = []
            for idx in order:
                results.append((int(loaded.ids[idx]), float(scores[idx])))
            return results

        index_dim = int(loaded.d)
        if query_dim != index_dim:
            raise VectorIndexError(
                f"Query vector dimension {query_dim} does not match FAISS index dimension "
                f"{index_dim}. Rebuild index with matching embedding provider/model."
            )
        scores, ids = loaded.search(q, top_k)
        results: list[tuple[int, float]] = []
        for chunk_id, score in zip(ids[0], scores[0], strict=False):
            if int(chunk_id) == -1:
                continue
            results.append((int(chunk_id), float(score)))
        return results

    def build(self, vectors: list[np.ndarray], ids: list[int], dim: int) -> None:
        if len(vectors) != len(ids):
            raise VectorIndexError("Vector count and id count mismatch.")
        if not vectors:
            raise VectorIndexError("No vectors to index.")

        matrix = np.vstack(vectors).astype(np.float32)
        id_array = np.asarray(ids, dtype=np.int64)

        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        if FAISS_AVAILABLE:
            base = faiss.IndexFlatIP(dim)
            index = faiss.IndexIDMap2(base)
            index.add_with_ids(matrix, id_array)
            faiss.write_index(index, str(self.index_path))
            return

        with self.index_path.open("wb") as handle:
            np.savez_compressed(handle, ids=id_array, vectors=matrix, dim=np.asarray([dim], dtype=np.int64))

    def load(self) -> faiss.Index | _NumpyIndex:
        if not self.index_path.exists():
            raise VectorIndexError(f"FAISS index not found: {self.index_path}")

        if FAISS_AVAILABLE:
            try:
                return faiss.read_index(str(self.index_path))
            except Exception as exc:
                if not self._is_numpy_archive():
                    raise VectorIndexError(
                        f"Unable to load vector index at {self.index_path}. "
                        "The file is not a readable FAISS index for this runtime."
                    ) from exc

        if not self._is_numpy_archive():
            raise VectorIndexError(
                f"Unable to load vector index at {self.index_path}. "
                "This environment does not have faiss installed and the index is stored in native FAISS format. "
                "Install faiss or rebuild the index in numpy fallback format."
            )

        return self._load_numpy_index()

    def search_in_memory(
        self,
        query_vec: np.ndarray,
        *,
        vectors: list[np.ndarray],
        ids: list[int],
        top_k: int,
    ) -> list[tuple[int, float]]:
        if len(vectors) != len(ids):
            raise VectorIndexError("Vector count and id count mismatch.")
        if not vectors:
            return []
        loaded = _NumpyIndex(
            ids=np.asarray(ids, dtype=np.int64),
            vectors=np.vstack(vectors).astype(np.float32),
            dim=int(np.asarray(vectors[0]).shape[0]),
        )
        return self._search_loaded(loaded, query_vec, top_k)

    def search(self, query_vec: np.ndarray, top_k: int) -> list[tuple[int, float]]:
        loaded = self.load()
        return self._search_loaded(loaded, query_vec, top_k)
