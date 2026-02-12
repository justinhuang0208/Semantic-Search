from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import requests


class EmbeddingError(RuntimeError):
    pass


@dataclass(slots=True)
class EmbeddingResponse:
    vectors: list[np.ndarray]
    dim: int


class OpenRouterEmbedder:
    def __init__(
        self,
        api_key: str,
        model: str = "google/gemini-embedding-001",
        endpoint: str = "https://openrouter.ai/api/v1/embeddings",
        batch_size: int = 16,
        timeout: int = 60,
        max_retries: int = 4,
    ) -> None:
        if not api_key:
            raise EmbeddingError("OPENROUTER_API_KEY is required.")
        self.api_key = api_key
        self.model = model
        self.endpoint = endpoint
        self.batch_size = batch_size
        self.timeout = timeout
        self.max_retries = max_retries
        self.session = requests.Session()

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _post(self, payload: dict) -> dict:
        for attempt in range(self.max_retries + 1):
            response = self.session.post(
                self.endpoint,
                headers=self._headers(),
                json=payload,
                timeout=self.timeout,
            )

            if response.status_code == 200:
                return response.json()

            should_retry = response.status_code == 429 or response.status_code >= 500
            if should_retry and attempt < self.max_retries:
                time.sleep(2**attempt)
                continue

            raise EmbeddingError(
                f"Embedding request failed: HTTP {response.status_code} {response.text}"
            )

        raise EmbeddingError("Embedding request failed after retries.")

    def _parse_vectors(self, data: dict) -> list[np.ndarray]:
        items = data.get("data")
        if not isinstance(items, list) or not items:
            raise EmbeddingError("Invalid embedding response: missing data list.")

        items_sorted = sorted(items, key=lambda x: x.get("index", 0))
        vectors: list[np.ndarray] = []
        for item in items_sorted:
            emb = item.get("embedding")
            if not isinstance(emb, list) or not emb:
                raise EmbeddingError("Invalid embedding response: empty embedding vector.")
            vec = np.asarray(emb, dtype=np.float32)
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            vectors.append(vec)
        return vectors

    def _embed_batch(self, texts: list[str], input_type: str | None) -> list[np.ndarray]:
        payload = {
            "model": self.model,
            "input": texts,
            "encoding_format": "float",
        }
        if input_type:
            payload["input_type"] = input_type

        try:
            data = self._post(payload)
            return self._parse_vectors(data)
        except EmbeddingError as err:
            if input_type and "input_type" in str(err).lower():
                fallback_payload = {
                    "model": self.model,
                    "input": texts,
                    "encoding_format": "float",
                }
                data = self._post(fallback_payload)
                return self._parse_vectors(data)
            raise

    def embed_texts(self, texts: list[str], input_type: str | None) -> EmbeddingResponse:
        if not texts:
            return EmbeddingResponse(vectors=[], dim=0)

        vectors: list[np.ndarray] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            vectors.extend(self._embed_batch(batch, input_type=input_type))

        dim = len(vectors[0])
        for vec in vectors:
            if len(vec) != dim:
                raise EmbeddingError("Embedding dimension mismatch in response.")

        return EmbeddingResponse(vectors=vectors, dim=dim)
