from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import requests

DEFAULT_OPENROUTER_MODEL = "google/gemini-embedding-001"
DEFAULT_OLLAMA_MODEL = "qwen3-embedding:0.6b"


class EmbeddingError(RuntimeError):
    pass


@dataclass(slots=True)
class EmbeddingResponse:
    vectors: list[np.ndarray]
    dim: int


@dataclass(slots=True)
class EmbeddingRuntime:
    provider: str
    model: str
    cache_key: str
    embedder: "Embedder"


class Embedder(Protocol):
    def embed_texts(
        self, texts: list[str], input_type: str | None
    ) -> EmbeddingResponse:
        pass


class OpenRouterEmbedder:
    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_OPENROUTER_MODEL,
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
                raise EmbeddingError(
                    "Invalid embedding response: empty embedding vector."
                )
            vec = np.asarray(emb, dtype=np.float32)
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            vectors.append(vec)
        return vectors

    def _embed_batch(
        self, texts: list[str], input_type: str | None
    ) -> list[np.ndarray]:
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

    def embed_texts(
        self, texts: list[str], input_type: str | None
    ) -> EmbeddingResponse:
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


class OllamaEmbedder:
    def __init__(
        self,
        model: str = DEFAULT_OLLAMA_MODEL,
        endpoint: str = "http://localhost:11434/api/embed",
        batch_size: int = 16,
        timeout: int = 60,
        max_retries: int = 2,
    ) -> None:
        self.model = model
        self.endpoint = endpoint
        self.batch_size = batch_size
        self.timeout = timeout
        self.max_retries = max_retries
        self.session = requests.Session()

    def _post(self, payload: dict) -> dict:
        for attempt in range(self.max_retries + 1):
            try:
                response = self.session.post(
                    self.endpoint,
                    json=payload,
                    timeout=self.timeout,
                )
            except requests.RequestException as err:
                if attempt < self.max_retries:
                    time.sleep(2**attempt)
                    continue
                raise EmbeddingError(
                    "Unable to connect to Ollama embedding API at "
                    f"{self.endpoint}. Start Ollama with `ollama serve`."
                ) from err

            if response.status_code == 200:
                return response.json()

            should_retry = response.status_code == 429 or response.status_code >= 500
            if should_retry and attempt < self.max_retries:
                time.sleep(2**attempt)
                continue

            if response.status_code == 404:
                raise EmbeddingError(
                    f"Ollama model not found: {self.model}. "
                    f"Run `ollama pull {self.model}` and retry."
                )

            raise EmbeddingError(
                f"Ollama embedding request failed: HTTP {response.status_code} {response.text}"
            )

        raise EmbeddingError("Ollama embedding request failed after retries.")

    def _parse_vectors(self, data: dict) -> list[np.ndarray]:
        raw_vectors = data.get("embeddings")
        if (
            isinstance(raw_vectors, list)
            and raw_vectors
            and isinstance(raw_vectors[0], (int, float))
        ):
            raw_vectors = [raw_vectors]

        if not isinstance(raw_vectors, list) or not raw_vectors:
            single = data.get("embedding")
            if isinstance(single, list) and single:
                raw_vectors = [single]
            else:
                raise EmbeddingError(
                    "Invalid Ollama embedding response: missing embeddings."
                )

        vectors: list[np.ndarray] = []
        for emb in raw_vectors:
            if not isinstance(emb, list) or not emb:
                raise EmbeddingError(
                    "Invalid Ollama embedding response: empty embedding vector."
                )
            vec = np.asarray(emb, dtype=np.float32)
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            vectors.append(vec)
        return vectors

    def _embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        payload = {
            "model": self.model,
            "input": texts,
        }
        data = self._post(payload)
        return self._parse_vectors(data)

    def embed_texts(
        self, texts: list[str], input_type: str | None
    ) -> EmbeddingResponse:
        del input_type
        if not texts:
            return EmbeddingResponse(vectors=[], dim=0)

        vectors: list[np.ndarray] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            vectors.extend(self._embed_batch(batch))

        dim = len(vectors[0])
        for vec in vectors:
            if len(vec) != dim:
                raise EmbeddingError("Embedding dimension mismatch in response.")

        return EmbeddingResponse(vectors=vectors, dim=dim)


def resolve_embedder(
    *,
    use_local_embedding: bool,
    model: str | None,
    api_key: str | None,
) -> EmbeddingRuntime:
    resolved_model = (model or "").strip()

    if use_local_embedding:
        if not resolved_model:
            resolved_model = DEFAULT_OLLAMA_MODEL
        return EmbeddingRuntime(
            provider="ollama",
            model=resolved_model,
            cache_key=f"ollama::{resolved_model}",
            embedder=OllamaEmbedder(model=resolved_model),
        )

    if not resolved_model:
        resolved_model = DEFAULT_OPENROUTER_MODEL
    return EmbeddingRuntime(
        provider="openrouter",
        model=resolved_model,
        cache_key=resolved_model,
        embedder=OpenRouterEmbedder(api_key=api_key or "", model=resolved_model),
    )
