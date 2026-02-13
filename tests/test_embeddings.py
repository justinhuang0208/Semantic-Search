from __future__ import annotations

import unittest
from unittest import mock

import numpy as np
import requests

from semsearch.embeddings import (
    DEFAULT_OLLAMA_MODEL,
    DEFAULT_OPENROUTER_MODEL,
    EmbeddingError,
    OllamaEmbedder,
    resolve_embedder,
)


class EmbeddingsTests(unittest.TestCase):
    def test_resolve_embedder_openrouter_defaults(self) -> None:
        runtime = resolve_embedder(
            use_local_embedding=False,
            model=None,
            api_key="test-key",
        )
        self.assertEqual(runtime.provider, "openrouter")
        self.assertEqual(runtime.model, DEFAULT_OPENROUTER_MODEL)
        self.assertEqual(runtime.cache_key, DEFAULT_OPENROUTER_MODEL)

    def test_resolve_embedder_ollama_defaults(self) -> None:
        runtime = resolve_embedder(
            use_local_embedding=True,
            model=None,
            api_key=None,
        )
        self.assertEqual(runtime.provider, "ollama")
        self.assertEqual(runtime.model, DEFAULT_OLLAMA_MODEL)
        self.assertEqual(runtime.cache_key, f"ollama::{DEFAULT_OLLAMA_MODEL}")

    def test_ollama_embed_texts_parses_and_normalizes_vectors(self) -> None:
        embedder = OllamaEmbedder(model="qwen3-embedding:0.6b")
        with mock.patch.object(
            embedder,
            "_post",
            return_value={"embeddings": [[3.0, 4.0], [0.0, 2.0]]},
        ):
            response = embedder.embed_texts(["a", "b"], input_type="document")

        self.assertEqual(response.dim, 2)
        self.assertEqual(len(response.vectors), 2)
        self.assertTrue(np.allclose(response.vectors[0], np.asarray([0.6, 0.8], dtype=np.float32)))
        self.assertTrue(np.allclose(response.vectors[1], np.asarray([0.0, 1.0], dtype=np.float32)))

    def test_ollama_connection_error_has_actionable_message(self) -> None:
        embedder = OllamaEmbedder(max_retries=0)
        with mock.patch.object(
            embedder.session,
            "post",
            side_effect=requests.ConnectionError("boom"),
        ):
            with self.assertRaisesRegex(EmbeddingError, "ollama serve"):
                embedder.embed_texts(["hello"], input_type="document")


if __name__ == "__main__":
    unittest.main()
