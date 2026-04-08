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
from semsearch.rerankers import (
    CohereReranker,
    DEFAULT_COHERE_RERANKER_MODEL,
    DEFAULT_LOCAL_RERANKER_MODEL,
    DEFAULT_OPENROUTER_RERANKER_MODEL,
    OpenRouterReranker,
    QwenReranker,
    SubprocessReranker,
    resolve_reranker,
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

    def test_resolve_reranker_uses_subprocess_on_darwin(self) -> None:
        with mock.patch("semsearch.rerankers.platform.system", return_value="Darwin"):
            runtime = resolve_reranker(
                use_reranker=True,
                provider="local",
                model=None,
                device="auto",
            )

        self.assertIsNotNone(runtime)
        assert runtime is not None
        self.assertEqual(runtime.model, DEFAULT_LOCAL_RERANKER_MODEL)
        self.assertEqual(runtime.provider, "local-transformers-subprocess")
        self.assertIsInstance(runtime.reranker, SubprocessReranker)

    def test_subprocess_reranker_parses_scores(self) -> None:
        reranker = SubprocessReranker(
            model=DEFAULT_LOCAL_RERANKER_MODEL,
            device="cpu",
            instruction="test",
        )
        completed = mock.Mock(returncode=0, stdout='{"scores":[0.2,0.8]}', stderr="")
        with mock.patch("semsearch.rerankers.subprocess.run", return_value=completed) as run_mock:
            scores = reranker.score("q", ["a", "b"])

        self.assertEqual(scores, [0.2, 0.8])
        run_mock.assert_called_once()

    def test_cohere_reranker_maps_scores_back_to_original_order(self) -> None:
        reranker = CohereReranker(api_key="test-key")
        response = mock.Mock(status_code=200)
        response.json.return_value = {
            "results": [
                {"index": 1, "relevance_score": 0.9},
                {"index": 0, "relevance_score": 0.3},
            ]
        }
        with mock.patch.object(reranker.session, "post", return_value=response) as post_mock:
            scores = reranker.score("query", ["doc-a", "doc-b"])

        self.assertEqual(scores, [0.3, 0.9])
        post_mock.assert_called_once()

    def test_resolve_reranker_supports_cohere_provider(self) -> None:
        runtime = resolve_reranker(
            use_reranker=True,
            provider="cohere",
            model=None,
            device="auto",
            api_key="test-key",
        )

        self.assertIsNotNone(runtime)
        assert runtime is not None
        self.assertEqual(runtime.provider, "cohere")
        self.assertEqual(runtime.model, DEFAULT_COHERE_RERANKER_MODEL)
        self.assertEqual(runtime.device, "remote")

    def test_openrouter_reranker_maps_scores_back_to_original_order(self) -> None:
        reranker = OpenRouterReranker(api_key="test-key")
        response = mock.Mock(status_code=200)
        response.json.return_value = {
            "results": [
                {"index": 1, "relevance_score": 0.95},
                {"index": 0, "relevance_score": 0.25},
            ]
        }
        with mock.patch.object(reranker.session, "post", return_value=response) as post_mock:
            scores = reranker.score("query", ["doc-a", "doc-b"])

        self.assertEqual(scores, [0.25, 0.95])
        post_mock.assert_called_once()

    def test_resolve_reranker_supports_openrouter_provider(self) -> None:
        runtime = resolve_reranker(
            use_reranker=True,
            provider="openrouter",
            model=None,
            device="auto",
            api_key="test-key",
        )

        self.assertIsNotNone(runtime)
        assert runtime is not None
        self.assertEqual(runtime.provider, "openrouter")
        self.assertEqual(runtime.model, DEFAULT_OPENROUTER_RERANKER_MODEL)
        self.assertEqual(runtime.device, "remote")

    def test_qwen_reranker_seq_cls_formats_chat_prompt(self) -> None:
        reranker = QwenReranker(model="tomaarsen/Qwen3-Reranker-0.6B-seq-cls", device="cpu")
        formatted = reranker._format_input("query text", "doc text")

        self.assertIn("<|im_start|>system", formatted)
        self.assertIn('<Instruct>: Given a search query, retrieve relevant passages that answer the query.', formatted)
        self.assertIn("<Query>: query text", formatted)
        self.assertIn("<Document>: doc text", formatted)
        self.assertIn("<|im_start|>assistant", formatted)


if __name__ == "__main__":
    unittest.main()
