from __future__ import annotations

import unittest
from unittest import mock

from semsearch.embeddings import DEFAULT_OLLAMA_MODEL, DEFAULT_OPENROUTER_MODEL

try:
    from semsearch.cli import _api_key, _resolve_model, build_parser

    CLI_IMPORTABLE = True
except ModuleNotFoundError:
    _api_key = None  # type: ignore[assignment]
    _resolve_model = None  # type: ignore[assignment]
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

    def test_parser_accepts_use_local_embedding_for_all_commands(self) -> None:
        assert build_parser is not None
        parser = build_parser()

        args_ingest = parser.parse_args(["ingest", "--use-local-embedding"])
        args_query = parser.parse_args(["query", "hello", "--use-local-embedding"])
        args_eval = parser.parse_args(["eval", "--use-local-embedding"])

        self.assertTrue(args_ingest.use_local_embedding)
        self.assertTrue(args_query.use_local_embedding)
        self.assertTrue(args_eval.use_local_embedding)


if __name__ == "__main__":
    unittest.main()
