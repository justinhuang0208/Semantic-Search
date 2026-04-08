from __future__ import annotations

import json
import platform
import subprocess
import sys
from dataclasses import dataclass
from typing import Protocol

import requests


DEFAULT_LOCAL_RERANKER_MODEL = "tomaarsen/Qwen3-Reranker-0.6B-seq-cls"
DEFAULT_COHERE_RERANKER_MODEL = "rerank-v4.0-fast"
DEFAULT_OPENROUTER_RERANKER_MODEL = "cohere/rerank-v3.5"
DEFAULT_RERANKER_INSTRUCTION = (
    "Given a search query, retrieve relevant passages that answer the query."
)
COHERE_RERANK_ENDPOINT = "https://api.cohere.com/v2/rerank"
OPENROUTER_RERANK_ENDPOINT = "https://openrouter.ai/api/v1/rerank"
_SEQ_CLS_PROMPT_PREFIX = (
    '<|im_start|>system\n'
    'Judge whether the Document meets the requirements based on the Query and the Instruct '
    'provided. Note that the answer can only be "yes" or "no".<|im_end|>\n'
    '<|im_start|>user\n'
)
_SEQ_CLS_PROMPT_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


class RerankerError(RuntimeError):
    pass


@dataclass(slots=True)
class RerankerRuntime:
    provider: str
    model: str
    device: str
    reranker: "Reranker"


class Reranker(Protocol):
    def score(self, query: str, documents: list[str]) -> list[float]:
        pass


class QwenReranker:
    def __init__(
        self,
        *,
        model: str = DEFAULT_LOCAL_RERANKER_MODEL,
        device: str = "auto",
        instruction: str = DEFAULT_RERANKER_INSTRUCTION,
        max_length: int = 8192,
    ) -> None:
        self.model_name = model
        self.requested_device = device
        self.instruction = instruction
        self.max_length = max_length
        self._tokenizer = None
        self._model = None
        self._device = None
        self._yes_token_id = None
        self._no_token_id = None
        self._is_sequence_classification = self.model_name.endswith("-seq-cls")

    def _ensure_loaded(self) -> None:
        if self._model is not None and self._tokenizer is not None:
            return
        try:
            import torch
            from transformers import (
                AutoModelForCausalLM,
                AutoModelForSequenceClassification,
                AutoTokenizer,
            )
        except ModuleNotFoundError as err:
            raise RerankerError(
                "Local reranker dependencies are missing. Install torch and transformers>=4.51.0."
            ) from err

        self._device = self._resolve_device(torch)
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            padding_side="left",
        )
        if self._is_sequence_classification:
            self._model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name
            ).eval()
        else:
            self._model = AutoModelForCausalLM.from_pretrained(self.model_name).eval()
        self._model.to(self._device)
        if not self._is_sequence_classification:
            self._yes_token_id = self._tokenizer.convert_tokens_to_ids("yes")
            self._no_token_id = self._tokenizer.convert_tokens_to_ids("no")
            if self._yes_token_id is None or self._no_token_id is None:
                raise RerankerError("Unable to resolve yes/no tokens for reranker scoring.")

    def _resolve_device(self, torch_module) -> str:
        if self.requested_device != "auto":
            return self.requested_device
        mps_backend = getattr(torch_module.backends, "mps", None)
        if mps_backend is not None and torch_module.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _format_input(self, query: str, document: str) -> str:
        if self._is_sequence_classification:
            return (
                f"{_SEQ_CLS_PROMPT_PREFIX}"
                f"<Instruct>: {self.instruction}\n"
                f"<Query>: {query}\n"
                f"<Document>: {document}"
                f"{_SEQ_CLS_PROMPT_SUFFIX}"
            )
        return (
            f"<Instruct>: {self.instruction}\n"
            f"<Query>: {query}\n"
            f"<Document>: {document}"
        )

    @property
    def device(self) -> str:
        if self._device is None:
            self._ensure_loaded()
        assert self._device is not None
        return self._device

    def score(self, query: str, documents: list[str]) -> list[float]:
        if not documents:
            return []
        self._ensure_loaded()

        import torch

        assert self._tokenizer is not None
        assert self._model is not None

        inputs = self._tokenizer(
            [self._format_input(query, document) for document in documents],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )
        moved_inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with torch.no_grad():
            logits = self._model(**moved_inputs).logits
            if self._is_sequence_classification:
                return logits.squeeze(-1).sigmoid().detach().cpu().tolist()
            assert self._yes_token_id is not None
            assert self._no_token_id is not None
            token_logits = logits[:, -1, :]
            yes_no_logits = token_logits[:, [self._no_token_id, self._yes_token_id]]
            probs = torch.softmax(yes_no_logits, dim=1)
        return probs[:, 1].detach().cpu().tolist()


class SubprocessReranker:
    def __init__(
        self,
        *,
        model: str,
        device: str,
        instruction: str,
        max_length: int = 8192,
    ) -> None:
        self.model_name = model
        self.device = device
        self.instruction = instruction
        self.max_length = max_length

    def score(self, query: str, documents: list[str]) -> list[float]:
        if not documents:
            return []
        payload = {
            "model": self.model_name,
            "device": self.device,
            "instruction": self.instruction,
            "max_length": self.max_length,
            "query": query,
            "documents": documents,
        }
        completed = subprocess.run(
            [sys.executable, "-m", "semsearch.reranker_worker"],
            input=json.dumps(payload),
            text=True,
            capture_output=True,
            check=False,
        )
        if completed.returncode != 0:
            message = completed.stderr.strip() or completed.stdout.strip() or "unknown error"
            raise RerankerError(f"Local reranker subprocess failed: {message}")
        try:
            response = json.loads(completed.stdout)
        except json.JSONDecodeError as err:
            raise RerankerError(
                f"Local reranker subprocess returned invalid JSON: {completed.stdout!r}"
            ) from err
        scores = response.get("scores")
        if not isinstance(scores, list):
            raise RerankerError("Local reranker subprocess response is missing scores.")
        return [float(score) for score in scores]


class CohereReranker:
    def __init__(
        self,
        *,
        api_key: str,
        model: str = DEFAULT_COHERE_RERANKER_MODEL,
        endpoint: str = COHERE_RERANK_ENDPOINT,
        timeout: int = 60,
        max_tokens_per_doc: int = 4096,
    ) -> None:
        if not api_key:
            raise RerankerError("COHERE_API_KEY is required for Cohere reranker.")
        self.api_key = api_key
        self.model_name = model
        self.endpoint = endpoint
        self.timeout = timeout
        self.max_tokens_per_doc = max_tokens_per_doc
        self.session = requests.Session()

    def score(self, query: str, documents: list[str]) -> list[float]:
        if not documents:
            return []
        response = self.session.post(
            self.endpoint,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model_name,
                "query": query,
                "documents": documents,
                "top_n": len(documents),
                "max_tokens_per_doc": self.max_tokens_per_doc,
            },
            timeout=self.timeout,
        )
        if response.status_code != 200:
            raise RerankerError(
                f"Cohere rerank request failed: HTTP {response.status_code} {response.text}"
            )
        payload = response.json()
        results = payload.get("results")
        if not isinstance(results, list):
            raise RerankerError("Cohere rerank response is missing results.")
        scores = [0.0] * len(documents)
        for item in results:
            if not isinstance(item, dict):
                continue
            index = item.get("index")
            relevance_score = item.get("relevance_score")
            if not isinstance(index, int) or index < 0 or index >= len(documents):
                continue
            if not isinstance(relevance_score, (int, float)):
                continue
            scores[index] = float(relevance_score)
        return scores


class OpenRouterReranker:
    def __init__(
        self,
        *,
        api_key: str,
        model: str = DEFAULT_OPENROUTER_RERANKER_MODEL,
        endpoint: str = OPENROUTER_RERANK_ENDPOINT,
        timeout: int = 60,
        max_tokens_per_doc: int = 4096,
    ) -> None:
        if not api_key:
            raise RerankerError("OPENROUTER_API_KEY is required for OpenRouter reranker.")
        self.api_key = api_key
        self.model_name = model
        self.endpoint = endpoint
        self.timeout = timeout
        self.max_tokens_per_doc = max_tokens_per_doc
        self.session = requests.Session()

    def score(self, query: str, documents: list[str]) -> list[float]:
        if not documents:
            return []
        response = self.session.post(
            self.endpoint,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model_name,
                "query": query,
                "documents": documents,
                "top_n": len(documents),
                "max_tokens_per_doc": self.max_tokens_per_doc,
            },
            timeout=self.timeout,
        )
        if response.status_code != 200:
            raise RerankerError(
                f"OpenRouter rerank request failed: HTTP {response.status_code} {response.text}"
            )
        payload = response.json()
        results = payload.get("results")
        if not isinstance(results, list):
            raise RerankerError("OpenRouter rerank response is missing results.")
        scores = [0.0] * len(documents)
        for item in results:
            if not isinstance(item, dict):
                continue
            index = item.get("index")
            relevance_score = item.get("relevance_score")
            if not isinstance(index, int) or index < 0 or index >= len(documents):
                continue
            if not isinstance(relevance_score, (int, float)):
                continue
            scores[index] = float(relevance_score)
        return scores


def resolve_reranker(
    *,
    use_reranker: bool,
    provider: str,
    model: str | None,
    device: str,
    api_key: str | None = None,
    instruction: str = DEFAULT_RERANKER_INSTRUCTION,
) -> RerankerRuntime | None:
    if not use_reranker:
        return None
    resolved_provider = provider.strip().lower()
    resolved_model = (model or "").strip()
    if resolved_provider == "cohere":
        reranker = CohereReranker(
            api_key=api_key or "",
            model=resolved_model or DEFAULT_COHERE_RERANKER_MODEL,
        )
        return RerankerRuntime(
            provider="cohere",
            model=reranker.model_name,
            device="remote",
            reranker=reranker,
        )
    if resolved_provider == "openrouter":
        reranker = OpenRouterReranker(
            api_key=api_key or "",
            model=resolved_model or DEFAULT_OPENROUTER_RERANKER_MODEL,
        )
        return RerankerRuntime(
            provider="openrouter",
            model=reranker.model_name,
            device="remote",
            reranker=reranker,
        )
    if resolved_provider != "local":
        raise RerankerError(f"Unsupported reranker provider: {provider}")
    resolved_model = resolved_model or DEFAULT_LOCAL_RERANKER_MODEL
    if platform.system() == "Darwin":
        reranker = SubprocessReranker(
            model=resolved_model,
            device=device,
            instruction=instruction,
        )
        resolved_device = device
        provider = "local-transformers-subprocess"
    else:
        reranker = QwenReranker(
            model=resolved_model,
            device=device,
            instruction=instruction,
        )
        resolved_device = reranker.device
        provider = "local-transformers"
    return RerankerRuntime(
        provider=provider,
        model=resolved_model,
        device=resolved_device,
        reranker=reranker,
    )
