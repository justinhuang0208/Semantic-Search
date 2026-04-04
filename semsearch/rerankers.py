from __future__ import annotations

import json
import platform
import subprocess
import sys
from dataclasses import dataclass
from typing import Protocol


DEFAULT_QWEN_RERANKER_MODEL = "tomaarsen/Qwen3-Reranker-0.6B-seq-cls"
DEFAULT_RERANKER_INSTRUCTION = (
    "Given a search query, retrieve relevant passages that answer the query."
)
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
        model: str = DEFAULT_QWEN_RERANKER_MODEL,
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


def resolve_reranker(
    *,
    use_reranker: bool,
    model: str | None,
    device: str,
    instruction: str = DEFAULT_RERANKER_INSTRUCTION,
) -> RerankerRuntime | None:
    if not use_reranker:
        return None
    resolved_model = (model or "").strip() or DEFAULT_QWEN_RERANKER_MODEL
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
