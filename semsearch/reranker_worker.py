from __future__ import annotations

import json
import sys

from .rerankers import QwenReranker, RerankerError


def main() -> int:
    try:
        payload = json.load(sys.stdin)
        reranker = QwenReranker(
            model=str(payload["model"]),
            device=str(payload.get("device", "auto")),
            instruction=str(payload.get("instruction", "")),
            max_length=int(payload.get("max_length", 8192)),
        )
        scores = reranker.score(
            str(payload["query"]),
            [str(item) for item in payload.get("documents", [])],
        )
        json.dump({"scores": scores}, sys.stdout)
        return 0
    except RerankerError as err:
        print(str(err), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
