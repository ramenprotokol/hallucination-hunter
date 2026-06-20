"""Dataset loading.

Datasets are JSONL — one JSON object per line, each becoming an Example. This
format is append-friendly, diff-friendly, and streams without loading everything
into memory. See data/README.md for the schema.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Iterator

from .types import Example


def iter_jsonl(path: str | Path) -> Iterator[dict]:
    p = Path(path)
    with p.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line or line.startswith("//"):
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{p}:{lineno}: invalid JSON: {e}") from e


def load_dataset(path: str | Path) -> list[Example]:
    return [Example.from_dict(d) for d in iter_jsonl(path)]


def save_dataset(examples: Iterable[Example], path: str | Path) -> None:
    p = Path(path)
    with p.open("w", encoding="utf-8") as fh:
        for ex in examples:
            row = {
                "id": ex.id,
                "prompt": ex.prompt,
                "context": ex.context,
                "answer": ex.answer,
            }
            if ex.reference:
                row["reference"] = ex.reference
            labels = {}
            if ex.gold_hallucinated is not None:
                labels["hallucinated"] = ex.gold_hallucinated
            if ex.gold_unsupported_claims:
                labels["unsupported_claims"] = ex.gold_unsupported_claims
            if labels:
                row["labels"] = labels
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
