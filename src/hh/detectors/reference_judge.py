"""Reference-based judging.

When a gold/reference answer is available, judge each claim in the candidate
answer against that reference instead of a source document. Useful for closed-book
QA eval where you trust a curated answer key.

Reuses the groundedness machinery: the reference text *is* the context. With the
lexical verifier this needs no model; with the llm verifier it uses a judge.
"""

from __future__ import annotations

from ..models.base import Model
from ..types import Detection, Example
from .base import Detector, register
from .groundedness import GroundednessDetector


@register("reference_judge")
class ReferenceJudgeDetector(Detector):
    def __init__(self, verifier: str = "lexical", model: Model | None = None, **_: object) -> None:
        self._inner = GroundednessDetector(verifier=verifier, model=model)
        self.verifier_name = verifier

    def detect(self, example: Example) -> Detection:
        if not example.reference:
            raise ValueError(f"example {example.id!r} has no reference answer to judge against")
        # Treat the reference answer as the context the claims must match.
        proxy = Example(**{**example.__dict__, "context": example.reference})
        det = self._inner.detect(proxy)
        det.detector = f"reference_judge:{self.verifier_name}"
        det.example = example  # report against the original example
        return det
