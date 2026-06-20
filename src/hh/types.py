"""Core data types for hallucination-hunter.

The whole pipeline speaks in these types:

    Example  -> what we evaluate (a prompt, a source context, an answer, gold labels)
    Claim    -> one atomic factual statement pulled out of an answer
    Verdict  -> the detector's call on a single claim, against the context
    Detection-> all verdicts for one Example, plus the derived hallucination flag
    EvalResult-> dataset-level aggregate, including detection quality vs gold labels

Keeping these as plain dataclasses (no third-party deps) keeps the core importable
anywhere, including CI with no packages installed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Support(str, Enum):
    """How well a claim is backed by the source context.

    SUPPORTED      - the context entails the claim.
    CONTRADICTED   - the context states the opposite of the claim.
    UNVERIFIABLE   - the context neither supports nor contradicts it (no evidence).
    """

    SUPPORTED = "supported"
    CONTRADICTED = "contradicted"
    UNVERIFIABLE = "unverifiable"


# A claim counts as a hallucination if the context does not back it up. Both an
# outright contradiction and an unverifiable (fabricated / unsupported) claim are
# failures of groundedness — the two non-SUPPORTED verdicts.
HALLUCINATION_VERDICTS: frozenset[Support] = frozenset({Support.CONTRADICTED, Support.UNVERIFIABLE})


@dataclass(frozen=True)
class Claim:
    """One atomic factual statement extracted from an answer."""

    text: str
    # Index of the claim within the answer (order it was extracted in).
    index: int = 0


@dataclass(frozen=True)
class Verdict:
    """A detector's judgement on a single claim, relative to the context."""

    claim: Claim
    support: Support
    # 0.0-1.0 — how confident the detector is in this verdict.
    score: float = 1.0
    # Human-readable reason; for LLM verifiers this is the model's rationale.
    rationale: str = ""
    # The slice(s) of context the detector leaned on, when it can name them.
    evidence: str = ""

    @property
    def is_hallucination(self) -> bool:
        return self.support in HALLUCINATION_VERDICTS


@dataclass
class Example:
    """One row of an evaluation dataset.

    `answer` is the text we inspect for hallucinations. It may be supplied
    directly (a static eval set, fully reproducible) or generated at run time
    from `prompt` by a Model.

    `gold_hallucinated` is the human label, when present: True if the answer is
    known to contain at least one unsupported claim. It is what we score the
    detector against. Leave it None for unlabeled data.
    """

    id: str
    prompt: str = ""
    context: str = ""
    answer: str = ""
    reference: str = ""
    gold_hallucinated: bool | None = None
    # Optional: the specific claims a human marked as unsupported.
    gold_unsupported_claims: list[str] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Example:
        labels = d.get("labels", {}) or {}
        gold = labels.get("hallucinated", d.get("gold_hallucinated"))
        return cls(
            id=str(d.get("id", "")),
            prompt=d.get("prompt", ""),
            context=d.get("context", ""),
            answer=d.get("answer", ""),
            reference=d.get("reference", ""),
            gold_hallucinated=gold,
            gold_unsupported_claims=list(
                labels.get("unsupported_claims", d.get("gold_unsupported_claims", []))
            ),
            meta=d.get("meta", {}) or {},
        )


@dataclass
class Detection:
    """All claim verdicts for one Example, plus the overall hallucination call."""

    example: Example
    verdicts: list[Verdict]
    detector: str = ""

    @property
    def claims(self) -> list[Claim]:
        return [v.claim for v in self.verdicts]

    @property
    def hallucinated_claims(self) -> list[Verdict]:
        return [v for v in self.verdicts if v.is_hallucination]

    @property
    def predicted_hallucinated(self) -> bool:
        """An answer is flagged if any of its claims is not supported."""
        return any(v.is_hallucination for v in self.verdicts)

    @property
    def groundedness(self) -> float:
        """Fraction of claims that are supported (1.0 = fully grounded)."""
        if not self.verdicts:
            return 1.0
        supported = sum(1 for v in self.verdicts if v.support is Support.SUPPORTED)
        return supported / len(self.verdicts)
