"""Groundedness detection — the v0.1 flagship detector.

Idea: an answer is "grounded" if every factual claim it makes is supported by
the provided source context. Any claim the context does not support (either it
contradicts the claim, or there is simply no evidence) is a hallucination.

Pipeline:  answer --extract--> [claims] --verify each--> [verdicts] --> Detection

The verification step is pluggable:

- LexicalVerifier: deterministic token-overlap baseline. No model, no keys, runs
  anywhere. Honest baseline — fast and explainable, but blind to paraphrase.
- LLMVerifier: asks a judge model whether the context supports each claim and
  parses a structured verdict. More accurate; needs a backend (or the mock).
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod

from ..claims import ClaimExtractor, SentenceClaimExtractor
from ..models.base import Model
from ..types import Claim, Detection, Example, Support, Verdict
from .base import Detector, register


# --------------------------------------------------------------------------
# Verifiers
# --------------------------------------------------------------------------
class Verifier(ABC):
    @abstractmethod
    def verify(self, claim: Claim, context: str) -> Verdict:
        raise NotImplementedError


def _tokens(text: str) -> list[str]:
    return [t for t in "".join(c.lower() if c.isalnum() else " " for c in text).split()]


_STOP = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being", "of",
    "to", "in", "on", "at", "and", "or", "for", "with", "as", "by", "that",
    "this", "it", "its", "from", "has", "have", "had", "but", "not", "no",
}


def _content_tokens(text: str) -> set[str]:
    return {t for t in _tokens(text) if t not in _STOP and len(t) > 1}


class LexicalVerifier(Verifier):
    """Token-overlap baseline.

    Supported if most of the claim's content words appear in the context.
    Contradicted is approximated by a negation mismatch (the claim negates
    something the context asserts, or vice versa). Otherwise unverifiable.
    """

    NEGATIONS = {"not", "no", "never", "none", "cannot", "n't", "without"}

    def __init__(self, support_threshold: float = 0.6,
                 unverifiable_threshold: float = 0.4) -> None:
        self.support_threshold = support_threshold
        self.unverifiable_threshold = unverifiable_threshold

    def verify(self, claim: Claim, context: str) -> Verdict:
        ct = _content_tokens(claim.text)
        xt = _content_tokens(context)
        if not ct:
            return Verdict(claim, Support.UNVERIFIABLE, 0.0,
                           "empty claim after tokenization")
        overlap = len(ct & xt) / len(ct)
        claim_neg = bool(_tokens(claim.text)) and any(
            n in claim.text.lower() for n in self.NEGATIONS
        )
        context_neg = any(n in context.lower() for n in self.NEGATIONS)
        # Strong lexical match but opposite polarity -> likely contradiction.
        if overlap >= self.support_threshold and claim_neg != context_neg:
            return Verdict(claim, Support.CONTRADICTED, round(overlap, 3),
                           "high lexical overlap but mismatched negation")
        if overlap >= self.support_threshold:
            return Verdict(claim, Support.SUPPORTED, round(overlap, 3),
                           f"{len(ct & xt)}/{len(ct)} content words found in context")
        if overlap < self.unverifiable_threshold:
            return Verdict(claim, Support.UNVERIFIABLE, round(1 - overlap, 3),
                           "claim content words largely absent from context")
        # Middle ground: not enough evidence to call it supported.
        return Verdict(claim, Support.UNVERIFIABLE, round(1 - overlap, 3),
                       "partial overlap, insufficient to confirm support")


class LLMVerifier(Verifier):
    """Ask a judge model to classify support of a claim by the context."""

    PROMPT = (
        "You are a strict fact-checker. Decide whether the CONTEXT supports the "
        "CLAIM.\n"
        "- 'supported': the context clearly entails the claim.\n"
        "- 'contradicted': the context states the opposite of the claim.\n"
        "- 'unverifiable': the context does not contain enough to confirm it.\n"
        "Respond with ONLY a JSON object: "
        '{{"support": "...", "rationale": "..."}}\n\n'
        "CONTEXT:\n{context}\n\n"
        "CLAIM:\n{claim}\n"
    )

    def __init__(self, model: Model) -> None:
        self.model = model

    def verify(self, claim: Claim, context: str) -> Verdict:
        raw = self.model.generate(
            self.PROMPT.format(context=context, claim=claim.text), temperature=0.0
        )
        support, rationale = _parse_verdict(raw)
        return Verdict(claim, support, 1.0, rationale or "llm verdict")


def _parse_verdict(raw: str) -> tuple[Support, str]:
    raw = (raw or "").strip()
    start, end = raw.find("{"), raw.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            data = json.loads(raw[start : end + 1])
            support = str(data.get("support", "")).lower().strip()
            rationale = str(data.get("rationale", ""))
            if support in {s.value for s in Support}:
                return Support(support), rationale
        except (ValueError, TypeError):
            pass
    # Last resort: scan the text for a verdict word.
    low = raw.lower()
    for s in (Support.CONTRADICTED, Support.SUPPORTED, Support.UNVERIFIABLE):
        if s.value in low:
            return s, raw
    return Support.UNVERIFIABLE, "could not parse verdict"


# --------------------------------------------------------------------------
# Detector
# --------------------------------------------------------------------------
@register("groundedness")
class GroundednessDetector(Detector):
    """Extract claims, verify each against the context, aggregate."""

    def __init__(self, verifier: str = "lexical", model: Model | None = None,
                 extractor: ClaimExtractor | None = None, **_: object) -> None:
        self.extractor = extractor or SentenceClaimExtractor()
        self.verifier_name = verifier
        if verifier == "lexical":
            self.verifier: Verifier = LexicalVerifier()
        elif verifier == "llm":
            if model is None:
                raise ValueError("the 'llm' verifier requires a model backend")
            self.verifier = LLMVerifier(model)
        else:
            raise ValueError(f"unknown verifier {verifier!r}; use 'lexical' or 'llm'")

    def detect(self, example: Example) -> Detection:
        claims = self.extractor.extract(example.answer)
        verdicts = [self.verifier.verify(c, example.context) for c in claims]
        return Detection(example=example, verdicts=verdicts,
                         detector=f"groundedness:{self.verifier_name}")
