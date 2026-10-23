"""Claim extraction: break an answer into atomic factual statements.

Groundedness is checked one claim at a time, so the first step is splitting an
answer into claims. Two strategies, same interface:

- SentenceClaimExtractor: deterministic, dependency-free sentence splitting.
  Good enough for most answers and runs anywhere with no model.
- LLMClaimExtractor: asks a Model to decompose the answer into atomic claims
  (one fact each). More accurate on compound sentences; needs a backend.
"""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod

from .models.base import Model
from .types import Claim

# Abbreviations whose trailing period must not be treated as a sentence end.
_ABBREV = {"mr", "mrs", "ms", "dr", "st", "vs", "etc", "e.g", "i.e", "no", "fig"}
_SENT_END = re.compile(r"([.!?])\s+(?=[A-Z0-9\"'(])")


class ClaimExtractor(ABC):
    @abstractmethod
    def extract(self, answer: str) -> list[Claim]:
        raise NotImplementedError


class SentenceClaimExtractor(ClaimExtractor):
    """Split into sentences with a few abbreviation guards. No dependencies."""

    def extract(self, answer: str) -> list[Claim]:
        text = (answer or "").strip()
        if not text:
            return []
        # Insert split markers after sentence-ending punctuation, then guard
        # against common abbreviations by checking the token before the period.
        pieces = _SENT_END.split(text)
        # re.split with a capture group interleaves separators; rejoin them.
        sentences: list[str] = []
        buf = ""
        for part in pieces:
            if part in (".", "!", "?"):
                buf += part
                continue
            if buf:
                sentences.append(buf.strip())
            buf = part
        if buf.strip():
            sentences.append(buf.strip())

        merged = self._merge_abbreviations(sentences)
        return [Claim(text=s, index=i) for i, s in enumerate(merged) if s]

    @staticmethod
    def _merge_abbreviations(sentences: list[str]) -> list[str]:
        # A split is spurious when the *preceding* fragment ends in an
        # abbreviation (e.g. "He saw Dr." | "Smith." -> rejoin). We look back at
        # the fragment already in `out`, not the current one.
        out: list[str] = []
        for s in sentences:
            if out:
                prev_words = out[-1].split()
                prev_last = prev_words[-1].rstrip(".").lower() if prev_words else ""
                if prev_last in _ABBREV:
                    out[-1] = out[-1] + " " + s
                    continue
            out.append(s)
        return out


class LLMClaimExtractor(ClaimExtractor):
    """Decompose an answer into atomic claims using a model."""

    PROMPT = (
        "Break the following ANSWER into ATOMIC factual claims — one fact per claim.\n"
        "Rules:\n"
        "- Split every coordination: a sentence like 'X happened in 2019 and "
        "Y also happened' is TWO claims, never one.\n"
        "- Each claim must stand alone and be verifiable on its own.\n"
        "- Do not merge two facts just because they share a subject.\n"
        "- Keep the original wording as much as possible.\n"
        "Return ONLY a JSON array of strings.\n\nANSWER:\n{answer}\n"
    )

    # Second pass for claims that still look compound after extraction.
    SPLIT_PROMPT = (
        "The CLAIM below may state more than one fact. If it does, split it "
        "into atomic claims (original wording, one fact each). If it already "
        "states exactly one fact, return it unchanged.\n"
        "Return ONLY a JSON array of strings.\n\nCLAIM:\n{claim}\n"
    )

    def __init__(self, model: Model, *, split_compound: bool = True) -> None:
        self.model = model
        self.split_compound = split_compound

    def extract(self, answer: str) -> list[Claim]:
        text = (answer or "").strip()
        if not text:
            return []
        raw = self.model.generate(self.PROMPT.format(answer=text), temperature=0.0)
        items = [str(s).strip() for s in _parse_json_array(raw) if str(s).strip()]
        if not items:
            # Fall back to deterministic splitting rather than returning nothing.
            return SentenceClaimExtractor().extract(text)
        claims: list[Claim] = []
        for i, item in enumerate(self._atomic(items)):
            claims.append(Claim(text=item, index=i))
        return claims

    def _atomic(self, items: list[str]) -> list[str]:
        """Second pass: split claims that still look compound, keep the rest."""
        out: list[str] = []
        for item in items:
            if self.split_compound and _looks_compound(item):
                out.extend(self._refine(item))
            else:
                out.append(item)
        return out

    def _refine(self, claim_text: str) -> list[str]:
        raw = self.model.generate(self.SPLIT_PROMPT.format(claim=claim_text), temperature=0.0)
        parts = [str(s).strip() for s in _parse_json_array(raw) if str(s).strip()]
        # A refusal to split, garbage, or a 1-item reply all keep the original.
        return parts if len(parts) > 1 else [claim_text]


def _looks_compound(text: str) -> bool:
    """Heuristic: a coordinator joining two predicates is two facts in one."""
    return " and " in text.lower()


def _parse_json_array(raw: str) -> list:
    raw = raw.strip()
    start, end = raw.find("["), raw.rfind("]")
    if start == -1 or end == -1 or end < start:
        return []
    try:
        data = json.loads(raw[start : end + 1])
        return data if isinstance(data, list) else []
    except (ValueError, TypeError):
        return []
