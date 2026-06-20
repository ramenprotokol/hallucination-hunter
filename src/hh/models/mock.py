"""Deterministic, offline model backend.

This backend never calls the network. It exists so the whole tool — CLI, tests,
CI, demos — runs green with zero API keys. Two modes:

1. Fixture mode: return a canned completion for a prompt that matches a stored
   fixture (exact match, or by a registered key). Used by tests that need a
   specific judge response.
2. Heuristic mode (default fallback): when no fixture matches, return a small,
   rule-based JSON verdict so the LLM verifier still produces *something*
   structured and deterministic. This keeps the pipeline runnable end-to-end
   without pretending to be a real model.

It is honest about what it is: `mock` is a baseline/test backend, not a claim of
real model quality. Real evaluations use the `anthropic` or `openai` backends.
"""

from __future__ import annotations

import hashlib
import json

from .base import Model, register


@register("mock")
class MockModel(Model):
    def __init__(self, fixtures: dict[str, str] | None = None, **_: object) -> None:
        # Map of prompt (or fixture key) -> canned completion.
        self.fixtures: dict[str, str] = dict(fixtures or {})
        self.calls: list[str] = []

    # -- fixture management ------------------------------------------------
    def add_fixture(self, prompt: str, completion: str) -> None:
        self.fixtures[prompt] = completion

    @staticmethod
    def _digest(prompt: str) -> str:
        return hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:12]

    # -- Model interface ---------------------------------------------------
    def generate(self, prompt: str, *, system: str | None = None,
                 temperature: float = 0.0, max_tokens: int = 1024) -> str:
        self.calls.append(prompt)
        if prompt in self.fixtures:
            return self.fixtures[prompt]
        digest = self._digest(prompt)
        if digest in self.fixtures:
            return self.fixtures[digest]
        return self._heuristic(prompt)

    # -- heuristic fallback ------------------------------------------------
    def _heuristic(self, prompt: str) -> str:
        """A deterministic stand-in verdict for the LLM-verifier prompt format.

        The LLM verifier asks for a JSON object {support, rationale}. When no
        fixture is registered we approximate groundedness with simple token
        overlap between the CLAIM and CONTEXT sections of the prompt. This is
        intentionally crude — it only needs to be deterministic and structured.
        """
        claim = _section(prompt, "CLAIM:")
        context = _section(prompt, "CONTEXT:")
        if claim and context is not None:
            support = _overlap_support(claim, context)
            return json.dumps(
                {
                    "support": support,
                    "rationale": f"mock heuristic: token-overlap verdict for claim "
                    f"(deterministic, offline).",
                }
            )
        # Generic echo so callers that just want *some* text get something stable.
        return f"[mock:{self._digest(prompt)}]"


def _section(prompt: str, header: str) -> str | None:
    """Pull the text following a header line up to the next blank line / EOF."""
    idx = prompt.find(header)
    if idx == -1:
        return None
    rest = prompt[idx + len(header):]
    # Stop at the next double newline (section break) if present.
    end = rest.find("\n\n")
    chunk = rest if end == -1 else rest[:end]
    return chunk.strip()


def _tokens(text: str) -> set[str]:
    return {t for t in "".join(c.lower() if c.isalnum() else " " for c in text).split()
            if len(t) > 2}


def _overlap_support(claim: str, context: str) -> str:
    ct, xt = _tokens(claim), _tokens(context)
    if not ct:
        return "unverifiable"
    overlap = len(ct & xt) / len(ct)
    if overlap >= 0.6:
        return "supported"
    if overlap <= 0.1:
        return "unverifiable"
    return "unverifiable" if overlap < 0.4 else "supported"
