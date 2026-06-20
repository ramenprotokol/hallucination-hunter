"""Anthropic (Claude) backend.

Optional: the `anthropic` package is imported lazily inside `generate`, so the
core stays installable and testable with no third-party deps. Install with:

    pip install "hallucination-hunter[anthropic]"

and set ANTHROPIC_API_KEY in the environment.
"""

from __future__ import annotations

import os

from .base import Model, register

# A current, capable default. Override per call via the CLI's --model-id flag.
DEFAULT_MODEL_ID = "claude-sonnet-4-6"


@register("anthropic")
class AnthropicModel(Model):
    def __init__(self, model_id: str = DEFAULT_MODEL_ID,
                 api_key: str | None = None, **_: object) -> None:
        self.model_id = model_id
        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self._client = None

    def _ensure_client(self):
        if self._client is not None:
            return self._client
        try:
            import anthropic
        except ImportError as e:  # pragma: no cover - depends on optional extra
            raise RuntimeError(
                "the 'anthropic' package is required for this backend; install "
                "with: pip install 'hallucination-hunter[anthropic]'"
            ) from e
        if not self._api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is not set")
        self._client = anthropic.Anthropic(api_key=self._api_key)
        return self._client

    def generate(self, prompt: str, *, system: str | None = None,
                 temperature: float = 0.0, max_tokens: int = 1024) -> str:
        client = self._ensure_client()
        kwargs = {
            "model": self.model_id,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            kwargs["system"] = system
        resp = client.messages.create(**kwargs)
        return "".join(
            block.text for block in resp.content if getattr(block, "type", None) == "text"
        )
