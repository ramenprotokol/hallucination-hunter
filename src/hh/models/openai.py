"""OpenAI backend.

Optional: the `openai` package is imported lazily inside `generate`. Install with:

    pip install "hallucination-hunter[openai]"

and set OPENAI_API_KEY in the environment.
"""

from __future__ import annotations

import os

from .base import Model, register

DEFAULT_MODEL_ID = "gpt-4o-mini"


@register("openai")
class OpenAIModel(Model):
    def __init__(
        self, model_id: str = DEFAULT_MODEL_ID, api_key: str | None = None, **_: object
    ) -> None:
        self.model_id = model_id
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self._client = None

    def _ensure_client(self):
        if self._client is not None:
            return self._client
        try:
            import openai
        except ImportError as e:  # pragma: no cover - depends on optional extra
            raise RuntimeError(
                "the 'openai' package is required for this backend; install "
                "with: pip install 'hallucination-hunter[openai]'"
            ) from e
        if not self._api_key:
            raise RuntimeError("OPENAI_API_KEY is not set")
        self._client = openai.OpenAI(api_key=self._api_key)
        return self._client

    def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> str:
        client = self._ensure_client()
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        resp = client.chat.completions.create(
            model=self.model_id,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content or ""
