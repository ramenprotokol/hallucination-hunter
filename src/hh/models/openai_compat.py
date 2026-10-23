"""OpenAI-compatible backend (any `/chat/completions` endpoint, tool calling).

Works with OpenRouter, vLLM, Ollama's OpenAI endpoint and similar servers.
Implemented with `urllib.request` only, so it needs no third-party dependency.

Configuration, all from the environment:

- `HH_BASE_URL`  base URL of the API (default `https://openrouter.ai/api/v1`)
- `HH_MODEL`     model id; no default. Pick an open-source model with tool calling.
- `HH_API_KEY`   your own key, sent as `Authorization: Bearer <key>`

The key is never logged or included in an error message.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any

from .base import AssistantTurn, Model, Tool, ToolCall, register

DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_TIMEOUT = 120.0


@register("openai-compat")
class OpenAICompatModel(Model):
    supports_tools = True

    def __init__(
        self,
        model_id: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: float = DEFAULT_TIMEOUT,
        **_: object,
    ) -> None:
        self.model_id = model_id or os.environ.get("HH_MODEL", "")
        self.base_url = (base_url or os.environ.get("HH_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")
        self._api_key = api_key or os.environ.get("HH_API_KEY", "")
        self.timeout = timeout

    # -- config ---------------------------------------------------------------
    def _headers(self) -> dict[str, str]:
        if not self._api_key:
            raise RuntimeError("HH_API_KEY is not set; set it to your API key for this endpoint")
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._api_key}",
        }

    def _require_model(self) -> str:
        if not self.model_id:
            raise RuntimeError(
                "HH_MODEL is not set; pick an open-source model with tool calling "
                "and set HH_MODEL to its id"
            )
        return self.model_id

    # -- public interface ---------------------------------------------------
    def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> str:
        messages: list[dict[str, Any]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        # Reasoning models can spend the entire budget on invisible reasoning
        # and return null content; when a turn comes back completely empty,
        # escalate the budget before giving up.
        text = ""
        for budget in (max_tokens, max_tokens * 4, max_tokens * 16):
            payload = self._chat_payload(
                messages, tools=None, temperature=temperature, max_tokens=budget
            )
            text = self._text(self._post(payload))
            if text.strip():
                break
        return text

    def generate_with_tools(
        self,
        messages: list[dict[str, Any]],
        tools: list[Tool],
        *,
        system: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> AssistantTurn:
        wire_messages = self._to_wire(messages)
        if system:
            wire_messages.insert(0, {"role": "system", "content": system})
        turn = AssistantTurn()
        for budget in (max_tokens, max_tokens * 4, max_tokens * 16):
            payload = self._chat_payload(
                wire_messages,
                tools=[t.to_api() for t in tools] or None,
                temperature=temperature,
                max_tokens=budget,
            )
            turn = self._turn(self._post(payload))
            if turn.text.strip() or turn.tool_calls:
                return turn
        return turn

    # -- request / response helpers ------------------------------------------
    @staticmethod
    def _to_wire(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Neutral conversation -> OpenAI wire format (assistant tool_calls)."""
        wire: list[dict[str, Any]] = []
        for m in messages:
            if m.get("role") == "assistant" and m.get("tool_calls"):
                wire.append(
                    {
                        "role": "assistant",
                        "content": m.get("content") or None,
                        "tool_calls": [
                            {
                                "id": tc["id"],
                                "type": "function",
                                "function": {
                                    "name": tc["name"],
                                    "arguments": json.dumps(tc.get("arguments", {})),
                                },
                            }
                            for tc in m["tool_calls"]
                        ],
                    }
                )
            else:
                wire.append(m)
        return wire

    def _chat_payload(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None,
        temperature: float,
        max_tokens: int,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self._require_model(),
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        return payload

    def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers=self._headers(),
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                body = resp.read().decode("utf-8")
        except urllib.error.HTTPError as e:
            # Status only — never echo the request (which contains credentials)
            # or trust the body to be free of secrets.
            raise RuntimeError(f"the model API returned HTTP {e.code}") from e
        except TimeoutError as e:
            raise RuntimeError("could not reach the model API: timed out") from e
        except urllib.error.URLError as e:
            reason = getattr(e, "reason", e)
            raise RuntimeError(f"could not reach the model API: {reason}") from e
        try:
            return json.loads(body)
        except ValueError as e:
            raise RuntimeError("invalid JSON from the model API") from e

    @staticmethod
    def _message(resp: dict[str, Any]) -> dict[str, Any]:
        try:
            msg = resp["choices"][0]["message"]
        except (KeyError, IndexError, TypeError) as e:
            raise RuntimeError("unexpected response shape from the model API") from e
        return msg if isinstance(msg, dict) else {}

    @classmethod
    def _text(cls, resp: dict[str, Any]) -> str:
        msg = cls._message(resp)
        content = msg.get("content")
        return content if isinstance(content, str) else ""

    @classmethod
    def _turn(cls, resp: dict[str, Any]) -> AssistantTurn:
        msg = cls._message(resp)
        calls = []
        for i, tc in enumerate(msg.get("tool_calls") or []):
            fn = tc.get("function", {}) if isinstance(tc, dict) else {}
            raw_args = fn.get("arguments")
            try:
                parsed = json.loads(raw_args) if raw_args else {}
            except (ValueError, TypeError):
                parsed = raw_args  # not JSON — keep the original text
            if not isinstance(parsed, dict):
                parsed = {"_raw": parsed}
            calls.append(
                ToolCall(
                    id=str(tc.get("id", f"call_{i}")),
                    name=str(fn.get("name", "")),
                    arguments=parsed,
                )
            )
        content = msg.get("content")
        return AssistantTurn(text=content if isinstance(content, str) else "", tool_calls=calls)
