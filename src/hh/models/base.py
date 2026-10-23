"""Model backends: a thin, uniform interface over any text-generating LLM.

A `Model` does exactly one thing — turn a prompt into text. Detectors that need
a judge (the LLM verifier, the self-consistency detector) take a Model and stay
ignorant of which provider is behind it. That is the whole point of the seam:
swap Anthropic for OpenAI for the deterministic mock without touching detection
logic.

Backends that can also drive tools implement the second, opt-in seam:
`generate_with_tools` performs ONE assistant turn over a running conversation
and reports which tools the model wants called. The caller executes the tools,
appends the results to `messages`, and calls again until the model answers
without tool calls. Keeping the loop in the caller (not the backend) means the
same protocol works for every backend and stays trivially testable.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Tool:
    """A tool the model may call: a JSON-Schema'd function."""

    name: str
    description: str
    # JSON Schema object describing the arguments, e.g.
    # {"type": "object", "properties": {...}, "required": [...]}.
    parameters: dict[str, Any] = field(default_factory=dict)

    def to_api(self) -> dict[str, Any]:
        """The provider-neutral wire format (OpenAI-style function tools)."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


@dataclass(frozen=True)
class ToolCall:
    """One requested tool invocation from the model."""

    id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass
class AssistantTurn:
    """One assistant turn: plain text and/or requested tool calls."""

    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)


class Model(ABC):
    """Anything that can generate text from a prompt."""

    name: str = "model"
    # True when the backend implements generate_with_tools.
    supports_tools: bool = False

    @abstractmethod
    def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> str:
        """Return the model's completion for `prompt`."""
        raise NotImplementedError

    def generate_with_tools(
        self,
        messages: list[dict[str, Any]],
        tools: list[Tool],
        *,
        system: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> AssistantTurn:
        """One assistant turn over `messages` with `tools` available.

        `messages` is a provider-neutral conversation: user messages
        {"role": "user", "content": str}, assistant messages
        {"role": "assistant", "content": str, "tool_calls": [{"id", "name",
        "arguments"}]}, and tool results {"role": "tool", "tool_call_id": str,
        "content": str}. Returns the assistant's turn; the caller executes any
        requested tool calls and appends their results.
        """
        raise NotImplementedError(
            f"the {self.name!r} backend does not support tool calling; "
            "implement generate_with_tools() on it"
        )

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"<Model {self.name}>"


_REGISTRY: dict[str, type] = {}


def register(name: str):
    """Decorator to register a Model subclass under a CLI-friendly name."""

    def deco(cls):
        _REGISTRY[name] = cls
        cls.name = name
        return cls

    return deco


def available_backends() -> list[str]:
    return sorted(_REGISTRY)


def load_model(name: str, **kwargs) -> Model:
    """Instantiate a registered backend by name (e.g. 'mock', 'anthropic')."""
    if name not in _REGISTRY:
        raise ValueError(f"unknown model backend {name!r}; available: {available_backends()}")
    return _REGISTRY[name](**kwargs)


# Importing the package registers the built-in backends as a side effect.
def _register_builtins() -> None:
    from . import anthropic, mock, openai, openai_compat  # noqa: F401


_register_builtins()
