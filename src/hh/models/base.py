"""Model backends: a thin, uniform interface over any text-generating LLM.

A `Model` does exactly one thing — turn a prompt into text. Detectors that need
a judge (the LLM verifier, the self-consistency detector) take a Model and stay
ignorant of which provider is behind it. That is the whole point of the seam:
swap Anthropic for OpenAI for the deterministic mock without touching detection
logic.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class Model(ABC):
    """Anything that can generate text from a prompt."""

    name: str = "model"

    @abstractmethod
    def generate(self, prompt: str, *, system: str | None = None,
                 temperature: float = 0.0, max_tokens: int = 1024) -> str:
        """Return the model's completion for `prompt`."""
        raise NotImplementedError

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
        raise ValueError(
            f"unknown model backend {name!r}; available: {available_backends()}"
        )
    return _REGISTRY[name](**kwargs)


# Importing the package registers the built-in backends as a side effect.
def _register_builtins() -> None:
    from . import mock, anthropic, openai  # noqa: F401


_register_builtins()
