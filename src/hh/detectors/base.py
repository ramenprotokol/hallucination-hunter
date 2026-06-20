"""Detector interface and registry.

A Detector takes an Example and returns a Detection (per-claim verdicts plus the
overall hallucinated/grounded call). Everything downstream — metrics, reports —
works on Detection, so new detection strategies plug in here without changing the
rest of the pipeline.

v0.1 ships GroundednessDetector. SelfConsistencyDetector and ReferenceJudgeDetector
implement the same interface and are wired in behind the registry.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..types import Detection, Example


class Detector(ABC):
    name: str = "detector"

    @abstractmethod
    def detect(self, example: Example) -> Detection:
        raise NotImplementedError

    def detect_many(self, examples: list[Example]) -> list[Detection]:
        return [self.detect(ex) for ex in examples]


_REGISTRY: dict[str, type] = {}


def register(name: str):
    def deco(cls):
        _REGISTRY[name] = cls
        cls.name = name
        return cls

    return deco


def available_detectors() -> list[str]:
    return sorted(_REGISTRY)


def load_detector(name: str, **kwargs) -> Detector:
    if name not in _REGISTRY:
        raise ValueError(
            f"unknown detector {name!r}; available: {available_detectors()}"
        )
    return _REGISTRY[name](**kwargs)


def _register_builtins() -> None:
    from . import groundedness, self_consistency, reference_judge  # noqa: F401


_register_builtins()
