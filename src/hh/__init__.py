"""hallucination-hunter — a runnable harness for detecting and evaluating
hallucinations in LLM output.

Public surface:

    from hh import detect, load_detector, load_model, run_eval, load_dataset
"""

from __future__ import annotations

__version__ = "0.1.0"

from .dataset import load_dataset
from .detectors.base import available_detectors, load_detector
from .eval import EvalRun, run_eval
from .metrics import EvalResult, evaluate
from .models.base import available_backends, load_model
from .types import Detection, Example, Support, Verdict


def detect(answer: str, context: str = "", *, detector: str = "groundedness",
           verifier: str = "lexical", model=None) -> Detection:
    """Convenience one-shot: check a single answer against a context."""
    kwargs = {"verifier": verifier}
    if model is not None:
        kwargs["model"] = model
    det = load_detector(detector, **kwargs)
    return det.detect(Example(id="adhoc", context=context, answer=answer))


__all__ = [
    "__version__",
    "detect",
    "load_dataset",
    "load_detector",
    "available_detectors",
    "load_model",
    "available_backends",
    "run_eval",
    "EvalRun",
    "evaluate",
    "EvalResult",
    "Detection",
    "Example",
    "Support",
    "Verdict",
]
