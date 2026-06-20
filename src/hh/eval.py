"""Evaluation orchestrator: dataset -> detections -> metrics.

This is the glue the CLI calls. It optionally generates answers with a model
(for prompts that ship without a fixed answer), runs the chosen detector over
every example, and aggregates the results.
"""

from __future__ import annotations

from dataclasses import dataclass

from .detectors.base import Detector
from .metrics import EvalResult, evaluate
from .models.base import Model
from .types import Detection, Example


@dataclass
class EvalRun:
    result: EvalResult
    detections: list[Detection]


def run_eval(examples: list[Example], detector: Detector,
             generator: Model | None = None) -> EvalRun:
    """Run `detector` over `examples`, generating answers first if needed."""
    prepared: list[Example] = []
    for ex in examples:
        if not ex.answer and generator is not None and ex.prompt:
            answer = generator.generate(_gen_prompt(ex), temperature=0.0)
            ex = Example(**{**ex.__dict__, "answer": answer})
        prepared.append(ex)

    detections = detector.detect_many(prepared)
    result = evaluate(detections)
    return EvalRun(result=result, detections=detections)


def _gen_prompt(ex: Example) -> str:
    """Build the generation prompt, grounding the model in the context if given."""
    if ex.context:
        return (
            "Answer the question using ONLY the context below.\n\n"
            f"CONTEXT:\n{ex.context}\n\nQUESTION:\n{ex.prompt}\n"
        )
    return ex.prompt
