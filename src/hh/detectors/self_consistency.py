"""Self-consistency detection (SelfCheckGPT-style).

Premise: if a model truly knows a fact, it answers consistently across samples.
If it is fabricating, independent samples tend to disagree. So we sample the
model N times for the same prompt and measure how well the answer-under-test
agrees with the samples. Low agreement => likely hallucination.

This detector needs NO source context — it works when you have a prompt and a
model but no ground-truth document. It does need a backend that can be sampled at
non-zero temperature (the mock backend works for tests; real signal needs a real
model).
"""

from __future__ import annotations

from ..claims import SentenceClaimExtractor
from ..models.base import Model
from ..types import Claim, Detection, Example, Support, Verdict
from .base import Detector, register


def _content_tokens(text: str) -> set[str]:
    from .groundedness import _content_tokens as ct
    return ct(text)


@register("self_consistency")
class SelfConsistencyDetector(Detector):
    """Flag claims that are not echoed by independent samples of the model."""

    def __init__(self, model: Model | None = None, samples: int = 5,
                 temperature: float = 1.0, support_threshold: float = 0.6,
                 **_: object) -> None:
        if model is None:
            raise ValueError("self_consistency requires a model backend to sample")
        self.model = model
        self.samples = samples
        self.temperature = temperature
        self.support_threshold = support_threshold
        self.extractor = SentenceClaimExtractor()

    def _sample(self, prompt: str) -> list[str]:
        return [
            self.model.generate(prompt, temperature=self.temperature)
            for _ in range(self.samples)
        ]

    def detect(self, example: Example) -> Detection:
        # The answer under test: an explicit answer if given, else one drawn at
        # temperature 0 (the model's "best" answer).
        answer = example.answer or self.model.generate(example.prompt, temperature=0.0)
        samples = self._sample(example.prompt)
        sample_tokens = [_content_tokens(s) for s in samples]

        verdicts: list[Verdict] = []
        for claim in self.extractor.extract(answer):
            verdicts.append(self._verify(claim, sample_tokens))
        return Detection(example=Example(**{**example.__dict__, "answer": answer}),
                         verdicts=verdicts, detector=f"self_consistency:n={self.samples}")

    def _verify(self, claim: Claim, sample_tokens: list[set[str]]) -> Verdict:
        ct = _content_tokens(claim.text)
        if not ct:
            return Verdict(claim, Support.UNVERIFIABLE, 0.0, "empty claim")
        if not sample_tokens:
            return Verdict(claim, Support.UNVERIFIABLE, 0.0, "no samples drawn")
        # Fraction of samples whose content sufficiently overlaps the claim.
        agree = 0
        for st in sample_tokens:
            if ct and len(ct & st) / len(ct) >= self.support_threshold:
                agree += 1
        consistency = agree / len(sample_tokens)
        if consistency >= 0.5:
            return Verdict(claim, Support.SUPPORTED, round(consistency, 3),
                           f"echoed by {agree}/{len(sample_tokens)} samples")
        return Verdict(claim, Support.UNVERIFIABLE, round(1 - consistency, 3),
                       f"only {agree}/{len(sample_tokens)} samples agree — likely fabricated")
