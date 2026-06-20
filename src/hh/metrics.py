"""Metrics.

Two things get measured:

1. Hallucination rate of the *answers* — what fraction of answers the detector
   flags as containing at least one unsupported claim. This describes the data.

2. Detection quality — when gold labels are present, how well the detector's
   flag matches the human label. Reported as a confusion matrix plus
   precision / recall / F1 / accuracy. This describes the *detector*. The
   positive class is "hallucinated".
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from .types import Detection


@dataclass
class ConfusionMatrix:
    tp: int = 0  # predicted hallucinated, truly hallucinated
    fp: int = 0  # predicted hallucinated, truly clean
    tn: int = 0  # predicted clean, truly clean
    fn: int = 0  # predicted clean, truly hallucinated

    @property
    def total(self) -> int:
        return self.tp + self.fp + self.tn + self.fn

    @property
    def precision(self) -> float:
        denom = self.tp + self.fp
        return self.tp / denom if denom else 0.0

    @property
    def recall(self) -> float:
        denom = self.tp + self.fn
        return self.tp / denom if denom else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    @property
    def accuracy(self) -> float:
        return (self.tp + self.tn) / self.total if self.total else 0.0


@dataclass
class EvalResult:
    detector: str
    n_examples: int
    n_claims: int
    # Share of answers flagged as hallucinated by the detector.
    hallucination_rate: float
    mean_groundedness: float
    # Detection quality vs gold labels (None when the dataset is unlabeled).
    n_labeled: int = 0
    confusion: ConfusionMatrix | None = None

    def to_dict(self) -> dict:
        d = {
            "detector": self.detector,
            "n_examples": self.n_examples,
            "n_claims": self.n_claims,
            "hallucination_rate": round(self.hallucination_rate, 4),
            "mean_groundedness": round(self.mean_groundedness, 4),
            "n_labeled": self.n_labeled,
        }
        if self.confusion is not None:
            c = self.confusion
            d["detection_quality"] = {
                **asdict(c),
                "precision": round(c.precision, 4),
                "recall": round(c.recall, 4),
                "f1": round(c.f1, 4),
                "accuracy": round(c.accuracy, 4),
            }
        return d


def evaluate(detections: list[Detection]) -> EvalResult:
    if not detections:
        return EvalResult("", 0, 0, 0.0, 1.0)

    n = len(detections)
    n_claims = sum(len(d.verdicts) for d in detections)
    flagged = sum(1 for d in detections if d.predicted_hallucinated)
    mean_grounded = sum(d.groundedness for d in detections) / n

    cm = ConfusionMatrix()
    n_labeled = 0
    for d in detections:
        gold = d.example.gold_hallucinated
        if gold is None:
            continue
        n_labeled += 1
        pred = d.predicted_hallucinated
        if pred and gold:
            cm.tp += 1
        elif pred and not gold:
            cm.fp += 1
        elif not pred and not gold:
            cm.tn += 1
        else:
            cm.fn += 1

    return EvalResult(
        detector=detections[0].detector,
        n_examples=n,
        n_claims=n_claims,
        hallucination_rate=flagged / n,
        mean_groundedness=mean_grounded,
        n_labeled=n_labeled,
        confusion=cm if n_labeled else None,
    )
