"""Tests for hh.metrics: ConfusionMatrix, EvalResult, evaluate()."""

from __future__ import annotations

import pytest

from hh.metrics import ConfusionMatrix, evaluate
from hh.types import Claim, Detection, Example, Support, Verdict


def _make_detection(
    example_id: str, supports: list[Support], gold: bool | None = None
) -> Detection:
    ex = Example(id=example_id, answer="some answer", gold_hallucinated=gold)
    claims = [Claim(text=f"c{i}", index=i) for i in range(len(supports))]
    verdicts = [Verdict(claim=c, support=s) for c, s in zip(claims, supports, strict=True)]
    return Detection(example=ex, verdicts=verdicts, detector="test-det")


# ---------------------------------------------------------------------------
# ConfusionMatrix
# ---------------------------------------------------------------------------


class TestConfusionMatrix:
    def test_precision(self):
        cm = ConfusionMatrix(tp=3, fp=1, tn=5, fn=1)
        assert cm.precision == pytest.approx(3 / 4)

    def test_recall(self):
        cm = ConfusionMatrix(tp=3, fp=1, tn=5, fn=1)
        assert cm.recall == pytest.approx(3 / 4)

    def test_f1(self):
        cm = ConfusionMatrix(tp=3, fp=1, tn=5, fn=1)
        p = cm.precision
        r = cm.recall
        expected = 2 * p * r / (p + r)
        assert cm.f1 == pytest.approx(expected)

    def test_accuracy(self):
        cm = ConfusionMatrix(tp=3, fp=1, tn=5, fn=1)
        assert cm.accuracy == pytest.approx(8 / 10)

    def test_total(self):
        cm = ConfusionMatrix(tp=2, fp=3, tn=4, fn=1)
        assert cm.total == 10

    def test_precision_zero_denominator(self):
        cm = ConfusionMatrix(tp=0, fp=0, tn=5, fn=2)
        assert cm.precision == 0.0

    def test_recall_zero_denominator(self):
        cm = ConfusionMatrix(tp=0, fp=2, tn=5, fn=0)
        assert cm.recall == 0.0

    def test_f1_zero_when_both_zero(self):
        cm = ConfusionMatrix(tp=0, fp=0, tn=5, fn=0)
        assert cm.f1 == 0.0

    def test_accuracy_zero_total(self):
        cm = ConfusionMatrix(tp=0, fp=0, tn=0, fn=0)
        assert cm.accuracy == 0.0

    def test_perfect_precision_recall(self):
        cm = ConfusionMatrix(tp=5, fp=0, tn=5, fn=0)
        assert cm.precision == 1.0
        assert cm.recall == 1.0
        assert cm.f1 == 1.0
        assert cm.accuracy == 1.0


# ---------------------------------------------------------------------------
# evaluate()
# ---------------------------------------------------------------------------


class TestEvaluate:
    def test_empty_detections(self):
        result = evaluate([])
        assert result.n_examples == 0
        assert result.hallucination_rate == 0.0
        assert result.mean_groundedness == 1.0

    def test_hallucination_rate(self):
        # 2 hallucinated, 1 clean
        d1 = _make_detection("a", [Support.UNVERIFIABLE])
        d2 = _make_detection("b", [Support.SUPPORTED])
        d3 = _make_detection("c", [Support.CONTRADICTED])
        result = evaluate([d1, d2, d3])
        assert result.hallucination_rate == pytest.approx(2 / 3)

    def test_mean_groundedness(self):
        # d1 groundedness=0.5, d2 groundedness=1.0  =>  mean=0.75
        d1 = _make_detection("a", [Support.SUPPORTED, Support.UNVERIFIABLE])
        d2 = _make_detection("b", [Support.SUPPORTED])
        result = evaluate([d1, d2])
        assert result.mean_groundedness == pytest.approx(0.75)

    def test_n_claims_sum(self):
        d1 = _make_detection("a", [Support.SUPPORTED, Support.UNVERIFIABLE])
        d2 = _make_detection("b", [Support.SUPPORTED])
        result = evaluate([d1, d2])
        assert result.n_claims == 3

    def test_no_gold_labels_confusion_is_none(self):
        d = _make_detection("a", [Support.SUPPORTED], gold=None)
        result = evaluate([d])
        assert result.confusion is None

    def test_confusion_matrix_from_gold_labels(self):
        # TP: predicted=True, gold=True
        tp = _make_detection("tp", [Support.UNVERIFIABLE], gold=True)
        # FP: predicted=True, gold=False
        fp = _make_detection("fp", [Support.UNVERIFIABLE], gold=False)
        # TN: predicted=False, gold=False
        tn = _make_detection("tn", [Support.SUPPORTED], gold=False)
        # FN: predicted=False, gold=True
        fn = _make_detection("fn", [Support.SUPPORTED], gold=True)

        result = evaluate([tp, fp, tn, fn])
        assert result.confusion is not None
        cm = result.confusion
        assert cm.tp == 1
        assert cm.fp == 1
        assert cm.tn == 1
        assert cm.fn == 1
        assert result.n_labeled == 4

    def test_detector_name_from_first_detection(self):
        d = _make_detection("a", [Support.SUPPORTED])
        result = evaluate([d])
        assert result.detector == "test-det"

    def test_eval_result_to_dict(self):
        d = _make_detection("a", [Support.UNVERIFIABLE], gold=True)
        result = evaluate([d])
        d_out = result.to_dict()
        assert "hallucination_rate" in d_out
        assert "mean_groundedness" in d_out
        assert "n_examples" in d_out
        assert "detector" in d_out
