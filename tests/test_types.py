"""Tests for hh.types: Example, Verdict, Detection, Support."""

from __future__ import annotations

import pytest

from hh.types import Claim, Detection, Example, Support, Verdict


# ---------------------------------------------------------------------------
# Support / Verdict.is_hallucination
# ---------------------------------------------------------------------------

class TestVerdictIsHallucination:
    def _verdict(self, support: Support) -> Verdict:
        return Verdict(claim=Claim("test claim", 0), support=support)

    def test_supported_is_not_hallucination(self):
        v = self._verdict(Support.SUPPORTED)
        assert v.is_hallucination is False

    def test_contradicted_is_hallucination(self):
        v = self._verdict(Support.CONTRADICTED)
        assert v.is_hallucination is True

    def test_unverifiable_is_hallucination(self):
        v = self._verdict(Support.UNVERIFIABLE)
        assert v.is_hallucination is True


# ---------------------------------------------------------------------------
# Example.from_dict
# ---------------------------------------------------------------------------

class TestExampleFromDict:
    def test_basic_fields(self):
        d = {"id": "42", "context": "ctx", "answer": "ans"}
        ex = Example.from_dict(d)
        assert ex.id == "42"
        assert ex.context == "ctx"
        assert ex.answer == "ans"
        assert ex.gold_hallucinated is None

    def test_gold_hallucinated_from_labels_dict(self):
        d = {"id": "x", "labels": {"hallucinated": True}}
        ex = Example.from_dict(d)
        assert ex.gold_hallucinated is True

    def test_gold_hallucinated_false_from_labels_dict(self):
        d = {"id": "x", "labels": {"hallucinated": False}}
        ex = Example.from_dict(d)
        assert ex.gold_hallucinated is False

    def test_gold_hallucinated_fallback_to_top_level_key(self):
        d = {"id": "x", "gold_hallucinated": True}
        ex = Example.from_dict(d)
        assert ex.gold_hallucinated is True

    def test_unsupported_claims_from_labels(self):
        d = {
            "id": "x",
            "labels": {"unsupported_claims": ["claim A", "claim B"]},
        }
        ex = Example.from_dict(d)
        assert ex.gold_unsupported_claims == ["claim A", "claim B"]

    def test_unsupported_claims_fallback(self):
        d = {"id": "x", "gold_unsupported_claims": ["claim Z"]}
        ex = Example.from_dict(d)
        assert ex.gold_unsupported_claims == ["claim Z"]

    def test_labels_key_takes_precedence_over_top_level(self):
        # labels.hallucinated overrides gold_hallucinated at top level
        d = {"id": "x", "labels": {"hallucinated": False}, "gold_hallucinated": True}
        ex = Example.from_dict(d)
        assert ex.gold_hallucinated is False

    def test_id_cast_to_str(self):
        d = {"id": 99}
        ex = Example.from_dict(d)
        assert ex.id == "99"

    def test_missing_id_defaults_to_empty_string(self):
        ex = Example.from_dict({})
        assert ex.id == ""

    def test_meta_passthrough(self):
        d = {"id": "m", "meta": {"source": "wiki"}}
        ex = Example.from_dict(d)
        assert ex.meta == {"source": "wiki"}

    def test_reference_passthrough(self):
        d = {"id": "r", "reference": "gold answer"}
        ex = Example.from_dict(d)
        assert ex.reference == "gold answer"


# ---------------------------------------------------------------------------
# Detection properties
# ---------------------------------------------------------------------------

class TestDetection:
    def _make_detection(self, supports: list[Support]) -> Detection:
        claims = [Claim(text=f"claim {i}", index=i) for i in range(len(supports))]
        verdicts = [Verdict(claim=c, support=s) for c, s in zip(claims, supports)]
        ex = Example(id="test", answer="irrelevant")
        return Detection(example=ex, verdicts=verdicts, detector="test-det")

    def test_predicted_hallucinated_all_supported(self):
        det = self._make_detection([Support.SUPPORTED, Support.SUPPORTED])
        assert det.predicted_hallucinated is False

    def test_predicted_hallucinated_one_unverifiable(self):
        det = self._make_detection([Support.SUPPORTED, Support.UNVERIFIABLE])
        assert det.predicted_hallucinated is True

    def test_predicted_hallucinated_one_contradicted(self):
        det = self._make_detection([Support.SUPPORTED, Support.CONTRADICTED])
        assert det.predicted_hallucinated is True

    def test_groundedness_all_supported(self):
        det = self._make_detection([Support.SUPPORTED, Support.SUPPORTED])
        assert det.groundedness == 1.0

    def test_groundedness_mixed(self):
        det = self._make_detection([Support.SUPPORTED, Support.UNVERIFIABLE])
        assert det.groundedness == pytest.approx(0.5)

    def test_groundedness_empty_verdicts(self):
        det = self._make_detection([])
        assert det.groundedness == 1.0

    def test_hallucinated_claims_returns_only_bad(self):
        det = self._make_detection([Support.SUPPORTED, Support.CONTRADICTED, Support.UNVERIFIABLE])
        assert len(det.hallucinated_claims) == 2
        for v in det.hallucinated_claims:
            assert v.is_hallucination

    def test_claims_property(self):
        det = self._make_detection([Support.SUPPORTED, Support.UNVERIFIABLE])
        assert len(det.claims) == 2
        assert all(isinstance(c, Claim) for c in det.claims)
