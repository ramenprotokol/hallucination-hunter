"""Tests for all three detector implementations and the detector registry."""

from __future__ import annotations

import json

import pytest

from hh.detectors.base import available_detectors, load_detector
from hh.detectors.groundedness import GroundednessDetector, LexicalVerifier, LLMVerifier
from hh.detectors.self_consistency import SelfConsistencyDetector
from hh.models.mock import MockModel
from hh.types import Claim, Example, Support

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class TestDetectorRegistry:
    def test_available_detectors_includes_groundedness(self):
        assert "groundedness" in available_detectors()

    def test_available_detectors_includes_self_consistency(self):
        assert "self_consistency" in available_detectors()

    def test_available_detectors_includes_reference_judge(self):
        assert "reference_judge" in available_detectors()

    def test_available_detectors_sorted(self):
        ds = available_detectors()
        assert ds == sorted(ds)

    def test_load_detector_groundedness(self):
        d = load_detector("groundedness")
        assert isinstance(d, GroundednessDetector)

    def test_load_detector_unknown_raises(self):
        with pytest.raises(ValueError, match="unknown detector"):
            load_detector("totally_unknown_detector_xyz")


# ---------------------------------------------------------------------------
# LexicalVerifier
# ---------------------------------------------------------------------------


class TestLexicalVerifier:
    def setup_method(self):
        self.verifier = LexicalVerifier()

    def test_supported_when_claim_words_in_context(self):
        claim = Claim("Paris is the capital of France.", 0)
        context = "Paris is the capital of France and a major European city."
        verdict = self.verifier.verify(claim, context)
        assert verdict.support == Support.SUPPORTED

    def test_unverifiable_when_claim_words_absent(self):
        claim = Claim("Elephants can fly at supersonic speed.", 0)
        context = "Paris is the capital of France."
        verdict = self.verifier.verify(claim, context)
        assert verdict.support == Support.UNVERIFIABLE

    def test_contradicted_negation_mismatch(self):
        # Claim asserts negation; context has the same content words but no negation.
        # High overlap + negation mismatch -> CONTRADICTED
        claim = Claim("The sky is not blue.", 0)
        context = "The sky is blue and clear today."
        verdict = self.verifier.verify(claim, context)
        assert verdict.support == Support.CONTRADICTED

    def test_empty_claim_returns_unverifiable(self):
        claim = Claim("", 0)
        context = "Some context text here."
        verdict = self.verifier.verify(claim, context)
        assert verdict.support == Support.UNVERIFIABLE


# ---------------------------------------------------------------------------
# GroundednessDetector (lexical)
# ---------------------------------------------------------------------------


class TestGroundednessDetectorLexical:
    def test_grounded_answer(self):
        detector = load_detector("groundedness", verifier="lexical")
        ex = Example(
            id="g1",
            context="The Eiffel Tower is in Paris, France.",
            answer="The Eiffel Tower is in Paris.",
        )
        det = detector.detect(ex)
        assert det.predicted_hallucinated is False
        assert det.groundedness == pytest.approx(1.0)

    def test_hallucinated_answer(self):
        detector = load_detector("groundedness", verifier="lexical")
        ex = Example(
            id="h1",
            context="The Eiffel Tower is in Paris, France.",
            answer="The Eiffel Tower is in Berlin and was built in 1889.",
        )
        det = detector.detect(ex)
        # "Berlin" is not in context -> at least one claim hallucinated
        assert det.predicted_hallucinated is True

    def test_detector_name_in_result(self):
        detector = load_detector("groundedness", verifier="lexical")
        ex = Example(id="n1", context="x", answer="x")
        det = detector.detect(ex)
        assert "groundedness" in det.detector

    def test_verdicts_contain_claims(self):
        detector = load_detector("groundedness", verifier="lexical")
        ex = Example(
            id="v1",
            context="Cats are mammals. Dogs are mammals.",
            answer="Cats are mammals. Dogs are mammals.",
        )
        det = detector.detect(ex)
        assert len(det.verdicts) >= 1


# ---------------------------------------------------------------------------
# GroundednessDetector (llm verifier with MockModel)
# ---------------------------------------------------------------------------


class TestGroundednessDetectorLLM:
    def _make_llm_detector_with_fixture(self, claim_text, context_text, support_value):
        model = MockModel()
        prompt = LLMVerifier.PROMPT.format(context=context_text, claim=claim_text)
        model.add_fixture(
            prompt, json.dumps({"support": support_value, "rationale": "test rationale"})
        )
        return load_detector("groundedness", verifier="llm", model=model)

    def test_llm_verifier_supported(self):
        detector = self._make_llm_detector_with_fixture(
            "Paris is in France.", "Paris is in France.", "supported"
        )
        ex = Example(id="llm1", context="Paris is in France.", answer="Paris is in France.")
        det = detector.detect(ex)
        assert det.groundedness == pytest.approx(1.0)

    def test_llm_verifier_unverifiable(self):
        detector = self._make_llm_detector_with_fixture(
            "Paris is in France.", "Paris is in France.", "unverifiable"
        )
        ex = Example(id="llm2", context="Paris is in France.", answer="Paris is in France.")
        det = detector.detect(ex)
        assert det.predicted_hallucinated is True

    @pytest.mark.parametrize(
        "raw",
        [
            "The claim is not supported by the context.",
            '{"support": "unsupported", "rationale": "no evidence"}',
            "Unsupported.",
        ],
    )
    def test_negated_support_is_not_read_as_supported(self, raw):
        # "supported" is a substring of "unsupported" / "not supported"; a judge
        # saying the claim lacks support must not come back as SUPPORTED.
        model = MockModel()
        prompt = LLMVerifier.PROMPT.format(context="Paris is in France.", claim="Paris is big.")
        model.add_fixture(prompt, raw)
        detector = load_detector("groundedness", verifier="llm", model=model)
        det = detector.detect(
            Example(id="neg", context="Paris is in France.", answer="Paris is big.")
        )
        assert det.predicted_hallucinated is True

    def test_llm_verifier_no_model_raises(self):
        with pytest.raises(ValueError, match="requires a model"):
            load_detector("groundedness", verifier="llm")

    def test_unknown_verifier_raises(self):
        with pytest.raises(ValueError, match="unknown verifier"):
            load_detector("groundedness", verifier="magic_ai")


# ---------------------------------------------------------------------------
# SelfConsistencyDetector
# ---------------------------------------------------------------------------


class TestSelfConsistencyDetector:
    def test_basic_run_completes(self):
        model = MockModel()
        # Pre-seed the model so samples are deterministic
        model.add_fixture(
            "What color is the sky?", "The sky is blue. It appears blue due to Rayleigh scattering."
        )
        detector = SelfConsistencyDetector(model=model, samples=3, temperature=0.8)
        ex = Example(
            id="sc1",
            prompt="What color is the sky?",
            answer="The sky is blue.",
        )
        det = detector.detect(ex)
        assert det is not None
        assert "self_consistency" in det.detector

    def test_requires_model(self):
        with pytest.raises(ValueError, match="requires a model"):
            SelfConsistencyDetector(model=None)

    def test_samples_count_respected(self):
        model = MockModel()
        model.add_fixture("Prompt text.", "The answer.")
        detector = SelfConsistencyDetector(model=model, samples=2)
        ex = Example(id="sc2", prompt="Prompt text.", answer="The answer.")
        detector.detect(ex)
        # The model is called: once for each sample (the answer is pre-set, so no
        # generation call). Samples=2 means 2 generate calls from _sample.
        assert len(model.calls) == 2

    def test_no_answer_generates_one(self):
        model = MockModel()
        model.add_fixture("My prompt.", "Generated answer here.")
        detector = SelfConsistencyDetector(model=model, samples=2)
        ex = Example(id="sc3", prompt="My prompt.", answer="")
        det = detector.detect(ex)
        # answer should now be filled in (from model.generate)
        assert det.example.answer != ""


# ---------------------------------------------------------------------------
# ReferenceJudgeDetector
# ---------------------------------------------------------------------------


class TestReferenceJudgeDetector:
    def test_raises_when_no_reference(self):
        detector = load_detector("reference_judge", verifier="lexical")
        ex = Example(id="rj1", context="ctx", answer="answer", reference="")
        with pytest.raises(ValueError, match="no reference answer"):
            detector.detect(ex)

    def test_uses_reference_as_context(self):
        detector = load_detector("reference_judge", verifier="lexical")
        ex = Example(
            id="rj2",
            answer="Paris is the capital of France.",
            reference="Paris is the capital of France and a lovely city.",
        )
        det = detector.detect(ex)
        # The original example should be preserved on the detection
        assert det.example.id == "rj2"
        assert "reference_judge" in det.detector

    def test_hallucinated_against_reference(self):
        # The lexical verifier uses token overlap; "London is the capital of
        # France." shares "capital" and "France" with the reference, giving
        # ~0.67 overlap which is SUPPORTED by the lexical baseline.
        # Use an answer that shares NO content words with the reference to get
        # an UNVERIFIABLE verdict reliably.
        detector = load_detector("reference_judge", verifier="lexical")
        ex = Example(
            id="rj3",
            answer="Penguins swim underwater in Antarctica.",
            reference="Paris is the capital of France.",
        )
        det = detector.detect(ex)
        assert det.predicted_hallucinated is True
