"""Tests for hh.eval: run_eval, EvalRun."""

from __future__ import annotations

from hh.detectors.base import load_detector
from hh.eval import EvalRun, run_eval
from hh.models.mock import MockModel
from hh.types import Example

# ---------------------------------------------------------------------------
# run_eval with a lexical groundedness detector (no model needed)
# ---------------------------------------------------------------------------


class TestRunEvalLexical:
    def _detector(self):
        return load_detector("groundedness", verifier="lexical")

    def test_returns_eval_run(self):
        examples = [
            Example(id="1", context="The sky is blue.", answer="The sky is blue."),
        ]
        run = run_eval(examples, self._detector())
        assert isinstance(run, EvalRun)

    def test_result_has_correct_n_examples(self):
        examples = [
            Example(
                id="1",
                context="Paris is the capital of France.",
                answer="Paris is the capital of France.",
            ),
            Example(id="2", context="Water is wet.", answer="Fire is hot."),
        ]
        run = run_eval(examples, self._detector())
        assert run.result.n_examples == 2

    def test_detections_length_matches_examples(self):
        examples = [
            Example(id="a", context="x is y.", answer="x is y."),
            Example(id="b", context="a is b.", answer="a is b."),
            Example(id="c", context="c is d.", answer="c is d."),
        ]
        run = run_eval(examples, self._detector())
        assert len(run.detections) == 3

    def test_hallucination_rate_fully_grounded(self):
        examples = [
            Example(id="1", context="Water is wet.", answer="Water is wet."),
            Example(id="2", context="The sky is blue.", answer="The sky is blue."),
        ]
        run = run_eval(examples, self._detector())
        # Both answers should be grounded -> hallucination_rate == 0.0
        assert run.result.hallucination_rate == 0.0

    def test_confusion_matrix_present_when_gold_labels(self):
        examples = [
            Example(
                id="1",
                context="Paris is in France.",
                answer="Paris is in France.",
                gold_hallucinated=False,
            ),
            Example(
                id="2",
                context="Paris is in France.",
                answer="Berlin is in France.",
                gold_hallucinated=True,
            ),
        ]
        run = run_eval(examples, self._detector())
        assert run.result.confusion is not None


# ---------------------------------------------------------------------------
# run_eval with a generator (MockModel filling in answers)
# ---------------------------------------------------------------------------


class TestRunEvalWithGenerator:
    def test_generator_fills_missing_answers(self):
        model = MockModel()
        model.add_fixture("Tell me about water.", "Water is wet and liquid.")
        detector = load_detector("groundedness", verifier="lexical")
        examples = [
            Example(
                id="g1",
                prompt="Tell me about water.",
                context="Water is wet and liquid.",
                answer="",
            ),
        ]
        run = run_eval(examples, detector, generator=model)
        # The detection's example should have an answer now
        assert run.detections[0].example.answer != ""

    def test_generator_not_called_when_answer_exists(self):
        model = MockModel()
        detector = load_detector("groundedness", verifier="lexical")
        examples = [
            Example(id="g2", prompt="Ignored?", context="ctx", answer="Already have an answer."),
        ]
        run_eval(examples, detector, generator=model)
        # Model should not have been called at all
        assert model.calls == []

    def test_no_generator_none_leaves_empty_answer(self):
        detector = load_detector("groundedness", verifier="lexical")
        examples = [
            Example(id="g3", prompt="Tell me something.", context="", answer=""),
        ]
        # Should not raise even with empty answer
        run = run_eval(examples, detector, generator=None)
        assert run.result.n_examples == 1
