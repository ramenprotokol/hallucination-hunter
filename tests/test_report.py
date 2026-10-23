"""Tests for hh.report: to_json, to_markdown."""

from __future__ import annotations

import json

from hh.detectors.base import load_detector
from hh.eval import EvalRun, run_eval
from hh.metrics import evaluate
from hh.report import to_json, to_markdown
from hh.types import Claim, Detection, Example, Support, Verdict


def _make_run() -> EvalRun:
    examples = [
        Example(
            id="r1",
            context="Paris is the capital of France.",
            answer="Paris is the capital of France.",
            gold_hallucinated=False,
        ),
        Example(
            id="r2",
            context="Paris is the capital of France.",
            answer="London is the capital of France.",
            gold_hallucinated=True,
        ),
    ]
    detector = load_detector("groundedness", verifier="lexical")
    return run_eval(examples, detector)


class TestToJson:
    def test_returns_valid_json(self):
        run = _make_run()
        raw = to_json(run)
        parsed = json.loads(raw)
        assert isinstance(parsed, dict)

    def test_json_contains_required_keys(self):
        run = _make_run()
        parsed = json.loads(to_json(run))
        assert "hallucination_rate" in parsed
        assert "mean_groundedness" in parsed
        assert "n_examples" in parsed
        assert "detector" in parsed

    def test_json_n_examples_correct(self):
        run = _make_run()
        parsed = json.loads(to_json(run))
        assert parsed["n_examples"] == 2

    def test_json_detection_quality_present_when_labeled(self):
        run = _make_run()
        parsed = json.loads(to_json(run))
        assert "detection_quality" in parsed


class TestToJsonDetections:
    """JSON carries the same per-claim detail as the markdown report."""

    def test_json_carries_every_detection(self):
        parsed = json.loads(to_json(_make_run()))
        assert len(parsed["detections"]) == 2
        first = parsed["detections"][0]
        assert first["id"] == "r1"
        assert first["predicted_hallucinated"] is False
        assert 0.0 <= first["groundedness"] <= 1.0
        v = first["verdicts"][0]
        assert set(v) == {"claim", "support", "score", "rationale", "evidence", "trace"}
        assert v["claim"] == "Paris is the capital of France."
        assert v["support"] in ("supported", "contradicted", "unverifiable")
        assert v["trace"] == []

    def test_json_verdict_detail_is_untruncated(self):
        det = Detection(
            example=Example(id="d1", answer="A claim."),
            verdicts=[
                Verdict(
                    claim=Claim("A claim."),
                    support=Support.UNVERIFIABLE,
                    score=0.5,
                    rationale="no evidence quote verified against the sources",
                    evidence='"Bicycles cost $3 each way." (02_fares.md)',
                    trace=("search_docs(x) -> 1 hit", "read_doc(y) -> text"),
                )
            ],
        )
        run = EvalRun(result=_make_run().result, detections=[det])
        v = json.loads(to_json(run))["detections"][0]["verdicts"][0]
        assert v["score"] == 0.5
        assert v["rationale"] == "no evidence quote verified against the sources"
        assert v["evidence"] == '"Bicycles cost $3 each way." (02_fares.md)'
        assert v["trace"] == ["search_docs(x) -> 1 hit", "read_doc(y) -> text"]

    def test_json_empty_detections_is_an_empty_list(self):
        run = EvalRun(result=evaluate([]), detections=[])
        parsed = json.loads(to_json(run))
        assert parsed["detections"] == []
        assert parsed["n_examples"] == 0


class TestToMarkdown:
    def test_contains_detector_name(self):
        run = _make_run()
        md = to_markdown(run)
        assert "groundedness" in md

    def test_contains_metrics_table(self):
        run = _make_run()
        md = to_markdown(run)
        # The markdown table separator
        assert "---" in md

    def test_contains_precision_recall_f1(self):
        run = _make_run()
        md = to_markdown(run)
        assert "Precision" in md
        assert "Recall" in md
        assert "F1" in md

    def test_contains_title(self):
        run = _make_run()
        md = to_markdown(run, title="My Test Report")
        assert "My Test Report" in md

    def test_contains_example_ids(self):
        run = _make_run()
        md = to_markdown(run)
        assert "r1" in md

    def test_is_string(self):
        run = _make_run()
        assert isinstance(to_markdown(run), str)


class TestConfidenceDisplay:
    """Verdict confidence shows in the report when it is not a full call."""

    def _markdown_with_score(self, score: float) -> str:
        det = Detection(
            example=Example(id="conf", answer="A claim."),
            verdicts=[Verdict(claim=Claim("A claim."), support=Support.UNVERIFIABLE, score=score)],
        )
        run = EvalRun(result=_make_run().result, detections=[det])
        return to_markdown(run, show_examples=1)

    def test_below_full_confidence_is_shown(self):
        assert "(confidence 50%)" in self._markdown_with_score(0.5)

    def test_full_confidence_is_quiet(self):
        assert "confidence" not in self._markdown_with_score(1.0)
