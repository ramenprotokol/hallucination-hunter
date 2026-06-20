"""Tests for hh.report: to_json, to_markdown."""

from __future__ import annotations

import json

import pytest

from hh.detectors.base import load_detector
from hh.eval import EvalRun, run_eval
from hh.report import to_json, to_markdown
from hh.types import Example


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
