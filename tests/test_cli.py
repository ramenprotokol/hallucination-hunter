"""Tests for hh.cli: main() entry point."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hh.cli import main
from hh.types import Claim, Detection, Support, Verdict

# ---------------------------------------------------------------------------
# hh list
# ---------------------------------------------------------------------------


class TestCLIList:
    def test_list_returns_zero(self, capsys):
        rc = main(["list"])
        assert rc == 0

    def test_list_prints_detectors(self, capsys):
        main(["list"])
        captured = capsys.readouterr()
        assert "groundedness" in captured.out

    def test_list_prints_model_backends(self, capsys):
        main(["list"])
        captured = capsys.readouterr()
        assert "mock" in captured.out


# ---------------------------------------------------------------------------
# hh --version
# ---------------------------------------------------------------------------


class TestCLIVersion:
    def test_version_raises_system_exit_0(self):
        with pytest.raises(SystemExit) as exc_info:
            main(["--version"])
        assert exc_info.value.code == 0


# ---------------------------------------------------------------------------
# hh detect
# ---------------------------------------------------------------------------


class TestCLIDetect:
    def test_detect_json_format_returns_valid_json(self, capsys):
        rc = main(
            [
                "detect",
                "--answer",
                "Paris is the capital of France.",
                "--context",
                "Paris is the capital of France.",
                "--format",
                "json",
            ]
        )
        assert rc == 0
        captured = capsys.readouterr()
        parsed = json.loads(captured.out)
        assert "n_examples" in parsed

    def test_detect_markdown_format(self, capsys):
        rc = main(
            [
                "detect",
                "--answer",
                "The sky is blue.",
                "--context",
                "The sky is blue and clear.",
                "--format",
                "markdown",
            ]
        )
        assert rc == 0
        captured = capsys.readouterr()
        assert "groundedness" in captured.out

    def test_detect_no_context(self, capsys):
        rc = main(
            [
                "detect",
                "--answer",
                "Some statement here.",
                "--format",
                "json",
            ]
        )
        assert rc == 0

    def test_detect_with_prompt(self, capsys):
        rc = main(
            [
                "detect",
                "--answer",
                "Paris is in France.",
                "--context",
                "Paris is in France.",
                "--prompt",
                "Where is Paris?",
                "--format",
                "json",
            ]
        )
        assert rc == 0


# ---------------------------------------------------------------------------
# hh eval
# ---------------------------------------------------------------------------


class TestCLIEval:
    def _write_dataset(self, tmp_path: Path) -> Path:
        rows = [
            {
                "id": "1",
                "context": "Water is wet.",
                "answer": "Water is wet.",
                "labels": {"hallucinated": False},
            },
            {
                "id": "2",
                "context": "Water is wet.",
                "answer": "Fire is wet and purple.",
                "labels": {"hallucinated": True},
            },
        ]
        p = tmp_path / "tiny.jsonl"
        p.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
        return p

    def test_eval_markdown_returns_zero(self, tmp_path, capsys):
        dataset = self._write_dataset(tmp_path)
        rc = main(
            [
                "eval",
                "--dataset",
                str(dataset),
                "--format",
                "markdown",
            ]
        )
        assert rc == 0

    def test_eval_json_valid_json(self, tmp_path, capsys):
        dataset = self._write_dataset(tmp_path)
        main(
            [
                "eval",
                "--dataset",
                str(dataset),
                "--format",
                "json",
            ]
        )
        captured = capsys.readouterr()
        parsed = json.loads(captured.out)
        assert parsed["n_examples"] == 2

    def test_eval_markdown_contains_detector_name(self, tmp_path, capsys):
        dataset = self._write_dataset(tmp_path)
        main(
            [
                "eval",
                "--dataset",
                str(dataset),
                "--format",
                "markdown",
            ]
        )
        captured = capsys.readouterr()
        assert "groundedness" in captured.out

    def test_eval_output_to_file(self, tmp_path, capsys):
        dataset = self._write_dataset(tmp_path)
        out_file = tmp_path / "report.json"
        main(
            [
                "eval",
                "--dataset",
                str(dataset),
                "--format",
                "json",
                "--out",
                str(out_file),
            ]
        )
        assert out_file.exists()
        parsed = json.loads(out_file.read_text(encoding="utf-8"))
        assert "n_examples" in parsed


# ---------------------------------------------------------------------------
# hh investigate
# ---------------------------------------------------------------------------

INVESTIGATE_DOCS = Path(__file__).parent / "fixtures" / "investigate" / "sources"


class TestCLIInvestigate:
    def test_investigate_json_returns_valid_json(self, capsys):
        rc = main(
            [
                "investigate",
                "--answer",
                "An adult return ticket costs $14.",
                "--prompt",
                "What does a return cost?",
                "--docs",
                str(INVESTIGATE_DOCS),
                "--format",
                "json",
            ]
        )
        assert rc == 0
        captured = capsys.readouterr()
        parsed = json.loads(captured.out)
        assert parsed["detector"] == "investigate:mock"
        assert parsed["n_examples"] == 1

    def test_investigate_markdown_contains_agent_name(self, capsys):
        rc = main(
            [
                "investigate",
                "--answer",
                "The crossing takes 40 minutes.",
                "--context",
                "A crossing takes 40 minutes.",
                "--format",
                "markdown",
            ]
        )
        assert rc == 0
        captured = capsys.readouterr()
        assert "investigate" in captured.out

    def test_investigate_respects_step_budget(self, capsys):
        rc = main(
            [
                "investigate",
                "--answer",
                "The sky is blue.",
                "--max-steps",
                "1",
                "--format",
                "json",
            ]
        )
        assert rc == 0

    def test_eval_with_investigate_detector(self, tmp_path, capsys):
        rows = [{"id": "1", "answer": "The sky is blue.", "context": "The sky is blue."}]
        dataset = tmp_path / "ds.jsonl"
        dataset.write_text(json.dumps(rows[0]), encoding="utf-8")
        rc = main(
            [
                "eval",
                "--dataset",
                str(dataset),
                "--detector",
                "investigate",
                "--docs",
                str(INVESTIGATE_DOCS),
                "--format",
                "json",
            ]
        )
        assert rc == 0
        parsed = json.loads(capsys.readouterr().out)
        assert parsed["detector"] == "investigate:mock"

    def _write_answers(self, tmp_path: Path) -> Path:
        rows = [
            {
                "id": "a1",
                "question": "What does a return cost?",
                "answer": "An adult return ticket costs $14.",
            },
            {
                "id": "a2",
                "question": "Tell me about the boats.",
                "answer": "The current vessel, the Gannet, was built in Sunderland.",
            },
            # no answer -> nothing to investigate, skipped
            {"id": "a3", "question": "Empty row?", "answer": ""},
        ]
        p = tmp_path / "answers.jsonl"
        p.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
        return p

    def test_investigate_answers_file_runs_every_row(self, tmp_path, capsys):
        answers = self._write_answers(tmp_path)
        rc = main(
            [
                "investigate",
                "--answers",
                str(answers),
                "--docs",
                str(INVESTIGATE_DOCS),
                "--format",
                "json",
            ]
        )
        assert rc == 0
        parsed = json.loads(capsys.readouterr().out)
        assert parsed["detector"] == "investigate:mock"
        assert parsed["n_examples"] == 2  # the answer-less row is skipped
        # JSON now carries per-claim detail, same as markdown
        assert [d["id"] for d in parsed["detections"]] == ["a1", "a2"]
        first_verdict = parsed["detections"][0]["verdicts"][0]
        assert first_verdict["claim"]
        assert first_verdict["support"] in ("supported", "contradicted", "unverifiable")

    def test_investigate_answers_markdown_shows_all_rows(self, tmp_path, capsys):
        answers = self._write_answers(tmp_path)
        rc = main(
            [
                "investigate",
                "--answers",
                str(answers),
                "--docs",
                str(INVESTIGATE_DOCS),
                "--format",
                "markdown",
            ]
        )
        assert rc == 0
        out = capsys.readouterr().out
        # both investigated rows appear; the skipped row does not
        assert "a1" in out
        assert "a2" in out
        assert "a3" not in out
        # question doubles as the prompt in the report
        assert "What does a return cost?" in out

    def test_investigate_answers_respects_show_limit(self, tmp_path, capsys):
        answers = self._write_answers(tmp_path)
        main(
            [
                "investigate",
                "--answers",
                str(answers),
                "--docs",
                str(INVESTIGATE_DOCS),
                "--show",
                "1",
                "--format",
                "markdown",
            ]
        )
        out = capsys.readouterr().out
        assert "a1" in out
        assert "a2" not in out

    def test_investigate_requires_answer_or_answers(self, capsys):
        with pytest.raises(SystemExit) as exc_info:
            main(["investigate", "--format", "json"])
        assert "--answer" in str(exc_info.value)

    def test_investigate_rejects_answer_and_answers_together(self, tmp_path):
        answers = self._write_answers(tmp_path)
        with pytest.raises(SystemExit) as exc_info:
            main(["investigate", "--answer", "x", "--answers", str(answers)])
        assert "not both" in str(exc_info.value)

    def test_investigate_answers_bad_json_row_errors_with_location(self, tmp_path):
        p = tmp_path / "answers.jsonl"
        p.write_text('{"id": "ok", "answer": "fine."}\n{not json', encoding="utf-8")
        with pytest.raises(SystemExit) as exc_info:
            main(["investigate", "--answers", str(p)])
        assert "answers.jsonl:2" in str(exc_info.value)

    def test_investigate_answers_empty_file_errors(self, tmp_path):
        p = tmp_path / "empty.jsonl"
        p.write_text('{"id": "x", "answer": ""}\n', encoding="utf-8")
        with pytest.raises(SystemExit) as exc_info:
            main(["investigate", "--answers", str(p)])
        assert "no answers" in str(exc_info.value)

    def test_investigate_answers_prints_progress(self, tmp_path, capsys):
        answers = self._write_answers(tmp_path)
        rc = main(
            [
                "investigate",
                "--answers",
                str(answers),
                "--docs",
                str(INVESTIGATE_DOCS),
                "--format",
                "json",
            ]
        )
        assert rc == 0
        captured = capsys.readouterr()
        assert "[a1] 1/2" in captured.err
        assert "[a2] 2/2" in captured.err
        assert "claims" in captured.err
        assert "batch: 2/2 answers investigated, 0 failed" in captured.err

    def test_investigate_answers_survives_a_failing_answer(self, tmp_path, capsys, monkeypatch):
        answers = self._write_answers(tmp_path)

        class FlakyDetector:
            name = "flaky"

            def detect(self, ex):
                if ex.id == "a2":
                    raise RuntimeError("model API exploded on answer 2")
                verdict = Verdict(claim=Claim(ex.answer), support=Support.SUPPORTED)
                return Detection(example=ex, verdicts=[verdict], detector="flaky")

            def detect_many(self, examples):
                return [self.detect(e) for e in examples]

        monkeypatch.setattr("hh.cli.load_detector", lambda name, **kw: FlakyDetector())
        rc = main(
            [
                "investigate",
                "--answers",
                str(answers),
                "--docs",
                str(INVESTIGATE_DOCS),
                "--format",
                "json",
            ]
        )
        assert rc == 1  # one answer failed — signal it
        captured = capsys.readouterr()
        parsed = json.loads(captured.out)
        assert parsed["n_examples"] == 1  # a1 survived, a2 skipped
        assert "FAILED" in captured.err
        assert "model API exploded on answer 2" in captured.err
        assert "batch: 1/2 answers investigated, 1 failed" in captured.err
