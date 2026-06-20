"""Tests for hh.cli: main() entry point."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from hh.cli import main


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
        rc = main([
            "detect",
            "--answer", "Paris is the capital of France.",
            "--context", "Paris is the capital of France.",
            "--format", "json",
        ])
        assert rc == 0
        captured = capsys.readouterr()
        parsed = json.loads(captured.out)
        assert "n_examples" in parsed

    def test_detect_markdown_format(self, capsys):
        rc = main([
            "detect",
            "--answer", "The sky is blue.",
            "--context", "The sky is blue and clear.",
            "--format", "markdown",
        ])
        assert rc == 0
        captured = capsys.readouterr()
        assert "groundedness" in captured.out

    def test_detect_no_context(self, capsys):
        rc = main([
            "detect",
            "--answer", "Some statement here.",
            "--format", "json",
        ])
        assert rc == 0

    def test_detect_with_prompt(self, capsys):
        rc = main([
            "detect",
            "--answer", "Paris is in France.",
            "--context", "Paris is in France.",
            "--prompt", "Where is Paris?",
            "--format", "json",
        ])
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
        rc = main([
            "eval",
            "--dataset", str(dataset),
            "--format", "markdown",
        ])
        assert rc == 0

    def test_eval_json_valid_json(self, tmp_path, capsys):
        dataset = self._write_dataset(tmp_path)
        main([
            "eval",
            "--dataset", str(dataset),
            "--format", "json",
        ])
        captured = capsys.readouterr()
        parsed = json.loads(captured.out)
        assert parsed["n_examples"] == 2

    def test_eval_markdown_contains_detector_name(self, tmp_path, capsys):
        dataset = self._write_dataset(tmp_path)
        main([
            "eval",
            "--dataset", str(dataset),
            "--format", "markdown",
        ])
        captured = capsys.readouterr()
        assert "groundedness" in captured.out

    def test_eval_output_to_file(self, tmp_path, capsys):
        dataset = self._write_dataset(tmp_path)
        out_file = tmp_path / "report.json"
        main([
            "eval",
            "--dataset", str(dataset),
            "--format", "json",
            "--out", str(out_file),
        ])
        assert out_file.exists()
        parsed = json.loads(out_file.read_text(encoding="utf-8"))
        assert "n_examples" in parsed
