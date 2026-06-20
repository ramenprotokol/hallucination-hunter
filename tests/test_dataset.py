"""Tests for hh.dataset: load_dataset, save_dataset, iter_jsonl."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hh.dataset import load_dataset, save_dataset
from hh.types import Example

# ---------------------------------------------------------------------------
# Round-trip: save_dataset then load_dataset
# ---------------------------------------------------------------------------


class TestRoundTrip:
    def test_basic_round_trip(self, tmp_path: Path):
        examples = [
            Example(id="1", context="ctx1", answer="ans1"),
            Example(id="2", context="ctx2", answer="ans2"),
        ]
        p = tmp_path / "out.jsonl"
        save_dataset(examples, p)
        loaded = load_dataset(p)
        assert len(loaded) == 2
        assert loaded[0].id == "1"
        assert loaded[0].context == "ctx1"
        assert loaded[0].answer == "ans1"
        assert loaded[1].id == "2"

    def test_gold_labels_round_trip(self, tmp_path: Path):
        examples = [
            Example(
                id="a", answer="x", gold_hallucinated=True, gold_unsupported_claims=["claim X"]
            ),
        ]
        p = tmp_path / "labeled.jsonl"
        save_dataset(examples, p)
        loaded = load_dataset(p)
        assert loaded[0].gold_hallucinated is True
        assert loaded[0].gold_unsupported_claims == ["claim X"]

    def test_reference_field_round_trip(self, tmp_path: Path):
        examples = [Example(id="r", answer="a", reference="gold ref")]
        p = tmp_path / "ref.jsonl"
        save_dataset(examples, p)
        loaded = load_dataset(p)
        assert loaded[0].reference == "gold ref"

    def test_no_labels_omits_labels_key(self, tmp_path: Path):
        examples = [Example(id="nl", answer="x")]
        p = tmp_path / "nolabels.jsonl"
        save_dataset(examples, p)
        raw = p.read_text(encoding="utf-8")
        data = json.loads(raw.strip())
        assert "labels" not in data


# ---------------------------------------------------------------------------
# Comment and blank line handling
# ---------------------------------------------------------------------------


class TestSpecialLines:
    def test_blank_lines_skipped(self, tmp_path: Path):
        p = tmp_path / "blanks.jsonl"
        lines = [
            json.dumps({"id": "1", "answer": "a"}),
            "",
            json.dumps({"id": "2", "answer": "b"}),
            "   ",
        ]
        p.write_text("\n".join(lines), encoding="utf-8")
        loaded = load_dataset(p)
        assert len(loaded) == 2

    def test_comment_lines_skipped(self, tmp_path: Path):
        p = tmp_path / "comments.jsonl"
        lines = [
            "// This is a comment",
            json.dumps({"id": "1", "answer": "x"}),
            "// Another comment",
            json.dumps({"id": "2", "answer": "y"}),
        ]
        p.write_text("\n".join(lines), encoding="utf-8")
        loaded = load_dataset(p)
        assert len(loaded) == 2
        assert loaded[0].id == "1"
        assert loaded[1].id == "2"


# ---------------------------------------------------------------------------
# Malformed JSON
# ---------------------------------------------------------------------------


class TestMalformedJSON:
    def test_malformed_json_raises_value_error(self, tmp_path: Path):
        p = tmp_path / "bad.jsonl"
        p.write_text('{"id": "ok"}\nNOT_JSON\n', encoding="utf-8")
        with pytest.raises(ValueError, match="invalid JSON"):
            load_dataset(p)

    def test_error_message_contains_line_number(self, tmp_path: Path):
        p = tmp_path / "bad2.jsonl"
        p.write_text('{"id": "good"}\nBAD LINE\n', encoding="utf-8")
        with pytest.raises(ValueError, match="2"):
            load_dataset(p)
