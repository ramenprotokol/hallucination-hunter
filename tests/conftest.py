"""Shared fixtures for hallucination-hunter tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hh.models.mock import MockModel
from hh.types import Claim, Example, Support, Verdict


@pytest.fixture
def mock_model():
    """A plain MockModel with no pre-loaded fixtures."""
    return MockModel()


@pytest.fixture
def supported_example():
    """Example whose answer is fully grounded in the context."""
    return Example(
        id="supported-1",
        context="The sky is blue. Water is wet.",
        answer="The sky is blue.",
    )


@pytest.fixture
def hallucinated_example():
    """Example whose answer contains a claim not in the context."""
    return Example(
        id="hallucinated-1",
        context="The sky is blue.",
        answer="The sky is green and made of cheese.",
    )


@pytest.fixture
def tiny_dataset(tmp_path: Path) -> Path:
    """Write a tiny 2-row JSONL file and return its path."""
    rows = [
        {
            "id": "1",
            "context": "Paris is the capital of France.",
            "answer": "Paris is the capital of France.",
            "labels": {"hallucinated": False},
        },
        {
            "id": "2",
            "context": "Paris is the capital of France.",
            "answer": "London is the capital of France.",
            "labels": {"hallucinated": True},
        },
    ]
    p = tmp_path / "dataset.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    return p
