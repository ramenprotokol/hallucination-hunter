"""Tests for model backends: MockModel, load_model, available_backends."""

from __future__ import annotations

import os

import pytest

import hh
from hh.models.base import available_backends, load_model
from hh.models.mock import MockModel


# ---------------------------------------------------------------------------
# Registry helpers
# ---------------------------------------------------------------------------

class TestAvailableBackends:
    def test_returns_list(self):
        result = available_backends()
        assert isinstance(result, list)

    def test_includes_mock(self):
        assert "mock" in available_backends()

    def test_includes_anthropic(self):
        assert "anthropic" in available_backends()

    def test_includes_openai(self):
        assert "openai" in available_backends()

    def test_is_sorted(self):
        backends = available_backends()
        assert backends == sorted(backends)


class TestLoadModel:
    def test_load_mock(self):
        m = load_model("mock")
        assert isinstance(m, MockModel)

    def test_unknown_backend_raises_value_error(self):
        with pytest.raises(ValueError, match="unknown model backend"):
            load_model("nope")

    def test_load_model_via_public_api(self):
        m = hh.load_model("mock")
        assert isinstance(m, MockModel)


# ---------------------------------------------------------------------------
# MockModel behaviour
# ---------------------------------------------------------------------------

class TestMockModel:
    def test_exact_fixture_match(self):
        model = MockModel(fixtures={"hello": "world"})
        assert model.generate("hello") == "world"

    def test_digest_fixture_match(self):
        model = MockModel()
        prompt = "some long prompt text"
        digest = MockModel._digest(prompt)
        model.add_fixture(digest, "digest response")
        assert model.generate(prompt) == "digest response"

    def test_add_fixture_method(self):
        model = MockModel()
        model.add_fixture("my prompt", "my response")
        assert model.generate("my prompt") == "my response"

    def test_calls_are_recorded(self):
        model = MockModel()
        model.generate("alpha")
        model.generate("beta")
        assert model.calls == ["alpha", "beta"]

    def test_heuristic_claim_context_prompt(self):
        """Heuristic fires for CLAIM:/CONTEXT: structured prompts."""
        model = MockModel()
        prompt = "CLAIM:\nParis is the capital of France.\n\nCONTEXT:\nParis is the capital of France.\n"
        result = model.generate(prompt)
        # Should return a JSON object with a "support" key
        import json
        data = json.loads(result)
        assert "support" in data
        assert data["support"] in ("supported", "contradicted", "unverifiable")

    def test_heuristic_generic_fallback(self):
        """Generic echo for prompts without CLAIM/CONTEXT headers."""
        model = MockModel()
        result = model.generate("totally unrelated prompt with no structure")
        assert result.startswith("[mock:")

    def test_no_fixtures_returns_deterministic_result(self):
        """Same prompt always returns same thing (deterministic)."""
        model = MockModel()
        r1 = model.generate("some prompt")
        r2 = model.generate("some prompt")
        assert r1 == r2


# ---------------------------------------------------------------------------
# Anthropic: no API key -> RuntimeError on generate
# ---------------------------------------------------------------------------

class TestAnthropicNoKey:
    def test_load_anthropic_succeeds(self):
        """load_model('anthropic') should not raise — client is created lazily."""
        m = load_model("anthropic")
        assert m.name == "anthropic"

    def test_generate_raises_runtime_error_without_key(self, monkeypatch):
        """Calling generate without an API key must raise RuntimeError (no network)."""
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        m = load_model("anthropic")
        with pytest.raises(RuntimeError):
            m.generate("test prompt")


# ---------------------------------------------------------------------------
# OpenAI: no API key -> RuntimeError on generate
# ---------------------------------------------------------------------------

class TestOpenAINoKey:
    def test_load_openai_succeeds(self):
        m = load_model("openai")
        assert m.name == "openai"

    def test_generate_raises_runtime_error_without_key(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        m = load_model("openai")
        with pytest.raises(RuntimeError):
            m.generate("test prompt")
