"""Tests for model backends: MockModel, load_model, available_backends."""

from __future__ import annotations

import json

import pytest

import hh
from hh.models.base import (
    AssistantTurn,
    Model,
    Tool,
    ToolCall,
    available_backends,
    load_model,
)
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

    def test_includes_openai_compat(self):
        assert "openai-compat" in available_backends()

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
        prompt = (
            "CLAIM:\nParis is the capital of France.\n\nCONTEXT:\nParis is the capital of France.\n"
        )
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
# MockModel tool-calling (scripted playback)
# ---------------------------------------------------------------------------


class TestMockToolCalling:
    def test_supports_tools(self):
        assert MockModel().supports_tools is True

    def test_script_plays_back_in_order(self):
        turns = [
            AssistantTurn(tool_calls=[ToolCall(id="c1", name="t", arguments={"a": 1})]),
            AssistantTurn(text="final"),
        ]
        model = MockModel(script=turns)
        tools = [Tool(name="t", description="", parameters={})]
        t1 = model.generate_with_tools([{"role": "user", "content": "q"}], tools)
        t2 = model.generate_with_tools([{"role": "user", "content": "q"}], tools)
        assert t1 == turns[0]
        assert t2 == turns[1]

    def test_script_repeats_last_turn_when_exhausted(self):
        model = MockModel(script=[AssistantTurn(text="final")])
        tools: list[Tool] = []
        for _ in range(3):
            turn = model.generate_with_tools([{"role": "user", "content": "q"}], tools)
            assert turn.text == "final"
            assert turn.tool_calls == []

    def test_no_script_yields_heuristic_final_turn(self):
        model = MockModel()
        turn = model.generate_with_tools(
            [{"role": "user", "content": "CLAIM:\nx\n\nCONTEXT:\nx"}], []
        )
        assert turn.tool_calls == []
        assert "support" in turn.text

    def test_tool_turns_are_recorded(self):
        model = MockModel(script=[AssistantTurn(text="done")])
        model.generate_with_tools([{"role": "user", "content": "hello"}], [])
        assert model.calls == ["hello"]
        assert len(model.turns) == 1


# ---------------------------------------------------------------------------
# Tool protocol dataclasses
# ---------------------------------------------------------------------------


class TestToolProtocol:
    def test_to_api_shape(self):
        tool = Tool(name="t", description="d", parameters={"type": "object"})
        api = tool.to_api()
        assert api["type"] == "function"
        assert api["function"]["name"] == "t"
        assert api["function"]["parameters"] == {"type": "object"}

    def test_base_model_generate_with_tools_raises(self):
        class Bare(Model):
            def generate(self, prompt, *, system=None, temperature=0.0, max_tokens=1024):
                return ""

        with pytest.raises(NotImplementedError, match="does not support tool calling"):
            Bare().generate_with_tools([], [])


# ---------------------------------------------------------------------------
# openai-compat: config errors and key hygiene
# ---------------------------------------------------------------------------


class TestOpenAICompatConfig:
    def test_load_succeeds(self):
        m = load_model("openai-compat")
        assert m.name == "openai-compat"

    def test_missing_key_raises(self, monkeypatch):
        monkeypatch.delenv("HH_API_KEY", raising=False)
        m = load_model("openai-compat", model_id="test-model", base_url="http://unit.test/v1")
        with pytest.raises(RuntimeError, match="HH_API_KEY is not set"):
            m.generate("test prompt")

    def test_missing_model_raises(self, monkeypatch):
        monkeypatch.setenv("HH_API_KEY", "test-key-xyz")
        monkeypatch.delenv("HH_MODEL", raising=False)
        m = load_model("openai-compat", base_url="http://unit.test/v1")
        m._post = lambda payload: {"choices": [{"message": {"content": "x"}}]}  # type: ignore[method-assign]
        with pytest.raises(RuntimeError, match="open-source model with tool calling"):
            m.generate("hi")

    def test_env_config_is_used(self, monkeypatch):
        monkeypatch.setenv("HH_API_KEY", "test-key-xyz")
        monkeypatch.setenv("HH_MODEL", "test-model")
        monkeypatch.setenv("HH_BASE_URL", "http://unit.test/v1/")
        m = load_model("openai-compat")
        assert m.model_id == "test-model"
        assert m.base_url == "http://unit.test/v1"
        assert m._headers()["Authorization"] == "Bearer test-key-xyz"

    def test_default_base_url(self, monkeypatch):
        monkeypatch.delenv("HH_BASE_URL", raising=False)
        assert load_model("openai-compat").base_url == "https://openrouter.ai/api/v1"

    def test_error_message_never_contains_the_key(self, monkeypatch):
        """On a non-JSON response, nothing secret leaks into the message."""
        monkeypatch.setenv("HH_API_KEY", "test-key-xyz")
        m = load_model("openai-compat", model_id="test-model", base_url="http://unit.test/v1")

        class FakeResp:
            def read(self):
                return b"not json"

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

        import hh.models.openai_compat as compat_module

        def fake_urlopen(req, timeout):
            return FakeResp()

        monkeypatch.setattr(compat_module.urllib.request, "urlopen", fake_urlopen)
        with pytest.raises(RuntimeError) as exc_info:
            m.generate("hi")
        assert "test-key-xyz" not in str(exc_info.value)


class TestOpenAICompatWireFormat:
    """The pure request/response mapping, no network."""

    def _model(self):
        return load_model("openai-compat", model_id="test-model", base_url="http://unit.test/v1")

    def test_to_wire_reshapes_assistant_tool_calls(self):
        messages = [
            {"role": "user", "content": "q"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": "c1", "name": "t", "arguments": {"a": 1}}],
            },
            {"role": "tool", "tool_call_id": "c1", "content": "result"},
        ]
        wire = self._model()._to_wire(messages)
        assert wire[0] == {"role": "user", "content": "q"}
        tc = wire[1]["tool_calls"][0]
        assert tc["type"] == "function"
        assert tc["function"]["name"] == "t"
        assert json.loads(tc["function"]["arguments"]) == {"a": 1}
        assert wire[2] == {"role": "tool", "tool_call_id": "c1", "content": "result"}

    def test_turn_parses_tool_calls(self):
        resp = {
            "choices": [
                {
                    "message": {
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "search_docs",
                                    "arguments": '{"query": "fares"}',
                                },
                            }
                        ],
                    }
                }
            ]
        }
        turn = self._model()._turn(resp)
        assert turn.text == ""
        assert len(turn.tool_calls) == 1
        assert turn.tool_calls[0].name == "search_docs"
        assert turn.tool_calls[0].arguments == {"query": "fares"}

    def test_turn_with_bad_arguments_json(self):
        resp = {
            "choices": [
                {
                    "message": {
                        "tool_calls": [{"id": "c", "function": {"name": "t", "arguments": "oops"}}]
                    }
                }
            ]
        }
        turn = self._model()._turn(resp)
        assert turn.tool_calls[0].arguments == {"_raw": "oops"}

    def test_unexpected_response_shape_raises(self):
        with pytest.raises(RuntimeError, match="unexpected response shape"):
            self._model()._message({"error": {"message": "boom"}})

    def test_payload_includes_tools_and_choice(self):
        tools = [Tool(name="t", description="d", parameters={"type": "object"})]
        payload = self._model()._chat_payload(
            [{"role": "user", "content": "q"}],
            tools=[t.to_api() for t in tools],
            temperature=0.0,
            max_tokens=99,
        )
        assert payload["tool_choice"] == "auto"
        assert payload["tools"][0]["function"]["name"] == "t"
        assert payload["max_tokens"] == 99

    def test_generate_retries_when_reasoning_ate_the_budget(self):
        """Reasoning models can return null content; retry with more tokens."""
        monkey = pytest.MonkeyPatch()
        model = self._model()
        seen_budgets = []

        def fake_post(payload):
            seen_budgets.append(payload["max_tokens"])
            content = "answer" if payload["max_tokens"] > 1024 else None
            return {"choices": [{"message": {"content": content}}]}

        monkey.setattr(model, "_post", fake_post)
        try:
            assert model.generate("hi") == "answer"
        finally:
            monkey.undo()
        assert seen_budgets == [1024, 4096]  # succeeds on the first escalation

    def test_generate_returns_empty_when_all_budgets_exhausted(self):
        model = self._model()
        calls = []

        def fake_post(payload):
            calls.append(payload["max_tokens"])
            return {"choices": [{"message": {"content": None}}]}

        model._post = fake_post  # type: ignore[method-assign]
        assert model.generate("hi") == ""
        assert calls == [1024, 4096, 16384]

    def test_generate_with_tools_retries_when_turn_is_empty(self):
        model = self._model()
        calls = []

        def fake_post(payload):
            calls.append(payload["max_tokens"])
            if payload["max_tokens"] > 1024:
                tc = {
                    "id": "c1",
                    "function": {"name": "search_docs", "arguments": '{"query": "x"}'},
                }
                return {"choices": [{"message": {"content": None, "tool_calls": [tc]}}]}
            return {"choices": [{"message": {"content": None}}]}

        model._post = fake_post  # type: ignore[method-assign]
        turn = model.generate_with_tools([{"role": "user", "content": "q"}], [])
        assert calls == [1024, 4096]  # succeeds on the first escalation
        assert turn.tool_calls[0].name == "search_docs"


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
