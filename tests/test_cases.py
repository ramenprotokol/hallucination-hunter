"""Tests for `hh cases`: the test-case runner and its run log."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import hh.cli as cli
from hh.models.base import AssistantTurn, Model, ToolCall

SECRET = "sk-test-SECRET-1234567890"
DOCS = Path(__file__).parent / "fixtures" / "investigate" / "sources"


def _final(support: str, evidence: str = "", rationale: str = "done") -> AssistantTurn:
    return AssistantTurn(
        text=json.dumps({"support": support, "rationale": rationale, "evidence": evidence})
    )


def _call(id_: str, tool: str, **kwargs) -> AssistantTurn:
    return AssistantTurn(tool_calls=[ToolCall(id=id_, name=tool, arguments=kwargs)])


class Scripted(Model):
    """Offline backend: claim lists keyed by answer, agent turns from a queue."""

    name = "scripted"
    supports_tools = True

    def __init__(self, claims: dict[str, list[str]], turns: list):
        self.claims = claims
        self.turns = list(turns)
        self.model_id = "scripted-1"
        self.base_url = "https://models.example.com/v1/secret-path"
        self._api_key = SECRET

    def generate(self, prompt, *, system=None, temperature=0.0, max_tokens=1024):
        for answer, claims in self.claims.items():
            if answer in prompt:
                return json.dumps(claims)
        return "[]"

    def generate_with_tools(
        self, messages, tools, *, system=None, temperature=0.0, max_tokens=1024
    ):
        item = self.turns.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def _write_cases(tmp_path, cases) -> str:
    p = tmp_path / "cases.jsonl"
    p.write_text("\n".join(json.dumps(c) for c in cases) + "\n", encoding="utf-8")
    return str(p)


def _run(tmp_path, monkeypatch, capsys, model, cases):
    monkeypatch.setattr(cli, "load_model", lambda *a, **k: model)
    log = tmp_path / "run.log"
    rc = cli.main(
        ["cases", "--cases", _write_cases(tmp_path, cases), "--docs", str(DOCS), "--log", str(log)]
    )
    out = capsys.readouterr()
    return rc, out.out, out.err, log.read_text(encoding="utf-8")


def _case(id_=1, **expect):
    return {
        "id": id_,
        "purpose": "Bug being fixed",
        "question": "What does a return cost?",
        "answer": f"Answer number {id_}.",
        "expect": expect,
    }


def test_passing_case(tmp_path, monkeypatch, capsys):
    model = Scripted(
        {"Answer number 1.": ["An adult return is $14"]},
        [_call("c1", "search_docs", query="return fare"), _final("unverifiable")],
    )
    case = _case(claims=[{"contains": "$14", "verdict": "unverifiable"}], must_call=["search_docs"])
    rc, out, _, log = _run(tmp_path, monkeypatch, capsys, model, [case])
    assert rc == 0
    assert "case 1 PASS" in out
    assert "cases passed: 1 of 1" in out
    assert "result: PASS" in log


def test_failing_case_reason(tmp_path, monkeypatch, capsys):
    model = Scripted({"Answer number 1.": ["An adult return is $14"]}, [_final("supported")])
    case = _case(claims=[{"contains": "$14", "verdict": "contradicted"}])
    rc, out, _, log = _run(tmp_path, monkeypatch, capsys, model, [case])
    assert rc == 0
    assert 'case 1 FAIL expected "$14" contradicted, got unverifiable' in out
    assert 'result: FAIL  (expected "$14" contradicted, got unverifiable)' in log
    assert "cases passed: 0 of 1" in log


def test_missing_claim_text(tmp_path, monkeypatch, capsys):
    model = Scripted({"Answer number 1.": ["Something else"]}, [_final("unverifiable")])
    case = _case(claims=[{"contains": "$14", "verdict": "supported"}])
    _, out, _, _ = _run(tmp_path, monkeypatch, capsys, model, [case])
    assert 'expected a claim containing "$14", found none' in out


def test_must_call_and_must_not_call(tmp_path, monkeypatch, capsys):
    model = Scripted(
        {"Answer number 1.": ["A claim"]},
        [_call("c1", "read_doc", name="02_fares.md"), _final("unverifiable")],
    )
    case = _case(must_call=["search_docs"], must_not_call=["read_doc"])
    _, out, _, log = _run(tmp_path, monkeypatch, capsys, model, [case])
    assert "expected a call to search_docs, none made" in out
    assert "read_doc was called but must not be" in out
    assert "result: FAIL" in log


def test_max_claims(tmp_path, monkeypatch, capsys):
    # An answer with no facts gives zero claims, so max_claims 0 passes.
    model = Scripted({}, [_final("unverifiable")])
    empty = {"id": 1, "question": "q", "answer": "", "expect": {"max_claims": 0}}
    _, out, _, _ = _run(tmp_path, monkeypatch, capsys, model, [empty])
    assert "case 1 PASS" in out
    # One claim comes back, so max_claims 0 fails with a short reason.
    two = {"id": 2, "question": "q", "answer": "asdf qwer.", "expect": {"max_claims": 0}}
    _, out, _, _ = _run(tmp_path, monkeypatch, capsys, model, [two])
    assert "case 2 FAIL expected at most 0 claims, got 1" in out


def test_api_error_mid_run_continues(tmp_path, monkeypatch, capsys):
    model = Scripted(
        {"Answer number 1.": ["One"], "Answer number 2.": ["Two"]},
        [RuntimeError("the model API returned HTTP 500"), _final("unverifiable")],
    )
    cases = [_case(1), _case(2, claims=[{"contains": "Two", "verdict": "unverifiable"}])]
    rc, out, _, log = _run(tmp_path, monkeypatch, capsys, model, cases)
    assert rc == 0
    assert "case 1 FAIL error: the model API returned HTTP 500" in out
    assert "case 2 PASS" in out
    assert "cases passed: 1 of 2" in out
    assert "cases passed: 1 of 2" in log


def test_log_contents(tmp_path, monkeypatch, capsys):
    from hh.agent import SYSTEM_PROMPT

    model = Scripted(
        {"Answer number 1.": ["Fares are listed"]},
        [
            AssistantTurn(
                text="Let me look.",
                tool_calls=[ToolCall("c1", "search_docs", {"query": "adult return fare"})],
            ),
            _final("unverifiable"),
        ],
    )
    _, _, _, log = _run(tmp_path, monkeypatch, capsys, model, [_case()])
    assert log.startswith("run: ")
    assert "model: scripted-1" in log
    assert "endpoint: https://models.example.com\n" in log
    assert "secret-path" not in log
    assert SYSTEM_PROMPT in log
    assert '"name": "search_docs"' in log
    assert 'model: "Let me look."' in log
    assert 'tool call: search_docs({"query": "adult return fare"}) -> {"query"' in log
    assert "02_fares.md" in log
    assert "verdicts: [unverifiable]" in log
    assert "cases passed: 0 of 1" in log or "cases passed: 1 of 1" in log


def test_api_key_never_shown(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("HH_API_KEY", SECRET)
    model = Scripted(
        {"Answer number 1.": ["One"]},
        [RuntimeError(f"upstream said bad key {SECRET}")],
    )
    _, out, err, log = _run(tmp_path, monkeypatch, capsys, model, [_case()])
    assert SECRET not in log
    assert SECRET not in out
    assert SECRET not in err
    assert "case 1 FAIL" in out


def test_data_limits_docs(tmp_path, monkeypatch, capsys):
    model = Scripted({"Answer number 1.": ["One"]}, [_final("unverifiable")])
    case = _case()
    case["data"] = ["missing.csv"]
    _, out, _, _ = _run(tmp_path, monkeypatch, capsys, model, [case])
    assert "data file not found: missing.csv" in out


def test_usage_errors_exit_2(tmp_path, capsys):
    log = tmp_path / "run.log"
    rc = cli.main(
        ["cases", "--cases", str(tmp_path / "nope.jsonl"), "--docs", str(DOCS), "--log", str(log)]
    )
    assert rc == 2
    assert not log.exists()
    with pytest.raises(SystemExit) as e:
        cli.main(["cases", "--docs", str(DOCS)])
    assert e.value.code == 2
    bad = tmp_path / "bad.jsonl"
    bad.write_text("not json\n", encoding="utf-8")
    rc = cli.main(["cases", "--cases", str(bad), "--docs", str(DOCS), "--log", str(log)])
    assert rc == 2
