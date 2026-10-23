"""Test-case runner: `hh cases`.

Runs a fixed set of cases through the existing investigate agent, unchanged,
and writes a plain-text run log that shows what the model was sent and what it
did. Nothing about the agent changes: a recording wrapper sits around the model
backend and notes every request and reply, and tool results are read back from
the conversations the agent sends on the next turn.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from .agent import SYSTEM_PROMPT, InvestigateAgent
from .models.base import AssistantTurn, Model, Tool
from .types import Example

MODEL_TEXT_CHARS = 500
TOOL_RESULT_CHARS = 1200


class CaseFileError(ValueError):
    """The cases file is unusable (a usage error)."""


# --------------------------------------------------------------------------
# Recording wrapper
# --------------------------------------------------------------------------
@dataclass
class _Call:
    id: str
    name: str
    arguments: dict[str, Any]
    result: str | None = None


@dataclass
class Recorder:
    """What happened during one case, in order."""

    events: list[tuple[str, Any]] = field(default_factory=list)
    system: str | None = None
    tools: list[dict[str, Any]] | None = None
    _pending: dict[str, _Call] = field(default_factory=dict)

    @property
    def tool_calls(self) -> list[_Call]:
        return [c for kind, c in self.events if kind == "call"]

    def request(self, messages: list[dict[str, Any]], tools: list[Tool], system: str | None):
        if tools and self.system is None:
            self.system = system or ""
            self.tools = [t.to_api() for t in tools]
        # The newest result for an id is the one belonging to the open call.
        latest: dict[str, str] = {}
        for m in messages:
            if m.get("role") == "tool":
                latest[str(m.get("tool_call_id"))] = str(m.get("content", ""))
        for cid, call in self._pending.items():
            if call.result is None and cid in latest:
                call.result = latest[cid]

    def reply_text(self, text: str) -> None:
        if text.strip():
            self.events.append(("model", text))

    def reply_calls(self, turn: AssistantTurn) -> None:
        for tc in turn.tool_calls:
            call = _Call(tc.id, tc.name, dict(tc.arguments))
            self._pending[tc.id] = call
            self.events.append(("call", call))


class RecordingModel(Model):
    """Wraps a backend and records every request and reply, changing nothing."""

    def __init__(self, inner: Model) -> None:
        self.inner = inner
        self.name = inner.name
        self.supports_tools = inner.supports_tools
        self.recorder = Recorder()

    def __getattr__(self, item: str) -> Any:  # model_id, base_url, ...
        return getattr(self.inner, item)

    def generate(self, prompt, *, system=None, temperature=0.0, max_tokens=1024) -> str:
        text = self.inner.generate(
            prompt, system=system, temperature=temperature, max_tokens=max_tokens
        )
        self.recorder.reply_text(text)
        return text

    def generate_with_tools(
        self, messages, tools, *, system=None, temperature=0.0, max_tokens=1024
    ) -> AssistantTurn:
        self.recorder.request(messages, tools, system)
        turn = self.inner.generate_with_tools(
            messages, tools, system=system, temperature=temperature, max_tokens=max_tokens
        )
        self.recorder.reply_text(turn.text)
        self.recorder.reply_calls(turn)
        return turn


# --------------------------------------------------------------------------
# Cases file
# --------------------------------------------------------------------------
def load_cases(path: str) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    try:
        fh = open(path, encoding="utf-8")  # noqa: SIM115
    except OSError as e:
        raise CaseFileError(f"cannot read cases file {path}: {e.strerror or e}") from e
    with fh:
        for i, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except ValueError as e:
                raise CaseFileError(f"{path}:{i}: not valid JSON ({e})") from e
            if not isinstance(d, dict):
                raise CaseFileError(f"{path}:{i}: expected a JSON object per line")
            for key in ("question", "answer"):
                if not isinstance(d.get(key), str):
                    raise CaseFileError(f"{path}:{i}: '{key}' is required and must be text")
            d.setdefault("id", len(cases) + 1)
            cases.append(d)
    if not cases:
        raise CaseFileError(f"no cases found in {path}")
    return cases


# --------------------------------------------------------------------------
# Checking expectations
# --------------------------------------------------------------------------
def check_expectations(
    expect: dict[str, Any], verdicts: list[tuple[str, str]], called: list[str]
) -> list[str]:
    """Return one short reason per broken expectation (empty list = PASS)."""
    reasons: list[str] = []
    for want in expect.get("claims") or []:
        needle = str(want.get("contains", ""))
        verdict = str(want.get("verdict", "")).lower()
        found = [s for text, s in verdicts if needle.lower() in text.lower()]
        if not found:
            reasons.append(f'expected a claim containing "{needle}", found none')
        elif verdict not in found:
            reasons.append(f'expected "{needle}" {verdict}, got {" / ".join(found)}')
    for tool in expect.get("must_call") or []:
        if tool not in called:
            reasons.append(f"expected a call to {tool}, none made")
    for tool in expect.get("must_not_call") or []:
        if tool in called:
            reasons.append(f"{tool} was called but must not be")
    limit = expect.get("max_claims")
    if limit is not None and len(verdicts) > int(limit):
        reasons.append(f"expected at most {limit} claims, got {len(verdicts)}")
    return reasons


# --------------------------------------------------------------------------
# Running
# --------------------------------------------------------------------------
def _one_line(text: str, cap: int) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= cap else text[:cap]


def _endpoint(model: Model) -> str:
    base = getattr(model, "base_url", None)
    if not isinstance(base, str) or not base:
        return "n/a"
    parts = urlsplit(base)
    return f"{parts.scheme}://{parts.hostname}" + (f":{parts.port}" if parts.port else "")


def _model_id(model: Model, model_id: str | None) -> str:
    return str(getattr(model, "model_id", "") or model_id or model.name)


def _stage_docs(docs: str | None, data: Any, tmp: str) -> str | None:
    """Docs folder for one case: all of it, or just the files the case lists."""
    if data is None:
        return docs
    if not docs:
        raise FileNotFoundError("case lists data files but no --docs folder was given")
    root = Path(docs)
    for name in data:
        src = root / str(name)
        if not src.is_file():
            raise FileNotFoundError(f"data file not found: {name}")
        dest = Path(tmp) / str(name)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dest)
    return tmp


@dataclass
class CaseResult:
    id: str
    passed: bool
    reasons: list[str]
    lines: list[str]
    system: str | None = None
    tools: list[dict[str, Any]] | None = None


def run_case(case: dict[str, Any], model: Model, docs: str | None) -> CaseResult:
    cid = str(case["id"])
    purpose = str(case.get("purpose", ""))
    lines = [
        f"case {cid}  ({purpose})" if purpose else f"case {cid}",
        f'  user: "{case["question"]}" / answer: "{case["answer"]}"',
    ]
    rec = RecordingModel(model)
    verdicts: list[tuple[str, str]] = []
    reasons: list[str] = []
    with tempfile.TemporaryDirectory() as tmp:
        try:
            staged = _stage_docs(docs, case.get("data"), tmp)
            agent = InvestigateAgent(model=rec, docs=staged)
            example = Example(id=cid, prompt=case["question"], context="", answer=case["answer"])
            det = agent.detect(example)
            verdicts = [(v.claim.text, v.support.value) for v in det.verdicts]
        except Exception as e:  # noqa: BLE001 - a crash is a FAIL, the run goes on
            reasons.append("error: " + _one_line(str(e) or type(e).__name__, 200))
    for kind, item in rec.recorder.events:
        if kind == "model":
            lines.append(f'  model: "{_one_line(item, MODEL_TEXT_CHARS)}"')
        else:
            result = item.result if item.result is not None else "(no result recorded)"
            args = json.dumps(item.arguments, ensure_ascii=False)
            lines.append(
                f"  tool call: {item.name}({args}) -> {_one_line(result, TOOL_RESULT_CHARS)}"
            )
    if verdicts:
        shown = "; ".join(f'[{s}] "{t}"' for t, s in verdicts)
        lines.append(f"  verdicts: {shown}")
    elif not reasons:
        lines.append("  verdicts: (no claims)")
    if not reasons:
        called = [c.name for c in rec.recorder.tool_calls]
        reasons = check_expectations(case.get("expect") or {}, verdicts, called)
    passed = not reasons
    lines.append("  result: PASS" if passed else f"  result: FAIL  ({'; '.join(reasons)})")
    return CaseResult(cid, passed, reasons, lines, rec.recorder.system, rec.recorder.tools)


def _secrets(model: Model) -> list[str]:
    return [os.environ.get("HH_API_KEY", ""), str(getattr(model, "_api_key", ""))]


def _scrub(text: str, secrets: list[str]) -> str:
    for s in secrets:
        if s:
            text = text.replace(s, "[hidden]")
    return text


def _prompt_block(system: str, tools: list[dict[str, Any]], again: bool) -> list[str]:
    tag = " (changed)" if again else ""
    return [
        f"system prompt{tag}:",
        system,
        f"tools{tag}:",
        json.dumps(tools, indent=2, ensure_ascii=False),
    ]


def run_cases(
    cases: list[dict[str, Any]],
    model: Model,
    docs: str | None,
    model_id: str | None = None,
) -> tuple[str, list[CaseResult]]:
    """Run every case; return the full log text and the per-case results."""
    results = [run_case(c, model, docs) for c in cases]
    first = next((r for r in results if r.system is not None), None)
    if first is not None:
        shown_system: str = first.system or ""
        shown_tools: list[dict[str, Any]] = first.tools or []
    else:
        # Nothing reached the agent: show what it would have sent.
        shown_system = SYSTEM_PROMPT
        shown_tools = [t.to_api() for t in InvestigateAgent(model=model).tools]
    out = [
        f"run: {datetime.now(timezone.utc).replace(microsecond=0).isoformat()}",
        f"model: {_model_id(model, model_id)}",
        f"endpoint: {_endpoint(model)}",
        *_prompt_block(shown_system, shown_tools, again=False),
    ]
    for r in results:
        out.append("")
        if r.system is not None and (r.system != shown_system or r.tools != shown_tools):
            out += _prompt_block(r.system, r.tools or [], again=True) + [""]
            shown_system, shown_tools = r.system, r.tools or []
        out += r.lines
    passed = sum(1 for r in results if r.passed)
    out += ["", f"cases passed: {passed} of {len(results)}"]
    return _scrub("\n".join(out) + "\n", _secrets(model)), results


def summary_lines(results: list[CaseResult], model: Model) -> list[str]:
    lines = []
    for r in results:
        if r.passed:
            lines.append(f"case {r.id} PASS")
        else:
            lines.append(f"case {r.id} FAIL {'; '.join(r.reasons)}")
    passed = sum(1 for r in results if r.passed)
    lines.append(f"cases passed: {passed} of {len(results)}")
    return [_scrub(line, _secrets(model)) for line in lines]
