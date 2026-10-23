"""Agent-based investigation: `hh investigate`.

A tool-using detector. For each claim in the answer, a model agent decides how
to gather evidence — searching a folder of source documents, running a
groundedness check on a snippet it found, or sampling the model for
self-consistency — and then commits to a verdict naming the evidence it used.

The agent is just another Detector: it consumes an Example and produces a
Detection whose per-claim Verdicts carry `rationale`, `evidence`, and a compact
`trace` of the tool calls taken. Everything downstream (metrics, reports,
`hh eval`) works unchanged.

The loop lives here, not in the backends: each turn asks the Model (via
`generate_with_tools`) for text and/or tool calls; the agent executes the
calls, appends the results, and repeats until the model answers without tool
calls or the step budget runs out. Running out of budget, an unparsable final
answer, or an unknown tool all degrade to UNVERIFIABLE rather than crashing —
the same conservative fallback the LLM verifier uses.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .claims import LLMClaimExtractor, SentenceClaimExtractor
from .detectors.base import Detector, register
from .detectors.groundedness import LLMVerifier, _content_tokens, _parse_verdict
from .models.base import AssistantTurn, Model, Tool, ToolCall
from .types import Claim, Detection, Example, Support, Verdict

MAX_TOOL_RESULT_CHARS = 4000
MAX_FILE_CHARS = 200_000
SNIPPET_CHARS = 600

_SEARCH_TOOL = Tool(
    name="search_docs",
    description="Search the source documents. Returns the most relevant "
    "paragraphs with their document names.",
    parameters={
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "what to look for"},
            "limit": {"type": "integer", "description": "max paragraphs (default 3)"},
        },
        "required": ["query"],
    },
)

_READ_TOOL = Tool(
    name="read_doc",
    description="Read one source document in full, by name (as listed to you).",
    parameters={
        "type": "object",
        "properties": {"name": {"type": "string", "description": "document name"}},
        "required": ["name"],
    },
)

_GROUNDEDNESS_TOOL = Tool(
    name="groundedness_check",
    description="Run a strict fact-check of a claim against a context snippet "
    "you supply (e.g. a paragraph you found with search_docs). Returns a "
    "supported / contradicted / unverifiable call.",
    parameters={
        "type": "object",
        "properties": {
            "claim": {"type": "string"},
            "context": {"type": "string", "description": "the snippet to check against"},
        },
        "required": ["claim", "context"],
    },
)

_SELF_CONSISTENCY_TOOL = Tool(
    name="self_consistency_check",
    description="Sample the model several times for the prompt and report how "
    "often the samples agree with the claim. Low agreement suggests fabrication.",
    parameters={
        "type": "object",
        "properties": {
            "claim": {"type": "string"},
            "prompt": {"type": "string", "description": "the original question"},
            "samples": {"type": "integer", "description": "how many samples (default set by CLI)"},
        },
        "required": ["claim", "prompt"],
    },
)

SYSTEM_PROMPT = (
    "You are a meticulous fact-checker. You investigate ONE claim at a time "
    "against a folder of source documents.\n"
    "Work tool by tool: search the documents for evidence, read what you find, "
    "and verify the claim against it (groundedness_check) or sample the model "
    "for consistency (self_consistency_check).\n"
    "When you have enough evidence, STOP calling tools and reply with ONLY a "
    "JSON object, no other text:\n"
    '{"support": "supported|contradicted|unverifiable", '
    '"rationale": "<one sentence>", "evidence": "<the decisive quote(s), '
    'copied VERBATIM from the source documents>"}\n'
    "Quotes are re-checked against the files, so copy them EXACTLY from the "
    "tool results — search_docs snippets and read_doc text are the "
    "documents' own words. A 'supported' or 'contradicted' verdict MUST cite "
    "at least one exact quote from a source document; without a verifiable "
    "quote, answer 'unverifiable'. Use 'supported' only if a source document "
    "clearly backs the claim; 'contradicted' if a source states the opposite; "
    "otherwise 'unverifiable'."
)


# --------------------------------------------------------------------------
# Source documents
# --------------------------------------------------------------------------
def _fold(t: str) -> str:
    """Strip one common English inflection so search ignores plurals/tense.

    Applied to query and corpus tokens alike, so folding stays consistent on
    both sides: 'cars'->'car', 'carried'->'carry', 'sailings'->'sail'. Guarded
    so short words are left alone ('bus', 'tied', 'thing').
    """
    if len(t) > 4 and (t.endswith("ies") or t.endswith("ied")):
        return t[:-3] + "y"
    if len(t) >= 5 and t.endswith("es"):
        return t[:-2]
    if len(t) >= 5 and t.endswith("ed"):
        return t[:-2]
    if len(t) >= 6 and t.endswith("ing"):
        return t[:-3]
    if len(t) > 3 and t.endswith("s") and not t.endswith("ss"):
        return t[:-1]
    return t


def _search_tokens(text: str) -> set[str]:
    """Content tokens, folded twice so plural + tense both collapse."""
    return {_fold(_fold(t)) for t in _content_tokens(text)}


class DocStore:
    """A small, dependency-free folder of plain-text source documents."""

    def __init__(self, root: str | Path, *, max_file_chars: int = MAX_FILE_CHARS) -> None:
        self.root = Path(root)
        self.docs: dict[str, str] = {}
        for p in sorted(self.root.rglob("*")):
            if not p.is_file() or p.name.startswith("."):
                continue
            try:
                text = p.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            self.docs[str(p.relative_to(self.root))] = text[:max_file_chars]

    def names(self) -> list[str]:
        return sorted(self.docs)

    def read(self, name: str) -> str:
        if name not in self.docs:
            raise KeyError(f"no document named {name!r}")
        return self.docs[name][:MAX_TOOL_RESULT_CHARS]

    def search(self, query: str, limit: int = 3) -> list[dict[str, object]]:
        qt = _search_tokens(query)
        if not qt:
            return []
        hits: list[tuple[float, str, str]] = []
        for name, text in self.docs.items():
            for para in re.split(r"\n\s*\n", text):
                para = para.strip()
                if not para:
                    continue
                pt = _search_tokens(para)
                score = len(qt & pt) / len(qt)
                if score > 0:
                    hits.append((score, name, para[:SNIPPET_CHARS]))
        hits.sort(key=lambda h: (-h[0], h[1]))
        return [
            {"doc": name, "score": round(score, 3), "snippet": snippet}
            for score, name, snippet in hits[: max(1, limit)]
        ]


# --------------------------------------------------------------------------
# The agent
# --------------------------------------------------------------------------
def _assistant_message(turn: AssistantTurn) -> dict[str, object]:
    return {
        "role": "assistant",
        "content": turn.text,
        "tool_calls": [
            {"id": tc.id, "name": tc.name, "arguments": dict(tc.arguments)}
            for tc in turn.tool_calls
        ],
    }


def _short(text: str, cap: int = 80) -> str:
    text = " ".join(text.split())
    return text if len(text) <= cap else text[: cap - 1] + "…"


@register("investigate")
class InvestigateAgent(Detector):
    """Per-claim agentic investigation with document tools."""

    def __init__(
        self,
        model: Model | None = None,
        docs: str | Path | None = None,
        max_steps: int = 6,
        samples: int = 3,
        temperature: float = 1.0,
        atomic: bool = True,
        **_: object,
    ) -> None:
        if model is None:
            raise ValueError("investigate requires a model backend to drive the agent")
        if not model.supports_tools:
            raise ValueError(
                f"the {model.name!r} backend does not support tool calling; "
                "use a backend with generate_with_tools (e.g. 'openai-compat' or 'mock')"
            )
        self.model = model
        self.max_steps = max(1, max_steps)
        self.samples = max(1, samples)
        self.temperature = temperature
        # Atomic (default): one verdict per FACT — the model decomposes the
        # answer, so a compound sentence ("bought by X in 2019 and runs a route
        # to Y") yields separate verdicts instead of a single muddled one.
        # Falls back to sentence splitting when the model returns nothing
        # usable, and to the pure sentence splitter when atomic=False.
        self.atomic = atomic
        self.extractor = LLMClaimExtractor(model) if atomic else SentenceClaimExtractor()
        self.verifier = LLMVerifier(model)
        self.store = DocStore(docs) if docs else None
        self.tools = [_SEARCH_TOOL, _READ_TOOL, _GROUNDEDNESS_TOOL, _SELF_CONSISTENCY_TOOL]

    # -- Detector interface --------------------------------------------------
    def detect(self, example: Example) -> Detection:
        verdicts = [self._investigate(c, example) for c in self.extractor.extract(example.answer)]
        return Detection(
            example=example, verdicts=verdicts, detector=f"investigate:{self.model.name}"
        )

    # -- the loop -------------------------------------------------------------
    def _investigate(self, claim: Claim, example: Example) -> Verdict:
        messages: list[dict[str, object]] = [
            {"role": "user", "content": self._opening_message(claim, example)}
        ]
        trace: list[str] = []
        touched: list[str] = []
        # Quotes the investigation itself verified: groundedness_check calls
        # whose context really appears in a source. If the model's final quote
        # is sloppy, these can still carry the verdict.
        tool_quotes: list[tuple[str, str]] = []
        for _ in range(self.max_steps):
            turn = self.model.generate_with_tools(
                messages, self.tools, system=SYSTEM_PROMPT, temperature=0.0
            )
            if not turn.tool_calls:
                return _final_verdict(
                    claim, turn.text, touched, trace, tool_quotes, self.store, example.context
                )
            messages.append(_assistant_message(turn))
            for tc in turn.tool_calls:
                result = self._execute(tc, example, touched)
                trace.append(f"{tc.name}({_short(json.dumps(tc.arguments))}) -> {_short(result)}")
                if tc.name == "groundedness_check":
                    self._collect_tool_quote(result, tool_quotes, self.store, example.context)
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})
        # Budget spent. The evidence gathered so far is on the table; force one
        # tools-free turn so the model must commit to a verdict from it rather
        # than discarding the whole investigation.
        final = self.model.generate_with_tools(messages, [], system=SYSTEM_PROMPT, temperature=0.0)
        if not final.tool_calls and final.text.strip():
            return _final_verdict(
                claim, final.text, touched, trace, tool_quotes, self.store, example.context
            )
        return Verdict(
            claim,
            Support.UNVERIFIABLE,
            0.0,
            f"investigation budget exhausted after {self.max_steps} tool rounds",
            _evidence_from(touched),
            tuple(trace),
        )

    @staticmethod
    def _collect_tool_quote(
        result: str,
        tool_quotes: list[tuple[str, str]],
        store: DocStore | None,
        context: str,
    ) -> None:
        """Record a groundedness_check context that really is in a source."""
        data = _json_object(result)
        if not isinstance(data, dict):
            return
        support = str(data.get("support", "")).lower()
        if support not in ("supported", "contradicted"):
            return
        quote = _verify_evidence(str(data.get("evidence", "")), store, context)
        if quote:
            tool_quotes.append((support, quote))

    def _opening_message(self, claim: Claim, example: Example) -> str:
        doc_names = ", ".join(self.store.names()) if self.store else "(none)"
        parts = [
            "Investigate the CLAIM below and return a verdict.",
            "You may call tools to gather evidence first.",
            "",
            "CLAIM:",
            claim.text,
            "",
            "CONTEXT:",
            example.context.strip() or "(none provided — rely on the source documents)",
        ]
        if example.prompt:
            parts += ["", "QUESTION:", example.prompt]
        parts += ["", f"Source documents: {doc_names}"]
        return "\n".join(parts)

    # -- tool execution --------------------------------------------------------
    def _execute(self, call: ToolCall, example: Example, touched: list[str]) -> str:
        try:
            result = self._dispatch(call, example, touched)
        except Exception as e:  # noqa: BLE001 - tool errors go back to the model
            result = {"error": str(e)}
        return json.dumps(result)[:MAX_TOOL_RESULT_CHARS]

    def _dispatch(self, call: ToolCall, example: Example, touched: list[str]) -> dict[str, object]:
        args = call.arguments or {}
        if call.name == "search_docs":
            if self.store is None:
                raise ValueError("no source documents were provided (--docs)")
            query = str(args.get("query", "")).strip()
            if not query:
                raise ValueError("search_docs needs a query")
            hits = self.store.search(query, limit=int(args.get("limit", 3) or 3))
            touched.extend(h["doc"] for h in hits if isinstance(h["doc"], str))
            return {"query": query, "results": hits}
        if call.name == "read_doc":
            if self.store is None:
                raise ValueError("no source documents were provided (--docs)")
            name = str(args.get("name", "")).strip()
            text = self.store.read(name)
            touched.append(name)
            return {"doc": name, "text": text}
        if call.name == "groundedness_check":
            claim_text = str(args.get("claim", "")).strip()
            context = str(args.get("context", "")).strip()
            if not claim_text or not context:
                raise ValueError("groundedness_check needs a claim and a context")
            v = self.verifier.verify(Claim(text=claim_text), context)
            return {
                "support": v.support.value,
                "rationale": v.rationale,
                "evidence": context[:SNIPPET_CHARS],
            }
        if call.name == "self_consistency_check":
            return self._consistency(args, example)
        raise ValueError(f"unknown tool {call.name!r}")

    def _consistency(self, args: dict[str, Any], example: Example) -> dict[str, object]:
        claim_text = str(args.get("claim", "")).strip()
        prompt = str(args.get("prompt", "")).strip() or example.prompt
        if not claim_text:
            raise ValueError("self_consistency_check needs a claim")
        if not prompt:
            raise ValueError("self_consistency_check needs a prompt (or a question on the example)")
        n = int(args.get("samples", self.samples) or self.samples)
        ct = _content_tokens(claim_text)
        agree = 0
        for _ in range(max(1, n)):
            sample = self.model.generate(prompt, temperature=self.temperature)
            if ct and len(ct & _content_tokens(sample)) / len(ct) >= 0.6:
                agree += 1
        return {
            "samples": n,
            "agreeing": agree,
            "agreement": round(agree / max(1, n), 3),
            "note": "agreement = share of samples echoing the claim",
        }


# --------------------------------------------------------------------------
# Final-verdict parsing
# --------------------------------------------------------------------------
def _evidence_from(touched: list[str]) -> str:
    uniq = sorted(set(touched))
    return f"documents consulted: {', '.join(uniq)}" if uniq else ""


def _squash(text: str) -> str:
    return " ".join(text.split())


_SENT_END_RE = re.compile(r"[.!?](?=\s|$)")
# Tokens whose trailing period is not a sentence end ('Harrow Point Ferry Co.').
_ABBREV_TOKENS = {
    "mr",
    "mrs",
    "ms",
    "dr",
    "st",
    "vs",
    "etc",
    "no",
    "fig",
    "e.g",
    "i.e",
    "co",
    "ltd",
    "inc",
    "approx",
    "u.s",
}


def _token_before(text: str, pos: int) -> str:
    """The token ending just before `pos` (for the abbreviation guard)."""
    head = text[:pos].rstrip()
    if not head:
        return ""
    word = head.split(" ")[-1]
    return "".join(ch for ch in word.lower() if ch.isalnum() or ch == ".").rstrip(".")


def _sentence_span(text: str, start: int, end: int) -> tuple[int, int]:
    """Expand a matched span to the full sentence that contains it.

    A fragment like 'costs $3 each way' inside 'costs $3 each way, except
    Saturdays, when bikes cost $10.' must not verify as a bare fragment: the
    evidence shows the whole sentence, qualifier included, so nothing after
    (or before) the quoted words can hide.
    """
    s = 0
    for m in _SENT_END_RE.finditer(text, 0, start):
        if _token_before(text, m.start()) not in _ABBREV_TOKENS:
            s = m.end()
    while s < start and text[s] == " ":
        s += 1
    # The span already ends at a sentence boundary — leave the end alone.
    if end and text[end - 1] in ".!?" and (end == len(text) or text[end] == " "):
        return s, end
    e = len(text)
    for m in _SENT_END_RE.finditer(text, end):
        if _token_before(text, m.start()) not in _ABBREV_TOKENS:
            e = m.end()
            break
    return s, e


def _source_units(store: DocStore | None, context: str) -> list[tuple[str, str]]:
    """Searchable text units: squashed paragraphs, then the whole text.

    Paragraph units keep headings and unrelated sections out of a match's
    sentence expansion; the whole-text units catch quotes that span
    paragraphs.
    """
    units: list[tuple[str, str]] = []
    if store is not None:
        for name, text in store.docs.items():
            for para in re.split(r"\n\s*\n", text):
                p = _squash(para)
                if p:
                    units.append((name, p))
            units.append((name, _squash(text)))
    if context:
        for para in re.split(r"\n\s*\n", context):
            p = _squash(para)
            if p:
                units.append(("context", p))
        units.append(("context", _squash(context)))
    return units


def _find_in_sources(quote: str, units: list[tuple[str, str]]) -> str | None:
    """Verbatim word search: case- and terminal-punctuation-insensitive.

    The words must appear in the source exactly and in order — a substantive
    paraphrase still fails — but casing drift or a stray trailing '!' does
    not. On a match, the evidence shown is the FULL SENTENCE containing the
    quote in the source's own casing, so a qualifier next to the quoted words
    cannot hide.
    """
    lq = quote.lower().strip(" .,!?;:")
    if not lq:
        return None
    for name, text in units:
        idx = text.lower().find(lq)
        if idx != -1:
            start, end = _sentence_span(text, idx, idx + len(lq))
            return f'"{text[start:end]}" ({name})'
    return None


def _verify_evidence(raw: str, store: DocStore | None, context: str) -> str:
    """Keep only evidence quotes that appear verbatim in the sources.

    The model may glue a document's heading or its own attribution into the
    evidence line; the code re-checks every quoted snippet against the actual
    documents and re-attributes each quote to the document it really appears
    in, shown as the full sentence it sits in. Quotes that appear nowhere are
    dropped.
    """
    units = _source_units(store, context)
    if not units:
        return ""
    flat = (
        (raw or "")
        .replace("\u201c", '"')
        .replace("\u201d", '"')
        .replace("\u2018", "'")
        .replace("\u2019", "'")
    )
    quotes = re.findall(r'"([^"]{3,})"', flat)
    if not quotes and flat.strip():
        quotes = [flat.strip()]
    kept: list[str] = []
    for q in quotes:
        found = _find_in_sources(_squash(q), units)
        if found and found not in kept:
            kept.append(found)
    return "; ".join(kept)


def _final_verdict(
    claim: Claim,
    raw: str,
    touched: list[str],
    trace: list[str],
    tool_quotes: list[tuple[str, str]],
    store: DocStore | None,
    context: str,
) -> Verdict:
    fallback_evidence = _evidence_from(touched)
    obj = _json_object(raw)
    if isinstance(obj, dict):
        support_str = str(obj.get("support", "")).lower().strip()
        if support_str in {s.value for s in Support}:
            support = Support(support_str)
            rationale = str(obj.get("rationale", "")).strip()
            evidence = _verify_evidence(str(obj.get("evidence", "")), store, context)
            return _earned_verdict(
                claim,
                support,
                rationale or "agent verdict",
                evidence,
                tool_quotes,
                fallback_evidence,
                trace,
                True,
            )
    # Not usable JSON: fall back to the plain-verdict scanner. There is no
    # model-supplied evidence to verify on this path, so a support call can
    # only be earned by a quote the investigation itself verified.
    support, rationale = _parse_verdict(raw)
    return _earned_verdict(
        claim, support, rationale, "", tool_quotes, fallback_evidence, trace, False
    )


def _earned_verdict(
    claim: Claim,
    support: Support,
    rationale: str,
    evidence: str,
    tool_quotes: list[tuple[str, str]],
    fallback_evidence: str,
    trace: list[str],
    direct_call: bool,
) -> Verdict:
    """Apply the evidence rule to a support or contradiction call.

    A support or contradiction call is earned with verifiable backing: at
    least one quote that really appears in a source. If the given evidence did
    not verify (or there is none), a groundedness_check the investigation
    itself verified can carry the verdict; without one, the honest call is
    unverifiable. Unverifiable needs no quote by definition.

    Confidence reflects how the verdict was earned: 1.0 for a direct call
    backed by a verified quote, 0.9 when a tool-verified quote carried a
    support call, 0.8 when parsed from a non-JSON answer, 0.5 when downgraded.
    """
    score = 1.0 if direct_call else 0.8
    if support in (Support.SUPPORTED, Support.CONTRADICTED):
        if not evidence:
            for support_value, quote in tool_quotes:
                if support_value == support.value:
                    evidence, score = quote, 0.9
                    break
        if not evidence:
            note = (
                f"downgraded from {support.value}: no evidence quote verified against the sources"
            )
            return Verdict(
                claim,
                Support.UNVERIFIABLE,
                0.5,
                f"{rationale} [{note}]" if rationale else note,
                fallback_evidence,
                tuple(trace),
            )
    return Verdict(claim, support, score, rationale, evidence or fallback_evidence, tuple(trace))
    return Verdict(claim, support, 1.0, rationale, evidence or fallback_evidence, tuple(trace))


def _json_object(raw: str) -> object | None:
    raw = (raw or "").strip()
    start, end = raw.find("{"), raw.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        return json.loads(raw[start : end + 1])
    except (ValueError, TypeError):
        return None
