"""Reporting: turn an EvalRun into JSON or a readable Markdown report."""

from __future__ import annotations

import json

from .eval import EvalRun
from .types import Support


def to_json(run: EvalRun, indent: int = 2) -> str:
    return json.dumps(run.result.to_dict(), indent=indent)


_SYMBOL = {
    Support.SUPPORTED: "✓",
    Support.CONTRADICTED: "✗",
    Support.UNVERIFIABLE: "?",
}


def to_markdown(
    run: EvalRun, title: str = "Hallucination-Hunter Report", show_examples: int = 5
) -> str:
    r = run.result
    lines = [f"# {title}", ""]
    lines += [
        f"- **Detector:** `{r.detector}`",
        f"- **Examples:** {r.n_examples}",
        f"- **Claims checked:** {r.n_claims}",
        f"- **Hallucination rate:** {r.hallucination_rate:.1%} "
        f"(answers flagged with >=1 unsupported claim)",
        f"- **Mean groundedness:** {r.mean_groundedness:.1%} (share of claims supported)",
        "",
    ]

    if r.confusion is not None:
        c = r.confusion
        lines += [
            "## Detection quality (vs gold labels)",
            "",
            f"Labeled examples: **{r.n_labeled}**. Positive class = *hallucinated*.",
            "",
            "| Metric | Value |",
            "|---|---|",
            f"| Precision | {c.precision:.3f} |",
            f"| Recall | {c.recall:.3f} |",
            f"| F1 | {c.f1:.3f} |",
            f"| Accuracy | {c.accuracy:.3f} |",
            "",
            "| | gold: hallucinated | gold: clean |",
            "|---|---|---|",
            f"| **pred: hallucinated** | {c.tp} (TP) | {c.fp} (FP) |",
            f"| **pred: clean** | {c.fn} (FN) | {c.tn} (TN) |",
            "",
        ]

    if show_examples and run.detections:
        lines += ["## Sample verdicts", ""]
        for det in run.detections[:show_examples]:
            flag = "🚩 hallucinated" if det.predicted_hallucinated else "✅ grounded"
            lines.append(f"### `{det.example.id}` — {flag}")
            if det.example.prompt:
                lines.append(f"> {det.example.prompt}")
            lines.append("")
            for v in det.verdicts:
                sym = _SYMBOL.get(v.support, "?")
                lines.append(f"- {sym} **{v.support.value}** — {v.claim.text}")
                if v.rationale:
                    lines.append(f"  - _{v.rationale}_")
            lines.append("")

    return "\n".join(lines)
