# hallucination-hunter

[![CI](https://github.com/ramenprotokol/hallucination-hunter/actions/workflows/ci.yml/badge.svg)](https://github.com/ramenprotokol/hallucination-hunter/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue)](pyproject.toml)
![Runs offline](https://img.shields.io/badge/runs-offline%2C%20no%20API%20key-brightgreen)

**A small, runnable harness for detecting and evaluating hallucinations in LLM output.**

LLMs state things that aren't true with total confidence. `hallucination-hunter`
checks whether the claims in an answer are actually *grounded* — supported by a
source you trust — and measures how well a detector catches the ones that aren't.

It is a real tool, not a slide deck:

- A `hh` command-line tool and a clean Python API.
- A pluggable detector architecture (groundedness ships in v0.1; self-consistency
  and reference-judge are wired in behind the same interface).
- Pluggable model backends — **Anthropic**, **OpenAI**, and a deterministic
  **mock** so the whole thing runs, and the full test suite passes, **with zero
  API keys**.
- Labeled datasets and detection-quality metrics (precision / recall / F1), not
  just a vibe check.

> The core engine, the lexical detector, the mock backend, the metrics, and the
> CLI depend on **nothing but the Python standard library**. Clone it and it runs.

---

## Quickstart

```bash
git clone https://github.com/ramenprotokol/hallucination-hunter.git
cd hallucination-hunter
pip install -e ".[dev]"        # core + pytest; no API keys needed

# check a single answer against a source context
hh detect \
  --answer "The Eiffel Tower is in Paris. It was designed by Leonardo da Vinci." \
  --context "The Eiffel Tower is a wrought-iron tower in Paris, completed in 1889 by Gustave Eiffel's company."

# evaluate a detector over a labeled dataset and print detection quality
hh eval --dataset data/groundedness_eval.jsonl --detector groundedness --format markdown
```

That prints a per-claim verdict — no API key required, because the default
verifier is the deterministic lexical baseline:

```
### adhoc — 🚩 hallucinated

- ✓ supported     — The Eiffel Tower is in Paris.
    3/3 content words found in context
- ? unverifiable  — It was designed by Leonardo da Vinci.
    claim content words largely absent from context
```

The "in Paris" claim passes; the fabricated "da Vinci" claim is flagged.

To run with a real model judge:

```bash
pip install -e ".[anthropic]"
export ANTHROPIC_API_KEY=...
hh eval --dataset data/groundedness_eval.jsonl --verifier llm --model anthropic
```

### Agent mode: `hh investigate`

`hh investigate` checks an answer claim-by-claim with a tool-using agent. Each
FACT gets its own verdict: the answer is decomposed into atomic claims (pass
`--sentences` for cheaper sentence-level runs), then for each claim the model
decides how to gather evidence — search a folder of source documents, run a
groundedness check on a snippet it found, or sample the model for
self-consistency — and commits to a verdict that names the evidence it used,
with a per-claim trace of every tool call.

```bash
# with an OpenAI-compatible endpoint and your own key (no extra install);
# the model must support tool calling
export HH_API_KEY=sk-or-...
export HH_MODEL=<an open-source model with tool calling>
# optional, defaults to https://openrouter.ai/api/v1
export HH_BASE_URL=https://openrouter.ai/api/v1
hh investigate \
  --answer "An adult return is $14 and you can pay cash or card." \
  --prompt "What does a return cost?" \
  --docs ./source-docs \
  --model openai-compat

# a whole JSONL of answers in one go (one object per line, 'answer' required,
# 'question'/'prompt' and 'context' optional) — one combined report; progress
# prints per answer to stderr and a failing answer is skipped, not fatal
hh investigate --answers answers.jsonl --docs ./source-docs --model openai-compat

# offline, deterministic (mock backend, scripted or heuristic)
hh investigate --answer "The crossing takes 40 minutes." --docs ./source-docs
```

Because the agent is just another registered detector, `hh eval --detector
investigate --docs ./source-docs` runs it over a whole dataset too.

**Known limits** — read verdicts with these in mind:

- **No entailment check.** A verified quote proves the text exists in a source
  file, not that it supports the claim. A qualifier next to the quoted words is
  now shown (the evidence line is the full sentence), but nothing yet checks
  that the quote and the claim actually agree.
- **Non-deterministic.** The same answer can split into different claims on
  different runs (observed: 3 vs 8 for one answer), and a claim's verdict can
  flip — one claim came back `unverifiable`, `unverifiable`, then
  `contradicted` across three runs, two of them with the evidence checks
  already in place.
- **Slow and serial.** Every claim investigates independently, re-searching
  and re-reading the same documents; a 3-answer batch with 22 claims took
  ~9 minutes on the model API.

## How it works

An answer is **grounded** if every factual claim it makes is supported by the
provided context. The pipeline checks that one claim at a time:

```mermaid
flowchart LR
    A[Answer] --> B[Claim extraction]
    C[Source context] --> D
    B --> D[Verify each claim]
    D --> E{Supported?}
    E -- yes --> F[grounded]
    E -- no --> G[hallucination:\ncontradicted or unverifiable]
    F & G --> H[Detection]
    H --> I[Metrics:\nhallucination rate,\nprecision / recall / F1]
```

Each claim gets one of three verdicts:

| Verdict | Meaning | Counts as hallucination? |
|---|---|---|
| `supported` | the context entails the claim | no |
| `contradicted` | the context states the opposite | **yes** |
| `unverifiable` | no evidence in the context either way | **yes** |

## Results

The dependency-free **lexical** detector, run over the bundled 160-example
labeled set (`data/groundedness_eval.jsonl`, 271 claims). Positive class is
*hallucinated*:

| Metric | Value |
|---|---|
| Precision | 0.72 |
| Recall | 0.87 |
| F1 | 0.79 |
| Accuracy | 0.82 |

This is an honest baseline, not a leaderboard score: the lexical verifier needs
no model and no API key, so it trades precision for recall and is blind to
paraphrase. Reproduce it yourself — the numbers are generated, not hand-written:

```bash
hh eval --dataset data/groundedness_eval.jsonl --verifier lexical
```

The full report is committed at [`reports/groundedness_lexical.md`](reports/groundedness_lexical.md).
The `llm` verifier (a model judge) trades the offline guarantee for higher
precision.

## Architecture

Two seams keep the tool extensible: **detectors** (how you decide if something is
a hallucination) and **model backends** (which LLM, if any, does the judging).

```mermaid
classDiagram
    class Detector { <<interface>> +detect(Example) Detection }
    class Model { <<interface>> +generate(prompt) str }
    Detector <|-- GroundednessDetector
    Detector <|-- SelfConsistencyDetector
    Detector <|-- ReferenceJudgeDetector
    Model <|-- AnthropicModel
    Model <|-- OpenAIModel
    Model <|-- MockModel
    GroundednessDetector --> Verifier
    Verifier <|-- LexicalVerifier
    Verifier <|-- LLMVerifier
    LLMVerifier --> Model
```

- **Detectors** (`hh.detectors`) — `groundedness` (claims vs context),
  `self_consistency` (sample N times, flag disagreement, no context needed),
  `reference_judge` (judge against a gold answer).
- **Verifiers** — inside groundedness: `lexical` (token-overlap baseline, no
  model) or `llm` (a judge model with a strict rubric).
- **Backends** (`hh.models`) — `anthropic`, `openai`, `mock`. Optional providers
  are imported lazily, so missing them never breaks the core.

See [`docs/architecture.md`](docs/architecture.md), the
[methodology](docs/methodology.md), and the [taxonomy of hallucinations](docs/taxonomy.md).

## Python API

```python
import hh

context = "The Eiffel Tower is a wrought-iron tower in Paris, completed in 1889 by Gustave Eiffel's company."
result = hh.detect("The Eiffel Tower is in Paris. It was designed by Leonardo da Vinci.", context)

print(result.predicted_hallucinated)  # True
for v in result.verdicts:
    print(v.support.value, "-", v.claim.text)
# supported     - The Eiffel Tower is in Paris.
# unverifiable  - It was designed by Leonardo da Vinci.
```

> Note: the default verifier is a lexical (token-overlap) baseline — fast and
> offline, but it can miss contradictions that reuse the context's words (e.g.
> a wrong number). For those, use `verifier="llm"` with a model backend. See
> [`docs/methodology.md`](docs/methodology.md) for the honest tradeoffs.

## Datasets

JSONL, one example per line. The `data/` directory ships labeled sets you can run
immediately; the schema is documented in [`data/README.md`](data/README.md).

```json
{"id": "q001", "prompt": "...", "context": "...", "answer": "...",
 "labels": {"hallucinated": true, "unsupported_claims": ["..."]}}
```

## How this relates to existing work

Groundedness checking builds on ideas from **RAGAS** (RAG faithfulness metrics),
**SelfCheckGPT** (sampling-based consistency), and **DeepEval** (LLM-as-judge
test suites). `hallucination-hunter` is deliberately smaller and more
transparent: a single readable pipeline, a deterministic lexical baseline that
needs no model or API key, and labeled data plus precision/recall/F1 committed
in-repo so every number is reproducible offline. It is a clear reference
implementation, not a benchmark leaderboard — for standardized benchmark scores,
**TruthfulQA**, **FEVER**, and **FActScore** remain the references (see
[`research/summary.md`](research/summary.md)).

## Development

```bash
pip install -e ".[dev]"
pytest            # full suite runs offline, no API keys
ruff check .      # lint
ruff format .     # format
```

## License

MIT — see [LICENSE](LICENSE).
