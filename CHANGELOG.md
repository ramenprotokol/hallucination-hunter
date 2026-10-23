# Changelog

All notable changes to this project are documented here.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- `hh cases`: run a fixed set of test cases through `hh investigate` and write a
  plain-text run log (model, system prompt, tools, every tool call, verdicts,
  PASS or FAIL per case).
- Agent mode: `hh investigate` checks an answer claim-by-claim with a tool-using
  agent that can search a folder of source docs (`--docs`), run a groundedness
  check, or sample the model for self-consistency, then commits to a verdict
  naming the evidence it used (with a per-claim tool-call trace).
- Atomic claim splitting (default): the agent decomposes the answer into one
  verdict per FACT — a compound sentence ("bought by a Scottish group in 2019
  and runs a route to Fenwick") yields separate verdicts. A second pass splits
  claims that still contain coordinators; `--sentences` opts out for cheaper
  sentence-level runs.
- `hh investigate` document search folds simple inflections on both the query
  and the corpus, so "car" finds "Cars are not carried" and "sailings" matches
  "sailing"; matching is case-insensitive throughout.
- Budget fairness: when the tool-round budget runs out, the agent forces one
  tools-free turn so the verdict comes from the evidence already gathered
  instead of discarding the investigation.
- `hh investigate --answers FILE`: run a whole JSONL of answers in one go —
  one object per line (`answer` required; `question`/`prompt` and `context`
  optional) — producing one combined report. The markdown report shows every
  answer's verdicts by default (`--show N` to limit). Progress prints per
  answer to stderr (claims, verdict tally, timing); a failing answer is
  reported and skipped instead of sinking the batch — surviving verdicts are
  still emitted, and the exit code is 1 if any answer failed.
- Evidence verification: every quote in a verdict's evidence is re-checked
  against the actual source documents and re-attributed to the document it
  really appears in; quotes that appear nowhere are dropped. Matching ignores
  case and terminal punctuation (cosmetic drift the model introduces) and
  shows the file's own casing, while a substantive paraphrase still fails. A
  `supported` or `contradicted` verdict requires at least one verified quote —
  without one it is downgraded to `unverifiable` with the reason in its
  rationale, unless the investigation itself verified the same call via
  groundedness_check (its verified context quote carries the verdict). The
  rule applies on every final-answer path, including the plain-verdict
  scanner used when the model's final answer is not JSON.
- Verdict confidence now means something and is shown: the report displays
  `(confidence N%)` on verdicts below full confidence. Agent verdicts use a
  fixed scale — 1.0 verified call, 0.9 carried by a tool-verified quote, 0.8
  parsed from a non-JSON answer, 0.5 downgraded, 0.0 budget exhausted — while
  lexical and self-consistency verdicts keep their measured overlap /
  agreement, which the report now surfaces too.
- Qualifier protection: a verified evidence quote is displayed as the FULL
  sentence it appears in (abbreviation-aware — "Ferry Co." is not a sentence
  end — and bounded by its paragraph, so headings stay out). A quote of
  "costs $3 each way" now shows "costs $3 each way, except Saturdays, when
  bikes cost $10." — a qualifier next to the quoted words can no longer hide.
- JSON reports now carry the same detail as markdown: a `detections` array
  with every example (id, prompt, predicted flag, groundedness) and its
  per-claim verdicts — claim, support, score, rationale, evidence, and the
  full tool-call trace — untruncated, since JSON is a data export.
- `Model.generate_with_tools`: an opt-in tool-calling seam (one assistant turn
  per call; the caller owns the loop), implemented by the `mock` backend
  (scripted playback) and the new `openai-compat` backend.
- `openai-compat` model backend: any OpenAI-compatible `/chat/completions`
  endpoint with tool calling, over `urllib` (no third-party dependency).
  Configure with `HH_API_KEY` (your own key), `HH_MODEL` (no default) and
  `HH_BASE_URL` (default `https://openrouter.ai/api/v1`). The key is never
  logged. Turns that come back with null content (a reasoning model that spent
  its whole token budget thinking) are retried with a larger budget.

### Security
- CI runs with a read-only `GITHUB_TOKEN` (`permissions: contents: read`).
- The release workflow pins `pypa/gh-action-pypi-publish` to a full commit SHA (v1.14.2)
  instead of the movable `release/v1` branch, since that step holds the PyPI OIDC token.

### Fixed
- The `llm` verifier no longer reads a judge reply of "unsupported" or "not supported"
  as `supported` (the fallback matched the substring). Such claims are now `unverifiable`,
  so they count as hallucinations. Lexical results are unchanged.

## [0.1.0] - 2026-06-19

### Added
- Core evaluation engine and the `hh` CLI (`list`, `detect`, `eval`).
- Groundedness detector with a deterministic **lexical** verifier (no API key)
  and an **llm** verifier (model judge). Self-consistency and reference-judge
  detectors behind the same `Detector` interface.
- Pluggable model backends: Anthropic, OpenAI, and a deterministic mock.
- Labeled datasets (160 groundedness examples, 112 factual prompts) and
  detection-quality metrics (precision / recall / F1 / accuracy).
- Offline test suite (136 tests, Python 3.10–3.13), CI with lint + type-check +
  coverage, and methodology / taxonomy / rubric / architecture docs.

[Unreleased]: https://github.com/ramenprotokol/hallucination-hunter/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/ramenprotokol/hallucination-hunter/releases/tag/v0.1.0
