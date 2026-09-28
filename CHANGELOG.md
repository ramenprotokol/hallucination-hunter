# Changelog

All notable changes to this project are documented here.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
