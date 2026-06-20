# Contributing

Thanks for taking a look. This project aims to stay small, honest, and runnable.

## Setup

```bash
git clone https://github.com/ramenprotokol/hallucination-hunter.git
cd hallucination-hunter
pip install -e ".[dev]"
pytest
```

The core deliberately depends only on the Python standard library. Please keep it
that way: anything that needs a third-party package (model providers, etc.) goes
behind an optional extra in `pyproject.toml` and is imported lazily, so the test
suite always runs offline with no API keys.

## Ground rules

- **Tests run offline.** Use the `mock` model backend in tests — never the
  network. CI runs `pytest` on Python 3.10–3.13.
- **Add a test with every change.** Bug fix → a test that fails before, passes
  after. New feature → tests that cover it.
- **New detectors** implement the `Detector` interface in `src/hh/detectors/` and
  register with `@register("name")`. See `docs/architecture.md`.
- **New model backends** implement `Model` in `src/hh/models/` and register the
  same way; import the provider SDK lazily.
- **Be honest about results.** Report real metrics. Don't tune a detector against
  the eval set and present it as generalization.

## Before opening a PR

```bash
pytest -q          # all green
hh list            # CLI still imports cleanly
```
