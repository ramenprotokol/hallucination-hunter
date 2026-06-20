# Examples

## quickstart.py

A runnable walkthrough of the `hh` Python API:

1. `hh.detect()` on one grounded and one hallucinated answer, printing per-claim
   verdicts.
2. `hh.load_dataset()` + `hh.run_eval()` over `data/groundedness_eval.jsonl`,
   using the lexical verifier, printing results as a Markdown report.

The entire script uses only the lexical verifier and the mock backend — no API
keys required.

### How to run

From the repo root, with the package installed:

```bash
pip install -e .
python examples/quickstart.py
```

Without installing (using PYTHONPATH instead):

```bash
PYTHONPATH=src python examples/quickstart.py
```

The dataset portion degrades gracefully if `data/groundedness_eval.jsonl` is not
present — it prints a helpful message and the single-answer detection portion
still runs.
