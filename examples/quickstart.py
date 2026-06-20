"""quickstart.py — hallucination-hunter API walkthrough.

Demonstrates two things:

  (a) hh.detect() on one grounded and one hallucinated example, no API key needed.
  (b) hh.load_dataset() + hh.run_eval() over data/groundedness_eval.jsonl,
      using the lexical verifier, printing results via hh.report.to_markdown().

Run from the repo root with the package installed:

    pip install -e .
    python examples/quickstart.py

Or without installing (PYTHONPATH):

    PYTHONPATH=src python examples/quickstart.py
"""

import hh
import hh.report

# ---------------------------------------------------------------------------
# (a) Single-answer detection — grounded and hallucinated examples
# ---------------------------------------------------------------------------

# The source context is the ground truth. The verifier checks claims against it.
context = (
    "The Eiffel Tower is a wrought-iron lattice tower in Paris, France. "
    "It was designed and built by Gustave Eiffel's engineering company and "
    "completed in 1889 for the World's Fair."
)

# --- Example 1: grounded answer ---
# Both claims ("in Paris" and "1889") are directly supported by the context.
grounded_answer = "The Eiffel Tower is in Paris. It was completed in 1889."

result_grounded = hh.detect(grounded_answer, context)
# predicted_hallucinated is True if *any* claim is not supported.
print("=== Grounded answer ===")
print(f"Answer   : {grounded_answer}")
print(f"Flagged  : {result_grounded.predicted_hallucinated}")  # expect: False
print(f"Groundedness: {result_grounded.groundedness:.0%}")  # expect: 100%
for v in result_grounded.verdicts:
    print(f"  [{v.support.value:>13}] {v.claim.text}")
print()

# --- Example 2: hallucinated answer ---
# "Leonardo da Vinci" contradicts the context (Gustave Eiffel built it).
# The lexical verifier catches this as unverifiable because "Leonardo da Vinci"
# shares no content words with the context — it falls below the support threshold.
hallucinated_answer = "The Eiffel Tower is in Paris. It was designed by Leonardo da Vinci."

result_hallucinated = hh.detect(hallucinated_answer, context)
print("=== Hallucinated answer ===")
print(f"Answer   : {hallucinated_answer}")
print(f"Flagged  : {result_hallucinated.predicted_hallucinated}")  # expect: True
print(f"Groundedness: {result_hallucinated.groundedness:.0%}")  # expect: 50%
for v in result_hallucinated.verdicts:
    marker = "HALL" if v.is_hallucination else "  ok"
    print(f"  [{marker} | {v.support.value:>13}] {v.claim.text}")
    if v.rationale:
        print(f"    reason: {v.rationale}")
print()


# ---------------------------------------------------------------------------
# (b) Dataset evaluation — load JSONL, run eval, print metrics as Markdown
# ---------------------------------------------------------------------------

DATASET_PATH = "data/groundedness_eval.jsonl"

try:
    # load_dataset() reads a JSONL file, one Example per line.
    examples = hh.load_dataset(DATASET_PATH)

    # run_eval() runs the detector over every example and aggregates metrics.
    # We use the groundedness detector with the lexical verifier — no API key.
    detector = hh.load_detector("groundedness", verifier="lexical")
    run = hh.run_eval(examples, detector)

    # to_markdown() produces a human-readable report including the confusion
    # matrix and precision/recall/F1 when gold labels are present.
    print("=== Dataset evaluation ===")
    print(hh.report.to_markdown(run))

except FileNotFoundError:
    print(
        f"[quickstart] Dataset not found at '{DATASET_PATH}'.\n"
        "To run the dataset portion of this demo, either:\n"
        "  - ensure you're running from the repo root, or\n"
        "  - create a JSONL file at that path with labeled examples.\n"
        "See data/README.md for the schema.\n"
        "The single-answer detection demo above ran successfully."
    )
