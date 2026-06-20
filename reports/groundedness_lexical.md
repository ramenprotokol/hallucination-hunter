# Hallucination-Hunter Report

- **Detector:** `groundedness:lexical`
- **Examples:** 160
- **Claims checked:** 271
- **Hallucination rate:** 46.9% (answers flagged with >=1 unsupported claim)
- **Mean groundedness:** 66.2% (share of claims supported)

## Detection quality (vs gold labels)

Labeled examples: **160**. Positive class = *hallucinated*.

| Metric | Value |
|---|---|
| Precision | 0.720 |
| Recall | 0.871 |
| F1 | 0.788 |
| Accuracy | 0.819 |

| | gold: hallucinated | gold: clean |
|---|---|---|
| **pred: hallucinated** | 54 (TP) | 21 (FP) |
| **pred: clean** | 8 (FN) | 77 (TN) |

## Sample verdicts

### `ge001` — ✅ grounded
> Who wrote the play Hamlet and when was it written?

- ✓ **supported** — Hamlet was written by William Shakespeare, believed to have been composed between 1599 and 1601.
  - _8/9 content words found in context_
- ✓ **supported** — It is a tragedy centered on Prince Hamlet of Denmark.
  - _4/5 content words found in context_

### `ge002` — ✅ grounded
> What is the speed of light in a vacuum?

- ✓ **supported** — The speed of light in a vacuum is approximately 299,792,458 meters per second, denoted by the constant c.
  - _12/12 content words found in context_

### `ge003` — ✅ grounded
> What is the capital of France?

- ✓ **supported** — The capital of France is Paris, which has been the nation's political and cultural center for centuries.
  - _9/9 content words found in context_

### `ge004` — ✅ grounded
> What is the boiling point of water at standard pressure?

- ✓ **supported** — Water boils at 100 degrees Celsius (212 degrees Fahrenheit) at standard atmospheric pressure.
  - _10/10 content words found in context_

### `ge005` — ✅ grounded
> Who painted the Mona Lisa?

- ✓ **supported** — The Mona Lisa was painted by Leonardo da Vinci, believed to have been created between 1503 and 1519.
  - _11/11 content words found in context_

