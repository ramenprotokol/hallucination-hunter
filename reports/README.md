# Reports

Generated evaluation reports. Every report here is reproducible from the
committed datasets with the command shown at the top of this file.

## `groundedness_lexical.md`

The `groundedness` detector with the **lexical** verifier (token overlap, no
model, no API key) run over the full 160-example labeled set:

```bash
hh eval --dataset data/groundedness_eval.jsonl --verifier lexical \
        --format markdown --out reports/groundedness_lexical.md
```

This is an honest baseline. The lexical verifier is fast and explainable but
blind to paraphrase, so it trades precision for recall. The numbers in the
report are the real output of that command — nothing hand-edited.

## Generating an LLM report

The `llm` verifier needs a real backend and an API key, so its report is **not**
committed (results depend on the model and aren't reproducible offline). To
generate one yourself:

```bash
pip install -e ".[anthropic]"
export ANTHROPIC_API_KEY=...
hh eval --dataset data/groundedness_eval.jsonl --verifier llm --model anthropic \
        --format markdown --out reports/groundedness_llm.md
```

> Note: the `mock` backend exists only to make the test suite and CLI run
> offline. It is **not** a model and its numbers are not a meaningful evaluation
> — never present mock output as a real result.
