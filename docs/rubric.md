# Verifier Decision Rubric

This document specifies the exact decision criteria for assigning one of three
verdicts to a claim: `supported`, `contradicted`, or `unverifiable`.

The rubric applies to both human annotators and LLM judge verifiers. The LLM
verifier in this tool (`LLMVerifier`) is explicitly prompted with these
definitions. Human reviewers validating or correcting verdicts should use this
document as their standard.

---

## The Three Verdicts, Precisely Defined

### `supported`

The context **entails** the claim. That means a reader who knows only the
context text would conclude the claim is true. The support may be:

- **Direct** — the context contains the same information in similar words.
- **Semantic / paraphrase** — the context contains the same information but
  expressed differently (synonyms, reordered sentence, different phrasing).
- **Inferential** — the claim follows directly from the context by one
  unambiguous inference step (e.g., context says "built in 1889", claim says
  "more than 100 years old" — calculable from the year alone).

The claim must not assert more precision or specificity than the context
provides. See edge case 4 below.

### `contradicted`

The context **states the opposite** of the claim, or states something
incompatible with it. This includes:

- **Direct denial** — context says X, claim says not-X.
- **Numeric mismatch** — context gives a different number, date, or quantity.
- **Scope mismatch** — context qualifies a statement (e.g., "in some cases")
  and the claim drops the qualifier (e.g., "always").
- **Agent or attribution mismatch** — context attributes action to A, claim
  attributes it to B.

**Standard:** there must be a specific piece of context the claim conflicts
with. A contradiction is not "I doubt this" — it requires a concrete
counter-statement in the text.

### `unverifiable`

The context contains **no evidence** for or against the claim. The claim may
be true (based on world knowledge), false, or unknown — but the context does
not say either way.

**Standard:** if a reader knew only the context text, they could not determine
whether the claim is true or false. When uncertain between `supported` and
`unverifiable`, prefer `unverifiable`. When uncertain between `contradicted` and
`unverifiable`, prefer `contradicted` only if a specific conflicting statement
exists in the context.

---

## Worked Examples: `supported`

**Example 1 — direct match**

> Context: "The Eiffel Tower is a wrought-iron lattice tower in Paris, completed
> in 1889 by Gustave Eiffel's company."
>
> Claim: "The Eiffel Tower is located in Paris."
>
> Verdict: **supported** — "in Paris" is stated directly.

**Example 2 — paraphrase**

> Context: "Marie Curie won Nobel Prizes in Physics (1903) and Chemistry (1911)."
>
> Claim: "Curie received two Nobel Prizes."
>
> Verdict: **supported** — the context lists exactly two Nobel Prizes. The
> phrasing is different but the information is identical.

**Example 3 — single-step inference**

> Context: "The treaty was signed in 1648."
>
> Claim: "The treaty is more than 300 years old."
>
> Verdict: **supported** — 2024 − 1648 = 376, which is more than 300. The
> arithmetic is trivial and unambiguous.

**Example 4 — synonym**

> Context: "The patient died three hours after admission."
>
> Claim: "The patient passed away within hours of being admitted."
>
> Verdict: **supported** — "died" and "passed away" are synonyms; "within hours"
> is consistent with "three hours". An LLM judge would catch this; the lexical
> verifier would likely miss it.

---

## Worked Examples: `contradicted`

**Example 1 — numeric mismatch**

> Context: "The building is 324 metres tall."
>
> Claim: "The building is 450 metres tall."
>
> Verdict: **contradicted** — two different numbers are asserted for the same
> quantity; they cannot both be true.

**Example 2 — attribution mismatch**

> Context: "The theorem was proved by Gauss in 1801."
>
> Claim: "Euler proved the theorem."
>
> Verdict: **contradicted** — the context names a different author.

**Example 3 — scope dropped**

> Context: "The drug reduced symptoms in approximately 60% of trial participants."
>
> Claim: "The drug eliminates symptoms."
>
> Verdict: **contradicted** — "60% reduction in trial participants" is
> incompatible with "eliminates [in all cases]". The claim drops the
> quantifier and the qualifier, inverting the meaning.

**Example 4 — direct denial**

> Context: "No animals were harmed during the experiment."
>
> Claim: "Animals were used in the experiment."
>
> Verdict: **contradicted** — the context specifies no harm, but being
> used is different from being harmed. Actually this is ambiguous — but
> if the context had said "No animals were involved", then the claim
> would be **contradicted**. Ambiguous cases like this one should be
> flagged as **unverifiable** unless the negation is explicit and
> covers the exact predicate in the claim.

---

## Worked Examples: `unverifiable`

**Example 1 — absent entity**

> Context: "The Eiffel Tower was completed in 1889."
>
> Claim: "The architect trained in Paris."
>
> Verdict: **unverifiable** — the context says nothing about any architect's
> training location.

**Example 2 — plausible but unsupported elaboration**

> Context: "Marie Curie won the Nobel Prize in Chemistry in 1911."
>
> Claim: "She was the first woman to win the Nobel Prize in Chemistry."
>
> Verdict: **unverifiable** — this may be true, but the context does not state it.
> The context only says she won, not that she was the first.

**Example 3 — world knowledge leak**

> Context: "The company was founded in San Francisco."
>
> Claim: "San Francisco is in California."
>
> Verdict: **unverifiable** — the context says nothing about which state San
> Francisco is in. The claim may be world-knowledge-true, but the verifier's job
> is to check the *context*, not general knowledge.

**Example 4 — temporally shifted claim**

> Context: "As of 2020, the population was 3.2 million."
>
> Claim: "The current population is 3.2 million."
>
> Verdict: **unverifiable** — the context gives a figure for 2020; whether that
> is still current depends on information not in the context.

---

## Tricky Edge Cases

### Partial support

**Situation:** the context supports part of a compound claim but is silent on
another part.

> Context: "The author wrote three novels and one short story collection."
>
> Claim: "The author wrote three novels and received the Booker Prize."
>
> Verdict: **unverifiable** — the first half is supported but the Booker Prize
> is absent from the context. Because the claim as a whole cannot be confirmed,
> the verdict is `unverifiable`. Do not split the claim; evaluate it as stated.

### Paraphrase that the lexical verifier misses

**Situation:** a claim is semantically supported but shares few content words
with the context.

> Context: "The patient recovered fully."
>
> Claim: "The patient was discharged in good health."
>
> A human or LLM judge should mark this **supported** (full recovery implies
> discharge in good health). The lexical verifier may mark it **unverifiable**
> because "recovered" and "discharged in good health" share no significant
> content words.

This is a known limitation of the lexical verifier. The verdict depends on the
verifier, and the rubric is the authority — the lexical verifier's output may
be wrong in such cases.

### Numeric claim more specific than context

**Situation:** the context gives a range or approximation; the claim gives an
exact number within that range.

> Context: "The journey takes roughly two to three hours."
>
> Claim: "The journey takes 2.5 hours."
>
> Verdict: **unverifiable** — the context gives a range, not a specific figure.
> 2.5 hours is within the range but the context does not affirm that specific
> value. The claim is more precise than the evidence allows.

### Claim broader than context but consistent

**Situation:** the context gives a specific case; the claim generalises it.

> Context: "In this trial, the treatment was effective."
>
> Claim: "The treatment is effective."
>
> Verdict: **contradicted** — the context qualifies effectiveness to one trial;
> the claim drops the qualifier and implies general effectiveness. Prefer
> `contradicted` over `unverifiable` when the claim widens scope in a way that
> is inconsistent with the hedged language of the source.

### Negation scope

**Situation:** both claim and context use negation, but on different predicates.

> Context: "The committee did not approve the proposal."
>
> Claim: "The proposal was not submitted."
>
> Verdict: **unverifiable** — the context negates approval, not submission. The
> claim negates submission. These are different predicates; no contradiction exists,
> and no support exists for the claim.

---

## Summary Decision Tree

```
Does the context contain evidence about the claim?
├── No → unverifiable
└── Yes → Does the context entail the claim (directly or by paraphrase/inference)?
    ├── Yes → supported
    └── No → Does the context state something incompatible with the claim?
        ├── Yes → contradicted
        └── No (context mentions the topic but is neutral) → unverifiable
```

When in doubt, prefer `unverifiable` over `supported`, and prefer `contradicted`
only when a concrete conflicting statement exists in the context.
