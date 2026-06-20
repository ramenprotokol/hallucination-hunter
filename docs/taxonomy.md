# Taxonomy of LLM Hallucinations

This document classifies the hallucination types this tool is designed to detect,
maps each type to the verdict the tool would assign, and gives a concrete example
of each.

The classification follows the intrinsic / extrinsic distinction used in NLP
faithfulness literature, then lists the common failure modes that appear in
practice.

---

## 1. The Intrinsic / Extrinsic Axis

This is the most important distinction because it separates two different
kinds of groundedness failure:

### 1.1 Intrinsic hallucination

The answer contains a claim that **directly contradicts the source**. The source
says one thing; the answer says the opposite or an incompatible version of it.

> **Example.** Source: "The Eiffel Tower was completed in 1889."  
> Answer: "The Eiffel Tower was completed in 1901."  
> Tool verdict: **contradicted** — the number in the answer conflicts with the
> number in the source.

The tool catches these via the `contradicted` verdict. The lexical verifier
approximates this with negation mismatch; the LLM judge handles it more
reliably for numeric and semantic contradictions.

### 1.2 Extrinsic hallucination

The answer contains a claim that is **absent from the source** — the model
added something the document neither confirms nor denies. The model may be
drawing on training-time knowledge, or fabricating outright.

> **Example.** Source: "Marie Curie won the Nobel Prize in Chemistry in 1911."  
> Answer: "Marie Curie won the Nobel Prize in Chemistry in 1911. She was the
> first woman to receive a PhD from the University of Paris."  
> The second sentence is extrinsic — the source does not contain it.  
> Tool verdict: **unverifiable**

Extrinsic claims are not necessarily false, but they are not *grounded* in the
source. This tool treats them as hallucinations because the source is the
agreed-upon ground truth for the evaluation.

---

## 2. Common Failure Modes

Each failure mode below is mapped to its intrinsic / extrinsic category and to
the verdict this tool would most likely assign.

### 2.1 Fabricated entity

A person, place, organisation, or title that does not exist or was not in the
source is asserted as real.

> **Example.** "According to historian Dr. James Fallow, the treaty was signed
> in 1648." (Dr. Fallow is not mentioned anywhere in the source.)  
> Category: extrinsic  
> Tool verdict: **unverifiable** — the source contains no evidence for the
> entity or the attribution.

### 2.2 Wrong date or number

A date, year, statistic, or quantity that conflicts with the one stated in the
source.

> **Example.** Source says "the population was 3.2 million"; the answer says
> "the population was 4.5 million."  
> Category: intrinsic  
> Tool verdict: **contradicted** (LLM judge reliably; lexical verifier may miss
> this if the numbers share no content tokens).

### 2.3 False attribution

A quote or claim is attributed to a person or source that did not say or publish it.

> **Example.** "As Einstein said, 'compound interest is the eighth wonder of the
> world.'" The source does not attribute this quote to Einstein.  
> Category: extrinsic  
> Tool verdict: **unverifiable** — the source contains no evidence linking the
> quote to Einstein.

### 2.4 Temporal error

A time relationship is stated incorrectly — the wrong order of events, an
incorrect duration, or a misplaced sequence.

> **Example.** Source: "The law passed in 2018, two years after the report
> was published." Answer: "The law passed one year after the report."  
> Category: intrinsic  
> Tool verdict: **contradicted** — the time gap contradicts the source.

### 2.5 Overgeneralization

A claim that was true in a limited context is stated as universally true.

> **Example.** Source: "This drug reduced symptoms in 60% of trial participants."  
> Answer: "This drug eliminates symptoms."  
> Category: intrinsic (the weakening of "60% reduction" to "elimination" contradicts
> the quantified claim)  
> Tool verdict: **contradicted** if the LLM judge notices the semantic scope
> mismatch; **unverifiable** with the lexical verifier (the content words mostly
> overlap without flagging the quantifier gap).

### 2.6 Confident nonsense

A fluent, grammatically correct claim that has no factual basis — neither in the
source nor in reality — but is stated with full confidence.

> **Example.** "The speed of light is approximately 150,000 km/s."  
> If the source does not mention light speed, the verifier sees no evidence.  
> Category: extrinsic (absent from source); may also be intrinsic if the source
> gives the correct figure  
> Tool verdict: **unverifiable** (no source evidence) or **contradicted** (if the
> source states the correct value)

---

## 3. Summary Table

| Failure mode | Intrinsic or extrinsic | Typical tool verdict |
|---|---|---|
| Fabricated entity | extrinsic | unverifiable |
| Wrong date / number | intrinsic | contradicted |
| False attribution | extrinsic | unverifiable |
| Temporal error | intrinsic | contradicted |
| Overgeneralization | intrinsic | contradicted (LLM judge) / unverifiable (lexical) |
| Confident nonsense | extrinsic (or intrinsic if source has it) | unverifiable (or contradicted) |

---

## 4. A Note on the Lexical / LLM Verifier Gap

The table above shows that the verifier choice matters. The lexical verifier is
reliable for clear intrinsic cases where a number or proper noun in the claim
is absent from the context entirely. It is less reliable for overgeneralization
and semantic contradictions, where the content words overlap but the meaning
differs. The LLM judge handles these better but is not perfect — it can miss
subtle quantifier errors or apply the rubric inconsistently.

See [docs/rubric.md](rubric.md) for the exact decision criteria a verifier
(human or LLM) should apply.
