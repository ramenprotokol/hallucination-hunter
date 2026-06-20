# Related Work: Literature Note

This note situates `hallucination-hunter` within the landscape of prior work on
LLM hallucination detection and faithfulness evaluation. Coverage is intentionally
brief — two to four sentences per reference — and limited to what is directly
relevant to the tool's groundedness and self-consistency approaches.

**Disclaimer:** the references below are described from published knowledge and
should be verified against primary sources before formal citation. ArXiv IDs and
exact URLs are omitted where there is any uncertainty about their correctness;
authors, year, and title are sufficient to locate the originals.

---

## SelfCheckGPT (Manakul, Liusie, and Gales, 2023)

Title: *SelfCheckGPT: Zero-Resource Black-Box Hallucination Detection for
Generative Large Language Models*

The core idea: if a model truly knows a fact, it will state it consistently
across multiple independent samples from the same prompt. If it is fabricating,
the samples will disagree. SelfCheckGPT operationalises this by sampling a model
several times at non-zero temperature, then measuring how consistently each
sentence in a reference response is echoed by those samples — without any
external knowledge source.

`hallucination-hunter`'s `SelfConsistencyDetector` follows this premise directly.
It extracts claims from a candidate answer, draws N samples of the model on the
same prompt, and measures token-overlap agreement between each claim and the
samples. Low consistency triggers an `unverifiable` verdict. The main difference
is that this tool uses a simpler lexical agreement measure rather than the
BERTScore and NLI variants explored in the original paper.

---

## FActScore (Min, Krishna, Lyu, Lewis, Yih, Koh, Iyyer, Zettlemoyer, and Hajishirzi, 2023)

Title: *FActScore: Fine-grained Atomic Evaluation of Factual Precision in Long
Form Text Generation*

FActScore decomposes a long-form generated answer into atomic claims and checks
each claim against a retrieval corpus (typically Wikipedia), reporting precision
over claims rather than over whole answers. The decompose-then-verify pattern is
central, and FActScore uses language models both for decomposition and for the
verdict step.

`hallucination-hunter`'s `GroundednessDetector` uses the same decompose-then-verify
structure. The practical difference is scope: this tool operates against a
user-supplied context (not a retrieval corpus), making it suitable for RAG-style
evaluation where the relevant source document is known. The lexical verifier is a
cheaper, offline stand-in for the model-based NLI verifier FActScore uses; the
`llm` verifier moves closer to FActScore's approach.

---

## RAGAS Faithfulness Metric (Es, James, Espinosa-Anke, and Schockaert, 2023)

Title: *RAGAS: Automated Evaluation of Retrieval Augmented Generation*

RAGAS provides an evaluation suite for retrieval-augmented generation, with a
faithfulness metric that checks whether each statement in a generated answer is
grounded in the retrieved context. The metric is specifically designed for the
RAG setting where a context document is always available, and it uses an LLM
judge to assign claim-level verdicts.

The `groundedness` detector in `hallucination-hunter` addresses the same use case
and the same measurement question. The structural difference is that RAGAS is a
full RAG evaluation framework (covering also answer relevance and context
precision), while this tool is focused purely on the faithfulness / hallucination
detection piece and exposes explicit verifier choices (lexical vs LLM) so users
can trade accuracy for cost.

---

## TruthfulQA (Lin, Hilton, and Evans, 2022)

Title: *TruthfulQA: Measuring How Models Tell the Truth*

TruthfulQA is a benchmark of 817 questions designed to elicit imitative
falsehoods — answers that seem plausible because humans commonly believe them but
are factually wrong. It evaluates whether models can avoid stating falsehoods
they have absorbed during training, measured against reference true and false
answers by a trained classifier.

The connection to `hallucination-hunter` is methodological: TruthfulQA
demonstrates that detection quality needs to be measured against gold labels
rather than trusted implicitly. This tool adopts the same principle — `EvalResult`
reports precision, recall, and F1 against human-labeled hallucination flags when
they are present in the dataset. TruthfulQA's evaluation structure (question +
reference answers + binary hallucinated/not label) maps cleanly onto the JSONL
schema used here.

---

## FEVER (Thorne, Vlachos, Christodoulopoulos, and Mittal, 2018)

Title: *FEVER: A Large-scale Dataset for Fact Extraction and VERification*

FEVER introduced the task of fact verification against a knowledge base: given a
claim, classify it as SUPPORTED, REFUTED, or NOT ENOUGH INFO using evidence
retrieved from Wikipedia. This three-way verdict structure is the direct
intellectual precursor to the `Support` enum in `hallucination-hunter`
(`supported`, `contradicted`, `unverifiable`). The labels map almost exactly:
SUPPORTED → supported, REFUTED → contradicted, NOT ENOUGH INFO → unverifiable.

The key design difference is that FEVER operates in an information-retrieval
setting (evidence must be found), while this tool operates in a grounding setting
(context is provided and must be checked). The task is simpler here — no retrieval
step — but the verdict semantics and the measurement framework (precision, recall
over the positive class) are inherited from the FEVER tradition.
