"""Tests for the investigate agent: DocStore, tools, loop, verdict fallbacks."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hh.agent import DocStore, InvestigateAgent
from hh.claims import LLMClaimExtractor
from hh.detectors.base import available_detectors, load_detector
from hh.detectors.groundedness import LLMVerifier
from hh.models.base import AssistantTurn, Model, ToolCall
from hh.models.mock import MockModel
from hh.types import Example, Support

FIXTURE_DOCS = Path(__file__).parent / "fixtures" / "investigate" / "sources"

A2_ANSWER = (
    "Harrow Point Ferry Co. was founded in 1987 by Marta Okonkwo. Its current "
    "boat, the Gannet, was built in Sunderland and entered service in 2016. "
    "The company was bought by a Scottish shipping group in 2019 and now also "
    "runs a route to Fenwick Harbour."
)


def _final(support: str, evidence: str = "", rationale: str = "done") -> AssistantTurn:
    payload = {"support": support, "rationale": rationale, "evidence": evidence}
    return AssistantTurn(text=json.dumps(payload))


def _tool_call(id_: str, tool: str, **kwargs) -> AssistantTurn:
    return AssistantTurn(tool_calls=[ToolCall(id=id_, name=tool, arguments=kwargs)])


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class TestAgentRegistry:
    def test_investigate_is_registered(self):
        assert "investigate" in available_detectors()

    def test_load_via_registry(self):
        agent = load_detector("investigate", model=MockModel())
        assert isinstance(agent, InvestigateAgent)

    def test_requires_model(self):
        with pytest.raises(ValueError, match="requires a model"):
            load_detector("investigate")

    def test_rejects_backend_without_tools(self):
        class BareModel(Model):
            def generate(self, prompt, *, system=None, temperature=0.0, max_tokens=1024):
                return ""

        with pytest.raises(ValueError, match="does not support tool calling"):
            InvestigateAgent(model=BareModel())


# ---------------------------------------------------------------------------
# DocStore
# ---------------------------------------------------------------------------


class TestDocStore:
    def test_lists_all_fixture_docs(self):
        store = DocStore(FIXTURE_DOCS)
        assert store.names() == ["01_service.md", "02_fares.md", "03_history.md"]

    def test_search_ranks_relevant_doc_first(self):
        store = DocStore(FIXTURE_DOCS)
        hits = store.search("adult return ticket cost")
        assert hits, "expected at least one hit"
        assert hits[0]["doc"] == "02_fares.md"

    def test_search_no_hits(self):
        store = DocStore(FIXTURE_DOCS)
        assert store.search("quantum entropy decoherence") == []

    def test_search_folds_plurals(self):
        # "Cars are not carried" in 01_service.md must match singular "car".
        store = DocStore(FIXTURE_DOCS)
        hits = store.search("car")
        assert hits, "plural fold should surface the 'Cars are not carried' doc"
        assert hits[0]["doc"] == "01_service.md"

    def test_search_folds_tense_and_plural_together(self):
        store = DocStore(FIXTURE_DOCS)
        hits = store.search("sailing")
        assert any(h["doc"] == "01_service.md" for h in hits)

    def test_search_is_case_insensitive(self):
        store = DocStore(FIXTURE_DOCS)
        upper = store.search("WINTER SCHEDULE")
        lower = store.search("winter schedule")
        assert upper and [h["doc"] for h in upper] == [h["doc"] for h in lower]

    def test_read_unknown_doc_raises(self):
        store = DocStore(FIXTURE_DOCS)
        with pytest.raises(KeyError):
            store.read("nope.md")

    def test_read_truncated_to_tool_cap(self):
        store = DocStore(FIXTURE_DOCS)
        assert len(store.read("01_service.md")) <= 4000


# ---------------------------------------------------------------------------
# Agent loop
# ---------------------------------------------------------------------------


class TestAgentLoop:
    def _answer_example(self) -> Example:
        return Example(
            id="ag1",
            prompt="What does an adult return cost?",
            answer="An adult return ticket costs $14.",
        )

    def test_direct_final_verdict_without_tools(self):
        # No script: the mock's heuristic answers immediately as a final turn.
        agent = InvestigateAgent(model=MockModel())
        det = agent.detect(self._answer_example())
        assert len(det.verdicts) == 1
        assert det.verdicts[0].support in {s for s in Support}
        assert det.detector == "investigate:mock"

    def test_search_then_ground_then_final(self):
        model = MockModel(
            script=[
                _tool_call("c1", "search_docs", query="adult return ticket"),
                _tool_call(
                    "c2",
                    "groundedness_check",
                    claim="An adult return ticket costs $14.",
                    context="An adult return ticket costs $14.",
                ),
                _final(
                    "supported",
                    evidence='"An adult return ticket costs $14." (02_fares.md)',
                ),
            ]
        )
        agent = InvestigateAgent(model=model, docs=FIXTURE_DOCS)
        det = agent.detect(self._answer_example())
        v = det.verdicts[0]
        assert v.support is Support.SUPPORTED
        assert v.evidence == '"An adult return ticket costs $14." (02_fares.md)'
        assert len(v.trace) == 2
        assert v.trace[0].startswith("search_docs(")
        assert v.trace[1].startswith("groundedness_check(")

    def test_evidence_falls_back_to_documents_touched(self):
        model = MockModel(
            script=[
                _tool_call("c1", "search_docs", query="ferry history"),
                _final("unverifiable"),
            ]
        )
        agent = InvestigateAgent(model=model, docs=FIXTURE_DOCS)
        det = agent.detect(self._answer_example())
        v = det.verdicts[0]
        assert v.support is Support.UNVERIFIABLE
        assert "01_service.md" in v.evidence or "03_history.md" in v.evidence

    def test_budget_exhaustion_is_unverifiable(self):
        model = MockModel(
            script=[_tool_call("c1", "search_docs", query="x")]  # repeats forever
        )
        agent = InvestigateAgent(model=model, docs=FIXTURE_DOCS, max_steps=3)
        det = agent.detect(self._answer_example())
        v = det.verdicts[0]
        assert v.support is Support.UNVERIFIABLE
        assert "budget exhausted" in v.rationale
        assert len(v.trace) == 3
        assert v.score == 0.0

    def test_budget_exhaustion_forces_final_answer_from_evidence(self):
        # The model keeps calling tools until the budget is spent; the loop then
        # forces one tools-free turn, which here commits to a supported verdict
        # from the evidence already gathered instead of discarding it all.
        model = MockModel(
            script=[
                _tool_call("c1", "search_docs", query="adult return ticket"),
                _tool_call("c2", "read_doc", name="02_fares.md"),
                _final("supported", evidence="An adult return ticket costs $14."),
            ]
        )
        agent = InvestigateAgent(model=model, docs=FIXTURE_DOCS, max_steps=2)
        det = agent.detect(self._answer_example())
        v = det.verdicts[0]
        assert v.support is Support.SUPPORTED
        assert v.evidence == '"An adult return ticket costs $14." (02_fares.md)'
        assert len(v.trace) == 2

    def test_unknown_tool_reports_error_to_model(self):
        model = MockModel(
            script=[
                _tool_call("c1", "web_search", query="x"),
                _final(
                    "contradicted",
                    evidence='"An adult return ticket costs $14." (02_fares.md)',
                    rationale="sources disagree",
                ),
            ]
        )
        agent = InvestigateAgent(model=model, docs=FIXTURE_DOCS)
        det = agent.detect(self._answer_example())
        v = det.verdicts[0]
        assert v.support is Support.CONTRADICTED
        assert "error" in v.trace[0]

    def test_unparsable_final_falls_back_to_scanner(self):
        model = MockModel(script=[AssistantTurn(text="I could not decide, sorry.")])
        agent = InvestigateAgent(model=model)
        det = agent.detect(self._answer_example())
        v = det.verdicts[0]
        assert v.support is Support.UNVERIFIABLE
        assert "could not parse" in v.rationale

    def test_search_without_docs_errors_gracefully(self):
        model = MockModel(
            script=[
                _tool_call("c1", "search_docs", query="x"),
                _final("supported", evidence='"An adult return ticket costs $14."'),
            ]
        )
        agent = InvestigateAgent(model=model, docs=None)
        det = agent.detect(
            Example(
                id="nodocs",
                answer="An adult return ticket costs $14.",
                context="An adult return ticket costs $14.",
            )
        )
        # The search error went back to the model; the verdict's quote verified
        # against the context, so supported stands.
        assert det.verdicts[0].support is Support.SUPPORTED
        assert "error" in det.verdicts[0].trace[0]


# ---------------------------------------------------------------------------
# Atomic claim splitting
# ---------------------------------------------------------------------------


class TestAtomicClaims:
    def test_compound_sentence_gets_one_verdict_per_fact(self):
        # The last sentence of a2 packs two fabrications; each must get its
        # own verdict instead of one muddled call for the whole sentence.
        extraction_prompt = LLMClaimExtractor.PROMPT.format(answer=A2_ANSWER)
        model = MockModel(
            fixtures={
                extraction_prompt: json.dumps(
                    [
                        "Harrow Point Ferry Co. was founded in 1987 by Marta Okonkwo.",
                        "Its current boat, the Gannet, was built in Sunderland and "
                        "entered service in 2016.",
                        "The company was bought by a Scottish shipping group in 2019.",
                        "The company now also runs a route to Fenwick Harbour.",
                    ]
                )
            },
            script=[_final("unverifiable")],  # repeats; the agent stops on it
        )
        agent = InvestigateAgent(model=model, docs=FIXTURE_DOCS)
        det = agent.detect(
            Example(id="a2", prompt="Tell me about the ferry company.", answer=A2_ANSWER)
        )
        assert len(det.verdicts) == 4
        assert any("Scottish shipping group" in v.claim.text for v in det.verdicts)
        assert any("Fenwick Harbour" in v.claim.text for v in det.verdicts)

    def test_sentences_mode_keeps_sentence_granularity(self):
        extraction_prompt = LLMClaimExtractor.PROMPT.format(answer=A2_ANSWER)
        model = MockModel(
            fixtures={extraction_prompt: json.dumps(["irrelevant — not used in sentences mode"])},
            script=[_final("supported")],
        )
        agent = InvestigateAgent(model=model, docs=FIXTURE_DOCS, atomic=False)
        det = agent.detect(Example(id="a2", answer=A2_ANSWER))
        # Three sentences -> three verdicts; Fenwick still bundled with the
        # Scottish-group claim — the trade-off --sentences makes.
        assert len(det.verdicts) == 3

    def test_atomic_falls_back_to_sentences_on_garbage(self):
        prompt = LLMClaimExtractor.PROMPT.format(answer="One sentence. Two sentence.")
        model = MockModel(fixtures={prompt: "not json"})
        model.script = [_final("supported")]
        agent = InvestigateAgent(model=model)
        det = agent.detect(Example(id="fb", answer="One sentence. Two sentence."))
        assert len(det.verdicts) == 2

    def test_atomic_extraction_call_goes_to_generate(self):
        prompt = LLMClaimExtractor.PROMPT.format(answer="A fact.")
        model = MockModel(fixtures={prompt: '["A fact."]'}, script=[_final("supported")])
        agent = InvestigateAgent(model=model)
        agent.detect(Example(id="g1", answer="A fact."))
        assert any("ATOMIC factual claims" in c for c in model.calls)

    def test_compound_claim_is_refined_into_facts(self):
        # Even when extraction returns a compound claim, the second pass must
        # split it: one verdict for the Scottish group, one for Fenwick.
        answer = A2_ANSWER
        compound = (
            "The company was bought by a Scottish shipping group in 2019 and "
            "now also runs a route to Fenwick Harbour."
        )
        model = MockModel(
            fixtures={
                LLMClaimExtractor.PROMPT.format(answer=answer): json.dumps(
                    [
                        "Harrow Point Ferry Co. was founded in 1987 by Marta Okonkwo.",
                        compound,
                    ]
                ),
                LLMClaimExtractor.SPLIT_PROMPT.format(claim=compound): json.dumps(
                    [
                        "The company was bought by a Scottish shipping group in 2019.",
                        "The company now also runs a route to Fenwick Harbour.",
                    ]
                ),
            },
            script=[_final("unverifiable")],
        )
        agent = InvestigateAgent(model=model, docs=FIXTURE_DOCS)
        det = agent.detect(Example(id="a2", answer=answer))
        assert len(det.verdicts) == 3
        assert any("Scottish shipping group" in v.claim.text for v in det.verdicts)
        assert any("Fenwick Harbour" in v.claim.text for v in det.verdicts)

    def test_refine_keeps_claim_when_model_refuses_to_split(self):
        compound = "The boat was built in Sunderland and entered service in 2016."
        model = MockModel(
            fixtures={
                LLMClaimExtractor.PROMPT.format(answer=compound): json.dumps([compound]),
                # Refusal: model says it is already atomic.
                LLMClaimExtractor.SPLIT_PROMPT.format(claim=compound): json.dumps([compound]),
            },
            script=[_final("supported")],
        )
        agent = InvestigateAgent(model=model)
        det = agent.detect(Example(id="nr", answer=compound))
        assert len(det.verdicts) == 1
        assert det.verdicts[0].claim.text == compound


# ---------------------------------------------------------------------------
# Evidence verification
# ---------------------------------------------------------------------------


class TestEvidenceVerification:
    """Evidence quotes are re-checked against the actual source files."""

    def _detect_with_evidence(
        self, evidence: str, docs=FIXTURE_DOCS, context: str = "", support: str = "supported"
    ):
        turn = AssistantTurn(
            text=json.dumps({"support": support, "rationale": "r", "evidence": evidence})
        )
        model = MockModel(script=[turn])
        agent = InvestigateAgent(model=model, docs=docs)
        return agent.detect(Example(id="ev", answer="A claim.", context=context)).verdicts[0]

    def test_heading_glued_to_attribution_is_stripped(self):
        # The a2 failure: the model glued the doc heading into the evidence
        # line. The quote itself is real, so it must be kept and re-attributed
        # to the document the code actually found it in.
        v = self._detect_with_evidence(
            '"The company was founded in 1987 by Marta Okonkwo, who ran a '
            'single wooden boat called the Petrel." '
            "(03_history.md, titled 'Harrow Point Ferry: history')"
        )
        assert v.evidence == (
            '"The company was founded in 1987 by Marta Okonkwo, who ran a '
            'single wooden boat called the Petrel." (03_history.md)'
        )
        assert "titled" not in v.evidence

    def test_fabricated_quote_is_dropped(self):
        # A quote that appears in no document must not survive as evidence.
        v = self._detect_with_evidence('"Napoleon founded the ferry in 1804." (03_history.md)')
        assert v.evidence == ""

    def test_multiple_quotes_each_attributed(self):
        v = self._detect_with_evidence(
            '"An adult return ticket costs $14." (02_fares.md); '
            '"A crossing takes 40 minutes." (01_service.md)'
        )
        assert '"An adult return ticket costs $14." (02_fares.md)' in v.evidence
        assert '"A crossing takes 40 minutes." (01_service.md)' in v.evidence

    def test_quote_from_context_attributed_to_context(self):
        v = self._detect_with_evidence(
            '"Paris is the capital of France."',
            docs=None,
            context="Paris is the capital of France.",
        )
        assert v.evidence == '"Paris is the capital of France." (context)'

    def test_curly_quotes_are_normalized(self):
        v = self._detect_with_evidence(
            "\u201cAn adult return ticket costs $14.\u201d (02_fares.md)"
        )
        assert v.evidence == '"An adult return ticket costs $14." (02_fares.md)'

    def test_case_drift_still_verifies_with_file_casing(self):
        # The a3 failure mode: the file says "Bicycles cost $3 each way." but
        # the model quoted it with drifted casing. The words match exactly, so
        # the quote verifies — and the evidence line shows the FILE's casing.
        v = self._detect_with_evidence('"bicycles cost $3 each way." (02_fares.md)')
        assert v.evidence == '"Bicycles cost $3 each way." (02_fares.md)'
        assert v.support is Support.SUPPORTED
        assert v.score == 1.0

    def test_terminal_punctuation_drift_still_verifies(self):
        v = self._detect_with_evidence('"Bicycles cost $3 each way!" (02_fares.md)')
        assert '"Bicycles cost $3 each way." (02_fares.md)' in v.evidence
        assert v.support is Support.SUPPORTED
        assert v.score == 1.0

    def test_paraphrased_quote_still_fails(self):
        # Different words (the claim's wording, not the file's) must not pass.
        v = self._detect_with_evidence('"A bike costs three dollars one-way." (02_fares.md)')
        assert v.evidence == ""
        assert v.support is Support.UNVERIFIABLE
        assert "downgraded from supported" in v.rationale
        assert v.score == 0.5

    def test_tool_verified_quote_rescues_sloppy_final_quote(self):
        # The investigation verified the fact (groundedness_check on a real
        # doc snippet), but the model's final evidence paraphrased it. The
        # verified tool quote carries the verdict instead of downgrading.
        verdict_prompt = LLMVerifier.PROMPT.format(
            claim="A bike costs $3 each way.", context="Bicycles cost $3 each way."
        )
        model = MockModel(
            fixtures={verdict_prompt: json.dumps({"support": "supported", "rationale": "matches"})},
            script=[
                _tool_call(
                    "c1",
                    "groundedness_check",
                    claim="A bike costs $3 each way.",
                    context="Bicycles cost $3 each way.",
                ),
                _final("supported", evidence="bikes are three bucks, trust me"),
            ],
        )
        agent = InvestigateAgent(model=model, docs=FIXTURE_DOCS)
        det = agent.detect(Example(id="a3", answer="A bike costs $3 each way."))
        v = det.verdicts[0]
        assert v.support is Support.SUPPORTED
        assert v.evidence == '"Bicycles cost $3 each way." (02_fares.md)'
        assert "downgraded" not in v.rationale
        assert v.score == 0.9  # carried by a tool-verified quote, not its own

    def test_tool_quote_with_opposite_support_does_not_rescue(self):
        verdict_prompt = LLMVerifier.PROMPT.format(
            claim="A bike costs $3 each way.", context="Bicycles cost $3 each way."
        )
        model = MockModel(
            fixtures={verdict_prompt: json.dumps({"support": "contradicted", "rationale": "no"})},
            script=[
                _tool_call(
                    "c1",
                    "groundedness_check",
                    claim="A bike costs $3 each way.",
                    context="Bicycles cost $3 each way.",
                ),
                _final("supported", evidence="made up"),
            ],
        )
        agent = InvestigateAgent(model=model, docs=FIXTURE_DOCS)
        det = agent.detect(Example(id="mis", answer="A bike costs $3 each way."))
        v = det.verdicts[0]
        assert v.support is Support.UNVERIFIABLE
        assert "downgraded from supported" in v.rationale

    def test_whitespace_insensitive_match(self):
        # A quote with ragged internal spacing still matches the file text.
        v = self._detect_with_evidence('"An   adult\treturn\nticket costs $14." (02_fares.md)')
        assert '"An adult return ticket costs $14." (02_fares.md)' in v.evidence

    def test_unquoted_verbatim_text_is_kept(self):
        # Evidence without any quote marks is checked as a whole.
        v = self._detect_with_evidence("A crossing takes 40 minutes.")
        assert v.evidence == '"A crossing takes 40 minutes." (01_service.md)'

    def test_supported_without_verified_quote_is_downgraded(self):
        # "Supported" is earned: no verifiable quote, no support call.
        v = self._detect_with_evidence('"Napoleon founded the ferry in 1804." (03_history.md)')
        assert v.support is Support.UNVERIFIABLE
        assert "downgraded from supported" in v.rationale

    def test_contradicted_without_verified_quote_is_downgraded(self):
        v = self._detect_with_evidence("no real quote", support="contradicted")
        assert v.support is Support.UNVERIFIABLE
        assert "downgraded from contradicted" in v.rationale

    def test_unverifiable_needs_no_quote(self):
        v = self._detect_with_evidence("", support="unverifiable")
        assert v.support is Support.UNVERIFIABLE
        assert "downgraded" not in v.rationale

    def test_supported_with_no_sources_at_all_is_downgraded(self):
        # No docs folder, no context: nothing can be verified, so nothing can
        # be called supported.
        v = self._detect_with_evidence('"An adult return ticket costs $14."', docs=None, context="")
        assert v.support is Support.UNVERIFIABLE
        assert "downgraded from supported" in v.rationale
        assert v.score == 0.5


class TestQualifierProtection:
    """A verified fragment shows the full sentence it is in — no hiding qualifiers."""

    def _detect_with_doc(self, tmp_path, doc_text, evidence):
        doc = tmp_path / "fares.md"
        doc.write_text(doc_text, encoding="utf-8")
        model = MockModel(script=[_final("supported", evidence=evidence)])
        agent = InvestigateAgent(model=model, docs=tmp_path)
        return agent.detect(Example(id="q", answer="A bike costs $3 each way.")).verdicts[0]

    def _fixture_evidence(self, evidence):
        model = MockModel(script=[_final("supported", evidence=evidence)])
        agent = InvestigateAgent(model=model, docs=FIXTURE_DOCS)
        return agent.detect(Example(id="ev", answer="A claim.")).verdicts[0]

    def test_qualifier_after_the_quote_is_shown(self, tmp_path):
        v = self._detect_with_doc(
            tmp_path,
            "# Fares\n\nBicycles cost $3 each way, except Saturdays, when bikes cost $10.\n",
            '"Bicycles cost $3 each way"',
        )
        assert v.support is Support.SUPPORTED
        assert v.evidence == (
            '"Bicycles cost $3 each way, except Saturdays, when bikes cost $10." (fares.md)'
        )

    def test_qualifier_before_the_quote_is_shown(self, tmp_path):
        v = self._detect_with_doc(
            tmp_path,
            "# Fares\n\nBicycles cost $3 each way. Except Saturdays, bikes cost $10 each way.\n",
            '"bikes cost $10 each way"',
        )
        assert v.evidence == '"Except Saturdays, bikes cost $10 each way." (fares.md)'

    def test_fragment_expands_to_whole_sentence(self):
        # Fixture docs: the sentence is "Since March 2024 payment is by card only."
        v = self._fixture_evidence('"payment is by card only"')
        assert v.evidence == '"Since March 2024 payment is by card only." (02_fares.md)'

    def test_abbreviated_company_name_is_not_a_boundary(self):
        v = self._fixture_evidence('"runs a passenger ferry between Harrow Point and Gull Island"')
        assert v.evidence == (
            '"Harrow Point Ferry Co. runs a passenger ferry between Harrow Point and '
            'Gull Island." (01_service.md)'
        )

    def test_heading_is_not_dragged_into_the_evidence(self):
        # The quote sits in the first sentence after a markdown heading; the
        # paragraph boundary keeps the heading out of the expansion.
        v = self._fixture_evidence('"The company was founded in 1987"')
        assert v.evidence == (
            '"The company was founded in 1987 by Marta Okonkwo, who ran a single '
            'wooden boat called the Petrel." (03_history.md)'
        )

    def test_two_fragments_of_one_sentence_dedupe(self):
        v = self._fixture_evidence('"payment is by card only" and "Since March 2024"')
        assert v.evidence == '"Since March 2024 payment is by card only." (02_fares.md)'


class TestScannerPathEvidenceRule:
    """Non-JSON finals go through the plain-verdict scanner — same evidence rule."""

    def _scan_agent(self, script, fixtures=None):
        model = MockModel(script=script, fixtures=fixtures or {})
        return InvestigateAgent(model=model, docs=FIXTURE_DOCS)

    def test_scanner_supported_without_quote_is_downgraded(self):
        agent = self._scan_agent([AssistantTurn(text="This claim is supported.")])
        v = agent.detect(Example(id="s1", answer="A claim.")).verdicts[0]
        assert v.support is Support.UNVERIFIABLE
        assert "downgraded from supported" in v.rationale

    def test_scanner_contradicted_without_quote_is_downgraded(self):
        agent = self._scan_agent([AssistantTurn(text="The claim is contradicted.")])
        v = agent.detect(Example(id="s2", answer="A claim.")).verdicts[0]
        assert v.support is Support.UNVERIFIABLE
        assert "downgraded from contradicted" in v.rationale

    def test_scanner_supported_rescued_by_tool_quote(self):
        verdict_prompt = LLMVerifier.PROMPT.format(
            claim="A bike costs $3 each way.", context="Bicycles cost $3 each way."
        )
        agent = self._scan_agent(
            [
                _tool_call(
                    "c1",
                    "groundedness_check",
                    claim="A bike costs $3 each way.",
                    context="Bicycles cost $3 each way.",
                ),
                AssistantTurn(text="This claim is supported."),
            ],
            fixtures={verdict_prompt: json.dumps({"support": "supported", "rationale": "m"})},
        )
        v = agent.detect(Example(id="s3", answer="A bike costs $3 each way.")).verdicts[0]
        assert v.support is Support.SUPPORTED
        assert v.evidence == '"Bicycles cost $3 each way." (02_fares.md)'
        assert v.score == 0.9  # rescued: non-JSON final + tool-verified quote

    def test_scanner_unverifiable_untouched_by_rule(self):
        agent = self._scan_agent([AssistantTurn(text="I could not decide, sorry.")])
        v = agent.detect(Example(id="s4", answer="A claim.")).verdicts[0]
        assert v.support is Support.UNVERIFIABLE
        assert "downgraded" not in v.rationale
        assert v.score == 0.8  # parsed from a non-JSON answer


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


class TestAgentTools:
    def test_self_consistency_tool_agreement(self):
        model = MockModel(
            script=[
                _tool_call(
                    "c1",
                    "self_consistency_check",
                    claim="An adult return ticket costs $14.",
                    prompt="How much is a return?",
                    samples=2,
                ),
                _final(
                    "supported",
                    rationale="samples agree",
                    evidence='"An adult return ticket costs $14."',
                ),
            ]
        )
        model.add_fixture("How much is a return?", "An adult return ticket costs $14.")
        agent = InvestigateAgent(model=model, samples=2)
        det = agent.detect(
            Example(
                id="sc",
                prompt="How much is a return?",
                answer="An adult return ticket costs $14.",
                context="An adult return ticket costs $14.",
            )
        )
        v = det.verdicts[0]
        assert v.support is Support.SUPPORTED
        assert '"agreement": 1.0' in v.trace[0]

    def test_self_consistency_disagreement_is_reported(self):
        model = MockModel(
            script=[
                _tool_call(
                    "c1",
                    "self_consistency_check",
                    claim="An adult return ticket costs $14.",
                    prompt="How much is a return?",
                    samples=2,
                ),
                _final("unverifiable", rationale="no agreement"),
            ]
        )
        model.add_fixture("How much is a return?", "Penguins live in Antarctica.")
        agent = InvestigateAgent(model=model, samples=2)
        det = agent.detect(
            Example(
                id="sc",
                prompt="How much is a return?",
                answer="An adult return ticket costs $14.",
            )
        )
        assert '"agreement": 0.0' in det.verdicts[0].trace[0]

    def test_groundedness_tool_via_mock_heuristic(self):
        model = MockModel(
            script=[
                _tool_call(
                    "c1",
                    "groundedness_check",
                    claim="The crossing takes 40 minutes.",
                    context="A crossing takes 40 minutes.",
                ),
                _final("supported", evidence='"A crossing takes 40 minutes."'),
            ]
        )
        agent = InvestigateAgent(model=model, docs=FIXTURE_DOCS)
        det = agent.detect(Example(id="gc", answer="The crossing takes 40 minutes."))
        v = det.verdicts[0]
        assert v.support is Support.SUPPORTED
        assert '"support": "supported"' in v.trace[0]
