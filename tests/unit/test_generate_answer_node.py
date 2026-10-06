# RET-C2-087 — Unit Tests: GenerateAnswerNode (inner domain node 4)
#
# Invocation canon: node(state) via BaseNode.__call__ with an ANONYMOUS caller.
#
# S-2 design decision (docs/02_design.md "Security Gates" — deviation from the
# A peer template golden): this node deliberately does NOT interpolate the caller's
# raw search_query into the rendered answer body — the golden's GenerateAnswerNode
# quotes the question back in its lead sentence; this template's queries may
# carry a member ID / points-balance figure the S-2 screen imperfectly
# redacted, so the query is never echoed at all, regardless of content. Every
# test class below exercises BOTH sides of that contract: the answer is
# correctly grounded in ranked_documents, AND the query text never appears in
# it (verified with a query containing a distinctive, easy-to-grep token).
#
# Mirrors docs/03_test_spec.md §2.5 (GEN-01..GEN-06).
# Deterministic — rule-assembled from ranked_documents only (grounded by
# construction; no LLM, no network). framework.* / src.* imports only.

from framework.schemas.trust_level import TrustLevel

from src.nodes.generate_answer_node import GenerateAnswerNode
from src.schemas.state import from_json, to_json

# A query with a distinctive marker token that would be very easy to spot if
# it ever leaked into the answer body.
_MARKER_QUERY = "ZQXMARKER1029384 how many points until redemption"


def _ranked(*entries):
    return to_json(list(entries))


def _doc(doc_id, title, excerpt, category="redemption_catalog", source="seeded kb"):
    return {
        "id": doc_id,
        "title": title,
        "category": category,
        "source": source,
        "score": 0.9,
        "excerpt": excerpt,
    }


def _make_state(ranked_documents, query=_MARKER_QUERY, **extra) -> dict:
    state = {
        "ranked_documents": ranked_documents,
        "search_query": query,
        "caller_trust_level": TrustLevel.ANONYMOUS.value,
        "node_history": [],
        "error_log": [],
        "session_id": "unit-session",
        "execution_time": {},
    }
    state.update(extra)
    return state


class TestGroundedAnswer:
    def test_gen_01_answer_carries_numbered_citation_markers(self):
        ranked = _ranked(
            _doc(
                "kb-011", "Minimum redemption threshold and increments", "the minimum redemption amount is 500 points."
            ),
            _doc("kb-010", "Redemption options overview", "points can be redeemed for digital gift cards."),
        )
        result = GenerateAnswerNode()(_make_state(ranked))
        answer = result["grounded_answer"]
        assert "[1] Minimum redemption threshold and increments:" in answer
        assert "[2] Redemption options overview:" in answer

    def test_gen_02_answer_never_echoes_the_query(self):
        """docs/02_design.md deviation from the golden: the caller's raw query
        (here carrying a distinctive marker token) must NEVER appear in the
        rendered answer — regardless of ranked-document content."""
        ranked = _ranked(_doc("kb-011", "Minimum redemption threshold and increments", "excerpt."))
        result = GenerateAnswerNode()(_make_state(ranked, query=_MARKER_QUERY))
        assert "ZQXMARKER1029384" not in result["grounded_answer"]
        assert _MARKER_QUERY not in result["grounded_answer"]

    def test_gen_02_lead_line_is_the_fixed_generic_line(self):
        # No "the question: ..." quoting habit (contrast with the golden) —
        # the lead line is a fixed, query-independent sentence.
        ranked = _ranked(_doc("kb-011", "Minimum redemption threshold and increments", "excerpt."))
        answer = GenerateAnswerNode()(_make_state(ranked))["grounded_answer"]
        assert answer.startswith(
            "Based on the seeded loyalty-program knowledge base, the following " "passages answer your question:"
        )

    def test_gen_03_citations_mirror_ranked_order(self):
        ranked = _ranked(
            _doc("kb-011", "Minimum redemption threshold and increments", "a.", source="Redemption Catalog, section 2"),
            _doc("kb-010", "Redemption options overview", "b."),
        )
        citations = from_json(GenerateAnswerNode()(_make_state(ranked))["citations"])
        assert [c["ref"] for c in citations] == [1, 2]
        assert [c["id"] for c in citations] == ["kb-011", "kb-010"]
        assert citations[0]["source"] == "Redemption Catalog, section 2"

    def test_citations_is_json_string(self):
        # ADR-005: list-shaped State fields travel as JSON strings.
        ranked = _ranked(_doc("kb-011", "Minimum redemption threshold and increments", "a."))
        result = GenerateAnswerNode()(_make_state(ranked))
        assert isinstance(result["citations"], str)

    def test_gen_04_answer_is_grounded_in_ranked_passages_only(self):
        ranked = _ranked(
            _doc("kb-011", "Minimum redemption threshold and increments", "500 points minimum, in increments of 100.")
        )
        answer = GenerateAnswerNode()(_make_state(ranked))["grounded_answer"]
        # Every content line traces to the single ranked passage.
        assert "500 points minimum, in increments of 100." in answer
        assert "[2]" not in answer

    def test_gen_05_answer_category_is_top_ranked_docs_category(self):
        ranked = _ranked(
            _doc("kb-011", "Minimum redemption threshold and increments", "a.", category="redemption_catalog"),
            _doc("kb-006", "Active promotion: double-points weekend", "b.", category="offer_catalog"),
        )
        result = GenerateAnswerNode()(_make_state(ranked))
        assert result["answer_category"] == "redemption_catalog"


class TestNoCoverage:
    def test_gen_06_empty_ranked_set_yields_no_coverage_answer(self):
        result = GenerateAnswerNode()(_make_state(_ranked()))
        assert "does not contain sufficient coverage" in result["grounded_answer"]
        assert from_json(result["citations"]) == []
        assert result["answer_category"] is None

    def test_missing_ranked_field_is_treated_as_no_coverage(self):
        state = _make_state(None)
        del state["ranked_documents"]
        result = GenerateAnswerNode()(state)
        assert "does not contain sufficient coverage" in result["grounded_answer"]

    def test_no_coverage_answer_also_never_echoes_the_query(self):
        state = _make_state(_ranked(), query=_MARKER_QUERY)
        result = GenerateAnswerNode()(state)
        assert "ZQXMARKER1029384" not in result["grounded_answer"]
