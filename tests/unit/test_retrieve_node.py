# RET-C2-087 — Unit Tests: RetrieveNode (inner domain node 2)
#
# Invocation canon: node(state) via BaseNode.__call__ with an ANONYMOUS caller.
# RULES C2 ⛔ RETIRED 2026-07-27: execute(self, state) takes NO config
# parameter — nodes.execute(state, config=...) would TypeError. Config knobs
# (kb_path / top_k / score_threshold) are exercised EXCLUSIVELY via the
# state-seeded `retrieval_config` JSON field (the same field
# DomainWorkflowGraph._extra_initial_state() republishes at runtime), always
# through node(state).
#
# Mirrors docs/03_test_spec.md §2.3 (RET-01..RET-08).
# Deterministic — keyword scoring over the seeded config/kb/loyalty_kb.json;
# no LLM, no network. framework.* / src.* imports only.

from framework.schemas.trust_level import TrustLevel

from src.nodes.retrieve_node import RetrieveNode
from src.schemas.state import from_json, to_json

_REDEMPTION_QUERY = "minimum points needed to redeem a gift card"


def _make_state(query=_REDEMPTION_QUERY, **extra) -> dict:
    state = {
        "search_query": query,
        "caller_trust_level": TrustLevel.ANONYMOUS.value,
        "node_history": [],
        "error_log": [],
        "session_id": "unit-session",
        "execution_time": {},
    }
    state.update(extra)
    return state


class TestRetrieveHappyPath:
    def test_ret_01_top_hit_is_redemption_threshold_entry(self):
        result = RetrieveNode()(_make_state())
        docs = from_json(result["retrieved_documents"])
        assert docs, "expected candidates for the redemption query"
        assert docs[0]["id"] == "kb-011"

    def test_ret_02_scores_sorted_descending(self):
        docs = from_json(RetrieveNode()(_make_state())["retrieved_documents"])
        scores = [d["score"] for d in docs]
        assert scores == sorted(scores, reverse=True)
        assert all(s > 0.0 for s in scores)

    def test_ret_03_entry_shape_and_excerpt_cap(self):
        docs = from_json(RetrieveNode()(_make_state())["retrieved_documents"])
        for doc in docs:
            assert set(doc.keys()) == {"id", "title", "category", "source", "score", "excerpt"}
            assert len(doc["excerpt"]) <= 400

    def test_retrieved_documents_is_json_string(self):
        # ADR-005: list-shaped State fields travel as JSON strings.
        result = RetrieveNode()(_make_state())
        assert isinstance(result["retrieved_documents"], str)


class TestRetrieveFilters:
    def test_ret_04_category_filter_restricts_pool(self):
        state = _make_state(
            query="birthday bonus points",
            query_filters=to_json({"category": "offer_catalog", "top_k": None}),
        )
        docs = from_json(RetrieveNode()(state)["retrieved_documents"])
        assert docs, "offer_catalog has seeded entries"
        assert {d["category"] for d in docs} == {"offer_catalog"}
        assert docs[0]["id"] == "kb-008"

    def test_ret_05_empty_query_yields_no_candidates(self):
        docs = from_json(RetrieveNode()(_make_state(query=""))["retrieved_documents"])
        assert docs == []


class TestRetrieveConfigViaState:
    """RULES C2 retired: config knobs travel ONLY through the state-seeded
    retrieval_config field — every call here still goes through node(state)."""

    def test_ret_06_state_retrieval_config_kb_path_override(self):
        state = _make_state(retrieval_config=to_json({"kb_path": "config/kb/does_not_exist.json"}))
        result = RetrieveNode()(state)
        assert from_json(result["retrieved_documents"]) == []
        notes = from_json(result.get("intake_notes"), [])
        assert any("not readable" in n for n in notes)

    def test_ret_07_state_retrieval_config_top_k_shapes_pool_size(self):
        # "points" scores >0 against 11 of the 12 seeded KB entries. Candidate
        # pool = max(top_k * 3, 10): top_k=1 -> pool 10 (truncates to 10 of
        # 11); the module default top_k=4 -> pool 12 (keeps all 11). This
        # pins the floor AND proves state-seeded top_k actually reaches the
        # node (execute() takes no config parameter — RULES C2 retired).
        narrow = from_json(
            RetrieveNode()(_make_state(query="points", retrieval_config=to_json({"top_k": 1})))["retrieved_documents"]
        )
        wide = from_json(RetrieveNode()(_make_state(query="points"))["retrieved_documents"])
        assert len(narrow) == 10
        assert len(wide) == 11

    def test_ret_08_missing_retrieval_config_falls_back_to_module_defaults(self):
        # No retrieval_config key at all — the node's own _DEFAULT_RETRIEVAL
        # (kb_path=config/kb/loyalty_kb.json) must still resolve the seeded KB.
        docs = from_json(RetrieveNode()(_make_state())["retrieved_documents"])
        assert docs and docs[0]["id"] == "kb-011"


class TestRetrieveNotesAccumulation:
    def test_ret_09_notes_append_never_clobber(self):
        state = _make_state(
            intake_notes=to_json(["earlier note from input validation"]),
            retrieval_config=to_json({"kb_path": "config/kb/bogus.json"}),
        )
        result = RetrieveNode()(state)
        notes = from_json(result["intake_notes"])
        assert notes[0] == "earlier note from input validation"
        assert len(notes) == 2
