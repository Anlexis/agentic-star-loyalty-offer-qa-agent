# RET-C2-087 — Unit Tests: InputValidateNode (inner domain node 1)
#
# Invocation canon: node(state) via BaseNode.__call__ with an ANONYMOUS caller
# (inner domain node). Payloads are lowercase and identifier-free so the
# framework's own input mask leaves them untouched.
#
# Deterministic — no LLM, no network. framework.* / src.* imports only.

import json

import pytest

from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel

from src.nodes.input_validate_node import InputValidateNode
from src.schemas.state import from_json


def _make_state(payload, **extra) -> dict:
    state = {
        "validated_input": payload,
        "caller_trust_level": TrustLevel.ANONYMOUS.value,
        "node_history": [],
        "error_log": [],
        "session_id": "unit-session",
        "execution_time": {},
    }
    state.update(extra)
    return state


class TestPlainTextParsing:
    def test_val_01_plain_text_becomes_query(self):
        result = InputValidateNode()(_make_state("points expiry policy"))
        assert result["search_query"] == "points expiry policy"
        filters = from_json(result["query_filters"])
        assert filters == {"category": None, "top_k": None, "score_threshold": None}

    def test_val_02_whitespace_is_collapsed(self):
        result = InputValidateNode()(_make_state("  points   expiry\n policy "))
        assert result["search_query"] == "points expiry policy"

    def test_query_filters_is_json_string(self):
        # ADR-005: structured State fields travel as JSON strings, never dicts.
        result = InputValidateNode()(_make_state("points expiry"))
        assert isinstance(result["query_filters"], str)
        assert isinstance(from_json(result["query_filters"]), dict)


class TestJsonEnvelopeParsing:
    def test_val_03_envelope_query_category_top_k(self):
        payload = json.dumps({"query": "redemption processing time", "category": "redemption_catalog", "top_k": 2})
        result = InputValidateNode()(_make_state(payload))
        assert result["search_query"] == "redemption processing time"
        filters = from_json(result["query_filters"])
        assert filters == {
            "category": "redemption_catalog",
            "top_k": 2,
            "score_threshold": None,
        }

    def test_question_alias_accepted(self):
        payload = json.dumps({"question": "when do unused points expire?"})
        result = InputValidateNode()(_make_state(payload))
        assert result["search_query"] == "when do unused points expire?"

    def test_category_is_normalised(self):
        payload = json.dumps({"query": "welcome bonus", "category": "  Offer_Catalog "})
        result = InputValidateNode()(_make_state(payload))
        assert from_json(result["query_filters"])["category"] == "offer_catalog"

    def test_val_04_malformed_json_falls_back_to_plain_text(self):
        payload = "{ this is not valid json but starts like it"
        result = InputValidateNode()(_make_state(payload))
        assert result["search_query"] == payload
        notes = from_json(result.get("intake_notes"), [])
        assert any("did not parse" in n for n in notes)


class TestTopKGuard:
    """VAL-05..07: the caller-supplied top_k is untrusted and guarded."""

    @pytest.mark.parametrize("bad", [99, -5, 0, 2.5, True, "many", "", None])
    def test_val_05_a_top_k_outside_its_bounds_refuses_the_request(self, bad):
        # Neither clamped nor dropped: substituting a different number answers a
        # question the caller did not ask, and nothing tells them it happened.
        payload = json.dumps({"query": "points expiry", "top_k": bad})
        result = InputValidateNode()(_make_state(payload))
        assert result["status"] == AgentStatus.ERROR.value
        assert "search_query" not in result
        assert any("top_k" in entry for entry in result["error_log"])

    @pytest.mark.parametrize("bad", ["NaN", "Infinity", "-Infinity", float("nan"), float("inf"), float("-inf")])
    def test_val_06_a_non_finite_number_is_refused(self, bad):
        # These parse as floats and arrive through plain JSON. Every comparison
        # against NaN is False, so an unchecked one changes the answer silently.
        state = _make_state("points expiry")
        state["input_context"] = {"score_threshold": bad}
        result = InputValidateNode()(state)
        assert result["status"] == AgentStatus.ERROR.value
        assert any("score_threshold" in entry for entry in result["error_log"])

    def test_a_rejected_value_is_never_echoed_back(self):
        secret = "9998887776665555"
        state = _make_state("points expiry")
        state["input_context"] = {"top_k": secret}
        result = InputValidateNode()(state)
        assert result["status"] == AgentStatus.ERROR.value
        assert secret not in " ".join(result["error_log"])

    def test_a_valid_override_survives_both_channels(self):
        payload = json.dumps({"query": "points expiry", "top_k": 9})
        state = _make_state(payload)
        state["input_context"] = {"score_threshold": 0.4}
        filters = from_json(InputValidateNode()(state)["query_filters"])
        assert filters["top_k"] == 9
        assert filters["score_threshold"] == 0.4

    def test_input_context_wins_over_the_in_question_envelope(self):
        payload = json.dumps({"query": "points expiry", "top_k": 3})
        state = _make_state(payload)
        state["input_context"] = {"top_k": 8}
        assert from_json(InputValidateNode()(state)["query_filters"])["top_k"] == 8

    def test_an_unknown_category_is_refused_not_silently_applied(self):
        # Filtering on a category that matches nothing discards every passage
        # and reports "no coverage" for a question the corpus does answer.
        payload = json.dumps({"query": "points expiry", "category": "offer"})
        result = InputValidateNode()(_make_state(payload))
        assert result["status"] == AgentStatus.ERROR.value
        assert any("category" in entry for entry in result["error_log"])


class TestSizeAndEmptyGuards:
    def test_val_08_oversize_query_is_truncated(self):
        payload = "redemption " * 300  # ~3300 chars after collapse
        result = InputValidateNode()(_make_state(payload))
        assert len(result["search_query"]) == 2000
        notes = from_json(result.get("intake_notes"), [])
        assert any("truncated" in n for n in notes)

    def test_val_09_empty_request_yields_note_not_error(self):
        result = InputValidateNode()(_make_state(""))
        assert result["search_query"] == ""
        notes = from_json(result.get("intake_notes"), [])
        assert any("empty request" in n for n in notes)

    def test_missing_validated_input_falls_back_to_user_input(self):
        # Contract documented on the node: reads validated_input OR user_input
        # (unit tests that don't run PreProcessNode first still work).
        state = _make_state("")
        del state["validated_input"]
        state["user_input"] = "points expiry policy"
        result = InputValidateNode()(state)
        assert result["search_query"] == "points expiry policy"
