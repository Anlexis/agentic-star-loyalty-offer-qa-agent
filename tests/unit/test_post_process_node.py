# RET-C2-087 — Unit Tests: PostProcessNode (outer post_process slot; S-3 gate)
#
# Invocation canon: node(state) via BaseNode.__call__. PostProcessNode is the
# second outer S-gate slot and requires VERIFIED_EXTERNAL (like PreProcessNode),
# so its behavioural tests build the state at that level; the ANONYMOUS
# rejection lives in test_s1_trust_gate.py.
#
# S-3 layering (two independent scans, verified empirically):
#   1. The node's own module-level _security_gate_output() (src/nodes/
#      post_process_node.py) runs INSIDE execute() against a narrow regex
#      (eyJ.../sk-.../Bearer .../db URI/keyword assignment). A hit returns
#      blocked_output_state() + {status: ERROR, error_log: [...]}: every
#      output-bearing field present and emptied, and formatted_output set to
#      the fixed BLOCKED_OUTPUT_NOTICE.
#   2. The framework FunctionNode default S-3 scan (shared.security.
#      credential_detector.detect_credentials — openai_key / jwt / aws_key /
#      bearer_token / conn_string) then runs on execute()'s RETURNED dict.
#      A hit the node's own regex missed (e.g. AKIA...) raises RuntimeError
#      inside the framework gate; BaseNode.__call__ catches it and returns a
#      bare 4-key error shape (status/error_log/node_history/execution_time),
#      DISCARDING this node's delta and therefore its clearing too.
#
# Why the clearing has to be explicit: LangGraph merges a node's partial dict
# into state, so a key merely omitted keeps its previous value; and
# AgentBaseGraph.get_output() resolves `formatted_output or result` with no
# status check, so an absent or empty formatted_output selects the ungated
# answer the gate just refused. `assert "formatted_output" not in result`
# asserted precisely that defect and is not used here.
#
# Path 2 cannot be contained inside this node (the delta is thrown away). The
# envelope-level status check in LoyaltyOfferQAAgent.get_output() is what
# covers it; tests/integration/test_output_containment_e2e.py drives both
# through the full graph.
#
# Mirrors docs/03_test_spec.md §2.7 (POST-01..POST-08).
# Deterministic — no LLM, no network. framework.* / src.* imports only.

from unittest.mock import MagicMock

import pytest

from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel

import src.nodes.post_process_node
from src.graph.graph import LoyaltyOfferQAGraphNode
from src.nodes.post_process_node import (
    BLOCKED_OUTPUT_NOTICE,
    _OUTPUT_BEARING_FIELDS,
    PostProcessNode,
    _security_gate_output,
)

_CLEAN_ANSWER = "a clean loyalty program answer about points expiry"

# Credential-shaped strings built at runtime so no credential-shaped literal
# ever sits in the repository — the repo-wide credential scan would flag one,
# and rightly so.
_FAKE_JWT = "eyJ" + "a" * 12 + "." + "b" * 12 + "." + "c" * 12
_FAKE_OPENAI_KEY = "sk-" + "A" * 25
_FAKE_BEARER = "Bearer " + "a" * 20
_FAKE_AWS_KEY = "AKIA" + "B" * 16
_FAKE_CONN_STRING = "postgresql://user:" + "pass" + "@host:5432/db"


def _make_state(result_text, **extra) -> dict:
    state = {
        "result": result_text,
        "caller_trust_level": TrustLevel.VERIFIED_EXTERNAL.value,
        "node_history": [],
        "error_log": [],
        "session_id": "unit-session",
        "execution_time": {},
    }
    state.update(extra)
    return state


class TestPostProcessClean:
    def test_post_01_clean_output_passes_through(self):
        result = PostProcessNode()(_make_state(_CLEAN_ANSWER))
        assert result["status"] == AgentStatus.SUCCESS.value
        # review finding 15 regression guard: State carries the plain string, never the enum.
        assert type(result["status"]) is str  # noqa: E721 - the exact type is the assertion
        assert result["formatted_output"] == _CLEAN_ANSWER

    def test_post_02_empty_result_is_non_fatal(self):
        result = PostProcessNode()(_make_state(""))
        assert result["status"] == AgentStatus.SUCCESS.value
        assert result["formatted_output"] == ""


def _make_answer_state(result_text, **extra) -> dict:
    """A state where every output-bearing field carries the answer, as the
    outer graph builds it via LoyaltyOfferQAGraphNode.merge_output()."""
    return _make_state(
        result_text,
        loyalty_answer=result_text,
        citations='[{"ref": 1, "id": "kb-001", "title": "t", "source": "s"}]',
        answer_category="program_rules",
        grounding_excerpts=f'["{result_text}"]',
        **extra,
    )


class TestPostProcessDomainGate:
    """POST-03..05: caught by the node's OWN _security_gate_output() —
    execute() returns status=ERROR plus a cleared, content-free delta."""

    def _assert_blocked_by_domain_gate(self, result, secret):
        assert result["status"] == AgentStatus.ERROR.value
        assert any("credential-like" in str(e) for e in result["error_log"])
        assert secret not in str(result.get("error_log", ""))
        # The blocked answer is CLEARED, not merely unwritten. LangGraph merges
        # partial deltas, so a key omitted from the returned dict keeps its old
        # value in state — the key must be present AND empty.
        for field in _OUTPUT_BEARING_FIELDS:
            assert field in result, f"{field} omitted from the block delta — the old value survives the merge"
            assert not result[field], f"{field} still carries content on a blocked answer"
        # Truthy on purpose: AgentBaseGraph.get_output() resolves
        # `formatted_output or result`, so an absent or empty value re-selects
        # the ungated answer this gate just refused.
        assert result["formatted_output"] == BLOCKED_OUTPUT_NOTICE
        assert result["formatted_output"], "a falsy placeholder re-opens the base-envelope fallback"
        assert secret not in str(result)

    def test_post_03_jwt_is_blocked(self):
        result = PostProcessNode()(_make_answer_state(f"session token {_FAKE_JWT}"))
        self._assert_blocked_by_domain_gate(result, _FAKE_JWT)

    def test_post_04_openai_key_is_blocked(self):
        result = PostProcessNode()(_make_answer_state(f"api key {_FAKE_OPENAI_KEY}"))
        self._assert_blocked_by_domain_gate(result, _FAKE_OPENAI_KEY)

    def test_post_05_bearer_token_is_blocked(self):
        result = PostProcessNode()(_make_answer_state(f"authorization: {_FAKE_BEARER}"))
        self._assert_blocked_by_domain_gate(result, _FAKE_BEARER)

    def test_post_05b_password_assignment_is_blocked(self):
        # 2026-07-30 fix: `password=<value>` matched NEITHER scan layer (docs/03
        # §2.7 formerly recorded it as a known gap). _SECRET_ASSIGNMENT_RE now
        # catches keyword-assignment shapes in the domain gate; this pins it.
        result = PostProcessNode()(_make_answer_state("reset with password=hunter2hunter2"))
        self._assert_blocked_by_domain_gate(result, "hunter2hunter2")

    def test_post_05c_ungrounded_figure_block_clears_the_same_fields(self):
        """The fidelity layer blocks a different way; containment is identical."""
        state = _make_answer_state("the redemption minimum is 777777 points")
        state["grounding_excerpts"] = '["the redemption minimum is 500 points"]'
        result = PostProcessNode()(state)
        assert result["status"] == AgentStatus.ERROR.value
        assert any("ungrounded figure" in str(e) for e in result["error_log"])
        assert "777777" not in str(result), "the ungrounded figure must not be echoed back"
        for field in _OUTPUT_BEARING_FIELDS:
            assert field in result and not result[field]
        assert result["formatted_output"] == BLOCKED_OUTPUT_NOTICE

    def test_post_05d_cleared_set_covers_every_field_the_merge_writes(self):
        """Inventory guard: a new caller-facing field added to the outer merge
        must be added to the cleared set, or this fails."""
        merged = LoyaltyOfferQAGraphNode().merge_output({}, {})
        assert set(merged) - {"status"} == set(_OUTPUT_BEARING_FIELDS)


class TestPostProcessFrameworkGate:
    """POST-06..07: shapes the framework's own detect_credentials() knows.

    The node's scan is the UNION of that list and the shapes it misses, so
    these are caught INSIDE execute() and get the cleared delta. That ordering
    is the point: a shape only the framework catches raises inside the @final
    S-3 gate, BaseNode.__call__ discards the node's whole delta — the clearing
    with it — and only the envelope-level status check in
    LoyaltyOfferQAAgent.get_output() would still contain it. Keeping the local
    set wider than the framework's keeps the state contained too."""

    def test_post_06_aws_access_key_is_blocked(self):
        result = PostProcessNode()(_make_answer_state(f"access key {_FAKE_AWS_KEY}"))
        assert result["status"] == AgentStatus.ERROR.value
        assert any("credential-like" in str(e) for e in result["error_log"])
        assert _FAKE_AWS_KEY not in str(result)
        for field in _OUTPUT_BEARING_FIELDS:
            assert field in result and not result[field]
        assert result["formatted_output"] == BLOCKED_OUTPUT_NOTICE

    def test_post_07_connection_string_is_blocked_by_this_template(self):
        # Owned here rather than left to the surrounding framework: where that
        # scan is absent the payload would otherwise reach the caller.
        result = PostProcessNode()(_make_answer_state(f"db at {_FAKE_CONN_STRING}"))
        assert result["status"] == AgentStatus.ERROR.value
        assert any("credential-like" in str(e) for e in result["error_log"])
        assert _FAKE_CONN_STRING not in str(result)
        for field in _OUTPUT_BEARING_FIELDS:
            assert field in result and not result[field]
        assert result["formatted_output"] == BLOCKED_OUTPUT_NOTICE

    @pytest.mark.parametrize(
        "sample",
        [
            "sk_live_" + "A" * 20,  # stripe_key
            "sk-" + "A" * 25,  # openai_key
            "eyJ" + "a" * 12 + "." + "b" * 12,  # jwt
            "AKIA" + "B" * 16,  # aws_key
            "Bearer " + "a" * 20,  # bearer_token
            "postgresql://localhost:5432/loyalty",  # conn_string, no inline creds
            "mongodb://replica-set-0.internal/loyalty",  # conn_string, no inline creds
        ],
    )
    def test_the_local_scan_is_not_narrower_than_the_frameworks(self, sample):
        """One sample per framework credential type.

        A shape only the framework catches is a bypass of this node's
        containment, not a second line of defence: the framework raises and the
        clearing is thrown away with the delta. Samples are enumerated by hand —
        a credential type added to the framework will not appear here on its
        own, so re-derive this list when the wheel version moves.
        """
        assert _security_gate_output(sample), f"{sample[:12]}... passes this scan but not the framework's"


class TestPostProcessAudit:
    def test_post_08_domain_audit_payload_on_success(self, monkeypatch):
        spy = MagicMock()
        monkeypatch.setattr(src.nodes.post_process_node, "emit_trace_event", spy)
        PostProcessNode()(_make_state(_CLEAN_ANSWER))
        events = [call.args[0] for call in spy.call_args_list]
        assert "post_process_complete" in events
        payload = spy.call_args_list[events.index("post_process_complete")].args[1]
        assert payload == {"has_output": True}

    def test_gate_blocked_audit_event_on_domain_violation(self, monkeypatch):
        spy = MagicMock()
        monkeypatch.setattr(src.nodes.post_process_node, "emit_trace_event", spy)
        PostProcessNode()(_make_state(f"leak {_FAKE_JWT}"))
        events = [call.args[0] for call in spy.call_args_list]
        assert "post_process_credential_blocked" in events
        payload = spy.call_args_list[events.index("post_process_credential_blocked")].args[1]
        assert payload["violation_count"] >= 1
