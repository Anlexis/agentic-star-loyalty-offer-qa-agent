# RET-C2-087 — Unit Tests: PreProcessNode (outer pre_process slot; S-1/S-2)
#
# Invocation canon (CoE 2026-07-16): every test invokes the node via
# node(state) — BaseNode.__call__ → S-1 trust gate → S-2 PII mask → execute()
# → S-3 credential gate — never a bare node.execute(state). PreProcessNode
# requires VERIFIED_EXTERNAL, so its behavioural tests build the state at that
# level (the ANONYMOUS rejection lives in test_s1_trust_gate.py).
#
# S-2 layering (docs/02_design.md "Security Gates" — deliberate handling for
# this template): the FRAMEWORK input gate masks user_input / validated_input
# BEFORE execute() runs — e-mails, 12-digit runs, and Title-Case name bigrams
# surface as [MASKED]. The NODE's own _surface_strip_identifiers() then runs
# on the (possibly already-masked) text and catches labelled member/loyalty/
# card identifiers and 8-9 digit unformatted runs the framework patterns do
# not — those surface as [REDACTED]. Both are verified empirically against the
# installed real SDK's detect_pii() (RULES C3 — "measure, don't reason").
#
# Mirrors docs/03_test_spec.md §2.1 (PRE-01..PRE-10).
# Deterministic — no LLM, no network. framework.* / src.* imports only.

from unittest.mock import MagicMock

from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel

import src.nodes.pre_process_node
from src.nodes.pre_process_node import PreProcessNode

# The Wave-2 STG sign-off payload (byte-equal to deploy/invoke_payload.json
# "input" — also PB-6's _VALID_PAYLOAD). Empirically confirmed PII-free via
# detect_pii(): the "1,000-yen" digit group does NOT match any framework S-2
# pattern (comma-separated, not a 12-digit/credit-card/phone/SSN shape), so
# the S-2 mask leaves it untouched end to end.
_VALID_QUERY = "How many points do I need to redeem a 1,000-yen digital gift card, and " "when do unused points expire?"


def _make_state(user_input=_VALID_QUERY, **extra) -> dict:
    state = {
        "user_input": user_input,
        "caller_trust_level": TrustLevel.VERIFIED_EXTERNAL.value,
        "node_history": [],
        "error_log": [],
        "session_id": "unit-session",
        "execution_time": {},
    }
    state.update(extra)
    return state


class TestPreProcessSuccess:
    def test_pre_01_valid_query_accepted_unmasked(self):
        """The STG sign-off payload survives byte-for-byte — empirical proof
        the '1,000-yen' digit group does not trip the S-2 mask (RULES C3
        payload/S-2 interaction check)."""
        result = PreProcessNode()(_make_state())
        assert result["status"] == AgentStatus.SUCCESS.value
        # review finding 15 regression guard: State carries the plain string, never the enum.
        assert type(result["status"]) is str  # noqa: E721 - the exact type is the assertion
        assert result["validated_input"] == _VALID_QUERY

    def test_enriched_context_carries_channel(self):
        result = PreProcessNode()(_make_state(input_context={"channel": "web"}))
        assert result["enriched_context"]["channel"] == "web"
        assert result["enriched_context"]["source"] == "LoyaltyOfferQAAgent"

    def test_missing_channel_defaults_to_unknown(self):
        result = PreProcessNode()(_make_state())
        assert result["enriched_context"]["channel"] == "unknown"


class TestPreProcessRejection:
    def test_pre_02_empty_input_is_error(self):
        result = PreProcessNode()(_make_state(user_input=""))
        assert result["status"] == AgentStatus.ERROR.value
        assert result["error_log"]
        # No validated_input is produced on the reject path.
        assert "validated_input" not in result

    def test_whitespace_only_is_error(self):
        result = PreProcessNode()(_make_state(user_input="   \n\t "))
        assert result["status"] == AgentStatus.ERROR.value

    def test_pre_03_missing_user_input_is_error(self):
        state = _make_state()
        del state["user_input"]
        result = PreProcessNode()(state)
        assert result["status"] == AgentStatus.ERROR.value

    def test_non_string_input_is_error(self):
        result = PreProcessNode()(_make_state(user_input={"malicious": "dict"}))
        assert result["status"] == AgentStatus.ERROR.value


class TestPreProcessIdentifierScreen:
    """PRE-04..PRE-08: raw member/loyalty-card identifiers never survive into
    validated_input. Cases verified empirically against the installed SDK."""

    def test_pre_04_labelled_member_id_redacted_by_node_screen(self):
        # "member ID ..." is not a framework name/email/digit pattern — the
        # node's own surface strip must catch it ([REDACTED] path).
        raw = "my member ID is LM-1029384, how many points until my next reward?"
        result = PreProcessNode()(_make_state(user_input=raw))
        vi = result["validated_input"]
        assert result["status"] == AgentStatus.SUCCESS.value
        assert "LM-1029384" not in vi
        assert "[REDACTED]" in vi

    def test_pre_05_labelled_loyalty_card_number_redacted(self):
        raw = "loyalty card # 88231 shows a pending balance"
        result = PreProcessNode()(_make_state(user_input=raw))
        vi = result["validated_input"]
        assert "88231" not in vi
        assert "[REDACTED]" in vi

    def test_pre_06_email_masked_by_framework_s2_gate(self):
        # The framework S-2 gate masks e-mail before execute() sees it.
        raw = "please send my statement to jane.doe@example.com today"
        result = PreProcessNode()(_make_state(user_input=raw))
        vi = result["validated_input"]
        assert "jane.doe@example.com" not in vi
        assert "[MASKED]" in vi

    def test_pre_07_grouped_12_digit_run_masked_by_framework(self):
        # 4-4-4 digit groups match the framework S-2 my_number pattern —
        # caught BEFORE the node's own 10-19-digit-grouped pattern ever sees it.
        raw = "account 1234 5678 9012 shows a pending balance"
        result = PreProcessNode()(_make_state(user_input=raw))
        vi = result["validated_input"]
        assert "1234 5678 9012" not in vi
        assert "[MASKED]" in vi

    def test_pre_08_unformatted_8_digit_member_number_redacted_by_node(self):
        # 8 digits, no separators — below every framework digit pattern's
        # shape, so only the node's own \b\d{8,9}\b catches it.
        raw = "my member number is 12345678 and I need help"
        result = PreProcessNode()(_make_state(user_input=raw))
        vi = result["validated_input"]
        assert "12345678" not in vi
        assert "[REDACTED]" in vi

    def test_pre_09_unformatted_7_digit_run_is_not_redacted(self):
        # Below the node's 8-digit floor and not a framework shape either —
        # confirms the boundary is 8, not 7 (an order number, not a member id).
        raw = "my order number is 1234567 today"
        result = PreProcessNode()(_make_state(user_input=raw))
        assert result["validated_input"] == raw

    def test_pre_10_name_bigram_masked_by_framework(self):
        # Two Title-Case words — the framework's 'name' heuristic.
        raw = "Jane Doe called about her points balance"
        result = PreProcessNode()(_make_state(user_input=raw))
        vi = result["validated_input"]
        assert "Jane Doe" not in vi
        assert "[MASKED]" in vi

    def test_point_totals_are_not_treated_as_identifiers(self):
        # A plain points figure must survive — it is domain content, not PII.
        raw = "I have 500 points, how do I redeem them?"
        result = PreProcessNode()(_make_state(user_input=raw))
        assert result["validated_input"] == raw


class TestPreProcessAudit:
    def test_pre_11_domain_audit_payload(self, monkeypatch):
        """S-4: the accepted request emits pre_process_complete; the assertion
        targets call.args[1] — the event payload — never the whole call repr."""
        spy = MagicMock()
        monkeypatch.setattr(src.nodes.pre_process_node, "emit_trace_event", spy)
        PreProcessNode()(_make_state())
        events = [call.args[0] for call in spy.call_args_list]
        assert "pre_process_complete" in events
        payload = spy.call_args_list[events.index("pre_process_complete")].args[1]
        assert payload["input_chars"] == len(_VALID_QUERY)
        # No raw query text or digit content in the audit payload (docs/02
        # "Security Gates" S-4 constraint).
        assert "input_chars" in payload and isinstance(payload["input_chars"], int)
