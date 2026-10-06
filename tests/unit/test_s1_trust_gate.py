# RET-C2-087 — Unit Tests: S-1 trust gate (Wave-1 seed suite, extended in Wave-2)
#
# Replaces the scaffold test_main_node.py (MainNode was retired by the Wave-1
# Cat-2 nested implementation — the main slot is now LoyaltyOfferQAGraphNode
# in src/graph/graph.py). Wave-2 EXTENDS this seed suite (C9): the outer
# post_process gate, runtime ANONYMOUS-allowed coverage for all 5 inner
# domain nodes (Wave-1 only exercised InputValidateNode), and the template's
# full trust matrix.
#
# S-1 contract (CoE 2026-07-16): tests must invoke nodes via node(state) —
# through BaseNode.__call__, which runs S-1 trust -> S-2 PII mask -> execute()
# -> S-3 — never via node.execute(state) directly, which bypasses the gate.
# A denial RETURNS an error dict (never raises) with status ERROR and
# "trust gate denied" in error_log; execute() never runs, so execute-only
# output keys are ABSENT from the returned dict.

from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel

from src.nodes.generate_answer_node import GenerateAnswerNode
from src.nodes.input_validate_node import InputValidateNode
from src.nodes.output_format_node import OutputFormatNode
from src.nodes.post_process_node import PostProcessNode
from src.nodes.pre_process_node import PreProcessNode
from src.nodes.rerank_filter_node import RerankFilterNode
from src.nodes.retrieve_node import RetrieveNode
from src.schemas.state import to_json


def _make_state(
    trust_value: str, user_input: str = "How many points do I need to redeem a gift card?", **extra
) -> dict:
    state = {
        "user_input": user_input,
        "input_context": {},
        "caller_trust_level": trust_value,
        "correlation_id": "s1-gate-test",
        "node_history": [],
        "error_log": [],
        "session_id": "test-session",
        "execution_time": {},
    }
    state.update(extra)
    return state


class TestS1TrustGate:
    """S-1 trust gate tests — all invocations go through node(state) / __call__."""

    def test_anonymous_caller_allowed_on_inner_node(self):
        """S-1: an ANONYMOUS caller passes an ANONYMOUS inner domain node."""
        node = InputValidateNode()  # required_trust_level = ANONYMOUS
        result = node(_make_state(TrustLevel.ANONYMOUS.value))
        assert "trust gate denied" not in str(result.get("error_log", []))
        assert result.get("search_query"), "inner node should produce a normalised query"

    def test_anonymous_caller_denied_on_pre_process(self):
        """S-1 rejection: ANONYMOUS caller on the VERIFIED_EXTERNAL PreProcessNode.

        __call__ must RETURN an error dict (never raise) with status ERROR and
        'trust gate denied' in the error_log. execute() never ran, so the
        execute-only output key (validated_input) must be ABSENT.
        """
        node = PreProcessNode()  # required_trust_level = VERIFIED_EXTERNAL
        result = node(_make_state(TrustLevel.ANONYMOUS.value))
        assert result.get("status") == AgentStatus.ERROR.value
        error_log = result.get("error_log", [])
        assert any(
            "trust gate denied" in str(e) for e in error_log
        ), f"Expected 'trust gate denied' in error_log, got: {error_log}"
        assert "validated_input" not in result, "execute() must not run on an S-1 denial — validated_input leaked"

    def test_verified_external_caller_passes_pre_process(self):
        """S-1: a VERIFIED_EXTERNAL caller clears the pre_process gate and
        the node writes the identifier-stripped validated_input."""
        node = PreProcessNode()
        result = node(_make_state(TrustLevel.VERIFIED_EXTERNAL.value))
        assert result.get("status") == AgentStatus.SUCCESS.value
        assert result.get("validated_input")

    def test_pre_process_empty_input_rejected_after_gate(self):
        """S-1 passes, then the node's own input validation rejects empty input."""
        node = PreProcessNode()
        result = node(_make_state(TrustLevel.VERIFIED_EXTERNAL.value, user_input=""))
        assert result.get("status") == AgentStatus.ERROR.value
        assert any("empty" in str(e) for e in result.get("error_log", []))

    def test_pre_process_redacts_labelled_member_id(self):
        """S-2: a labelled member/card identifier is redacted before validated_input
        is written (docs/02_design.md 'Security Gates' — deliberate S-2 handling)."""
        node = PreProcessNode()
        result = node(
            _make_state(
                TrustLevel.VERIFIED_EXTERNAL.value,
                user_input="my member ID is LM-1029384, how many points until my next reward?",
            )
        )
        assert result.get("status") == AgentStatus.SUCCESS.value
        assert "LM-1029384" not in result.get("validated_input", "")
        assert "[REDACTED]" in result.get("validated_input", "")

    def test_anonymous_caller_denied_on_post_process(self):
        """S-1 rejection on the other VERIFIED_EXTERNAL outer slot (post_process).

        The denial dict carries no execute-only key (formatted_output ABSENT).
        """
        node = PostProcessNode()  # required_trust_level = VERIFIED_EXTERNAL
        result = node(
            _make_state(
                TrustLevel.ANONYMOUS.value,
                result="a clean loyalty program answer about points expiry",
            )
        )
        assert result.get("status") == AgentStatus.ERROR.value
        assert any("trust gate denied" in str(e) for e in result.get("error_log", []))
        assert "formatted_output" not in result, "execute() must not run on an S-1 denial — formatted_output leaked"

    def test_verified_external_caller_passes_post_process(self):
        """S-1: a VERIFIED_EXTERNAL caller clears the post_process gate."""
        node = PostProcessNode()
        result = node(
            _make_state(
                TrustLevel.VERIFIED_EXTERNAL.value,
                result="a clean loyalty program answer about points expiry",
            )
        )
        assert result.get("status") == AgentStatus.SUCCESS.value
        assert result.get("formatted_output")


class TestS1InnerNodesAdmitAnonymous:
    """Wave-2 extension (C9): Wave-1 only runtime-exercised InputValidateNode
    at ANONYMOUS; this class covers the remaining four inner domain nodes so
    every S-1-admitting node in the template has a runtime (not just
    class-attribute) check. Behavioural detail for each node lives in its own
    tests/unit/test_<node>.py file — these assertions are S-1-focused only."""

    def test_anonymous_caller_allowed_on_retrieve(self):
        node = RetrieveNode()  # required_trust_level = ANONYMOUS
        state = {
            "search_query": "points expiry",
            "caller_trust_level": TrustLevel.ANONYMOUS.value,
            "node_history": [],
            "error_log": [],
            "session_id": "s1-gate-test",
            "execution_time": {},
        }
        result = node(state)
        assert "trust gate denied" not in str(result.get("error_log", []))
        assert "retrieved_documents" in result

    def test_anonymous_caller_allowed_on_rerank_filter(self):
        node = RerankFilterNode()  # required_trust_level = ANONYMOUS
        state = {
            "retrieved_documents": to_json([]),
            "caller_trust_level": TrustLevel.ANONYMOUS.value,
            "node_history": [],
            "error_log": [],
            "session_id": "s1-gate-test",
            "execution_time": {},
        }
        result = node(state)
        assert "trust gate denied" not in str(result.get("error_log", []))
        assert "ranked_documents" in result

    def test_anonymous_caller_allowed_on_generate_answer(self):
        node = GenerateAnswerNode()  # required_trust_level = ANONYMOUS
        state = {
            "ranked_documents": to_json([]),
            "caller_trust_level": TrustLevel.ANONYMOUS.value,
            "node_history": [],
            "error_log": [],
            "session_id": "s1-gate-test",
            "execution_time": {},
        }
        result = node(state)
        assert "trust gate denied" not in str(result.get("error_log", []))
        assert result.get("grounded_answer")

    def test_anonymous_caller_allowed_on_output_format(self):
        node = OutputFormatNode()  # required_trust_level = ANONYMOUS
        state = {
            "grounded_answer": "an answer body.",
            "citations": to_json([]),
            "caller_trust_level": TrustLevel.ANONYMOUS.value,
            "node_history": [],
            "error_log": [],
            "session_id": "s1-gate-test",
            "execution_time": {},
        }
        result = node(state)
        assert "trust gate denied" not in str(result.get("error_log", []))
        assert result.get("status") == AgentStatus.SUCCESS.value


class TestTrustLevelMatrix:
    """The template's declared trust matrix (docs/02_design.md §Security Gates).

    Outer S-gate slots (pre_process, post_process) require VERIFIED_EXTERNAL;
    the five inner domain nodes run behind that outer boundary and are
    declared ANONYMOUS per the Cat-2 nested convention (RULES B2).
    """

    def test_outer_gate_nodes_require_verified_external(self):
        assert PreProcessNode.required_trust_level is TrustLevel.VERIFIED_EXTERNAL
        assert PostProcessNode.required_trust_level is TrustLevel.VERIFIED_EXTERNAL

    def test_inner_domain_nodes_admit_anonymous(self):
        for node_cls in (
            InputValidateNode,
            RetrieveNode,
            RerankFilterNode,
            GenerateAnswerNode,
            OutputFormatNode,
        ):
            assert node_cls.required_trust_level is TrustLevel.ANONYMOUS, (
                f"{node_cls.__name__} must declare TrustLevel.ANONYMOUS " "(inner Cat-2 domain node)"
            )
