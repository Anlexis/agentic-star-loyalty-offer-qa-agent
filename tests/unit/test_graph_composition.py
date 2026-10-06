# RET-C2-087 — Unit Tests: nested Cat-2 graph composition (outer + end-to-end)
#
# Drives the REAL outer agent (LoyaltyOfferQAAgent / Graph) end-to-end via
# AgentBaseGraph.invoke(). The e2e context is
# InvocationContext(caller_trust_level=TrustLevel.VERIFIED_EXTERNAL) — the
# manifest's declared caller level; for_internal() is NEVER used (it would
# over-privilege the run and hide S-1 regressions).
#
# Mirrors docs/03_test_spec.md §3 (INT-05..INT-13).
# Deterministic — no LLM, no network. framework.* / src.* imports only.

import pathlib

from framework.graph.agent_base_graph import AgentBaseGraph
from framework.schemas.agent_status import AgentStatus
from framework.schemas.invocation_context import InvocationContext
from framework.schemas.trust_level import TrustLevel

from src.graph.domain_workflow_graph import DomainWorkflowGraph
from src.graph.graph import Graph, LoyaltyOfferQAAgent, LoyaltyOfferQAGraphNode
from src.nodes.post_process_node import PostProcessNode
from src.nodes.pre_process_node import PreProcessNode
from src.schemas.state import State, from_json, to_json

# The Wave-2 STG sign-off payload — byte-equal to deploy/invoke_payload.json
# "input" (also PB-6's _VALID_PAYLOAD). Empirically PII-free: the S-2 mask
# never touches it end to end (see test_pre_process_node.py PRE-01).
_VALID_PAYLOAD = (
    "How many points do I need to redeem a 1,000-yen digital gift card, and " "when do unused points expire?"
)


def _run(user_input: str, trust: TrustLevel = TrustLevel.VERIFIED_EXTERNAL) -> dict:
    ctx = InvocationContext(caller_trust_level=trust, caller_id="unit-suite")
    return Graph().invoke(user_input, ctx=ctx)


class TestOuterGraphConstruction:
    def test_int_05_inherits_agent_base_graph_directly(self):
        assert issubclass(LoyaltyOfferQAAgent, AgentBaseGraph)

    def test_int_05_graph_alias(self):
        assert Graph is LoyaltyOfferQAAgent

    def test_state_schema_is_state(self):
        assert LoyaltyOfferQAAgent().state_schema is State

    def test_int_06_compile_fills_all_backbone_slots(self):
        agent = LoyaltyOfferQAAgent()
        agent.compile()
        for slot in ("initialize", "pre_process", "main", "post_process", "finalize"):
            assert agent._nodes.get(slot) is not None, f"backbone slot not filled: {slot}"
        assert isinstance(agent._nodes["pre_process"], PreProcessNode)
        assert isinstance(agent._nodes["main"], LoyaltyOfferQAGraphNode)
        assert isinstance(agent._nodes["post_process"], PostProcessNode)

    def test_add_edges_is_not_overridden(self):
        # Backbone wiring belongs to the framework — the template must not
        # redefine it.
        assert "add_edges" not in LoyaltyOfferQAAgent.__dict__


class TestMainSlotGraphNode:
    def test_int_07_get_subgraph_returns_the_inner_graph(self):
        subgraph = LoyaltyOfferQAGraphNode().get_subgraph()
        assert isinstance(subgraph, DomainWorkflowGraph)
        assert subgraph.config["configurable"]["retrieval"], "inner config must carry the retrieval block"

    def test_int_08_extract_input_prefers_validated_input(self):
        node = LoyaltyOfferQAGraphNode()
        assert node.extract_input({"validated_input": "VI", "user_input": "UI"}) == "VI"
        assert node.extract_input({"user_input": "UI"}) == "UI"

    def test_int_09_merge_output_maps_the_inner_contract(self):
        node = LoyaltyOfferQAGraphNode()
        citations = to_json([{"ref": 1, "id": "kb-011", "title": "t", "source": "s"}])
        delta = node.merge_output(
            {},
            {
                "formatted_answer": "ANSWER",
                "citations": citations,
                "answer_category": "redemption_catalog",
                "status": AgentStatus.SUCCESS.value,
            },
        )
        # The inner formatted_answer surfaces as BOTH loyalty_answer and
        # result (the output gate reads state["result"]).
        assert delta == {
            "loyalty_answer": "ANSWER",
            "result": "ANSWER",
            "citations": citations,
            "answer_category": "redemption_catalog",
            "grounding_excerpts": None,
            "status": AgentStatus.SUCCESS.value,
        }

    def test_error_strategy_is_propagate_and_hitl_is_contained(self):
        assert LoyaltyOfferQAGraphNode.error_strategy == "propagate"
        assert LoyaltyOfferQAGraphNode.propagate_hitl is False

    def test_int_10_parent_config_never_empty_without_the_runtime_file(self, monkeypatch):
        # Even when config/config.yaml cannot be read the forwarded config
        # carries the fallback retrieval block — never {}.
        import src.graph.runtime_config as runtime_config

        monkeypatch.setattr(runtime_config, "RUNTIME_CONFIG_PATH", pathlib.Path("/nonexistent/config.yaml"))
        cfg = LoyaltyOfferQAGraphNode()._parent_config()
        assert cfg["configurable"]["retrieval"]["kb_path"] == "config/kb/loyalty_kb.json"
        assert cfg["configurable"]["retrieval"]["top_k"] == 4


class TestEndToEndInvoke:
    """Full agent run: outer backbone + inner domain workflow, no LLM."""

    def test_int_11_invoke_returns_success(self):
        result = _run(_VALID_PAYLOAD)
        assert (
            result.get("status") == AgentStatus.SUCCESS.value
        ), f"Expected success, got {result.get('status')}. result={result!r}"

    def test_int_11_output_is_the_gated_formatted_answer(self):
        output = _run(_VALID_PAYLOAD).get("output")
        assert isinstance(output, str) and output.strip()
        assert output.startswith("# Loyalty Program Q&A Result")
        assert "[1]" in output
        assert "does not reflect your live points balance or account history" in output

    def test_int_11_e2e_traverses_the_post_process_gate(self):
        history = _run(_VALID_PAYLOAD).get("node_history", [])
        for cls_name in ("PreProcessNode", "LoyaltyOfferQAGraphNode", "PostProcessNode"):
            assert cls_name in history, f"node_history missing {cls_name}: {history}"

    def test_int_12_structured_fields_surfaced_on_success(self):
        result = _run(_VALID_PAYLOAD)
        assert result.get("answer_category") == "redemption_catalog"
        citations = result.get("citations")
        assert citations and citations[0]["id"] == "kb-011"

    def test_no_coverage_query_still_terminates_success(self):
        result = _run("quantum telepathy sandwich recipes")
        assert result.get("status") == AgentStatus.SUCCESS.value
        assert "does not contain sufficient coverage" in result.get("output", "")
        assert result.get("answer_category") is None

    def test_int_13_anonymous_caller_is_denied_at_the_outer_boundary(self):
        """S-1 at graph level: an ANONYMOUS invoke is refused by the
        VERIFIED_EXTERNAL pre_process slot. The error state short-circuits the
        main slot — GraphNode's own S-2 gate (a deliberate no-op passthrough)
        sees the incoming status is already ERROR and __call__'s generic
        "input gate rejected" branch skips execute() entirely, so the inner
        subgraph is NEVER invoked (verified by instrumented trace, not
        assumed) — and routes past post_process to finalize. No domain answer
        is ever produced."""
        result = _run(_VALID_PAYLOAD, trust=TrustLevel.ANONYMOUS)
        assert result.get("status") == AgentStatus.ERROR.value
        assert not result.get("output")
        assert result.get("citations") is None
        assert result.get("answer_category") is None
        history = result.get("node_history", [])
        assert "PostProcessNode" not in history
        assert history[:2] == ["InitializeNode", "PreProcessNode"]


class TestStateRoundTrip:
    """ADR-005 helpers: producers to_json() on write, consumers from_json()."""

    def test_to_from_json_list_round_trip(self):
        original = [{"id": "kb-011", "score": 0.41, "title": "minimum redemption threshold"}]
        assert from_json(to_json(original)) == original

    def test_to_from_json_dict_round_trip(self):
        original = {"category": "redemption_catalog", "top_k": 3}
        assert from_json(to_json(original)) == original

    def test_to_json_none_passes_through(self):
        assert to_json(None) is None

    def test_from_json_malformed_returns_default(self):
        assert from_json("{not valid json", default=[]) == []
        assert from_json(None, default={}) == {}
        assert from_json("", default=[]) == []
