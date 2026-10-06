# RET-C2-087 — Unit Tests: DomainWorkflowGraph (inner BaseGraph)
#
# Inner-graph composition + a full inner invoke() over the seeded KB. The
# inner graph runs the 5 domain nodes (all ANONYMOUS) — the S-1 outer boundary
# is the AgentBaseGraph backbone's concern and is covered in
# test_graph_composition.py / the PoB suite.
#
# Mirrors docs/03_test_spec.md §3 (INT-01..INT-04).
# Deterministic — no LLM, no network. framework.* / src.* imports only.

from langgraph.graph import END

from framework.graph.base_graph import BaseGraph
from framework.schemas.agent_status import AgentStatus

from src.graph.domain_workflow_graph import DomainWorkflowGraph
from src.graph.graph import LoyaltyOfferQAGraphNode
from src.nodes.generate_answer_node import GenerateAnswerNode
from src.nodes.input_validate_node import InputValidateNode
from src.nodes.output_format_node import OutputFormatNode
from src.nodes.rerank_filter_node import RerankFilterNode
from src.nodes.retrieve_node import RetrieveNode
from src.schemas.state import State, from_json

_REDEMPTION_QUERY = "minimum points needed to redeem a gift card"


class TestInnerGraphConstruction:
    def test_int_01_inherits_base_graph(self):
        assert issubclass(DomainWorkflowGraph, BaseGraph)

    def test_int_01_registers_the_five_domain_nodes(self):
        inner = DomainWorkflowGraph()
        inner.register_nodes()
        assert set(inner._nodes.keys()) == {
            "input_validate",
            "retrieve",
            "rerank_filter",
            "generate_answer",
            "output_format",
        }
        assert isinstance(inner._nodes["input_validate"], InputValidateNode)
        assert isinstance(inner._nodes["retrieve"], RetrieveNode)
        assert isinstance(inner._nodes["rerank_filter"], RerankFilterNode)
        assert isinstance(inner._nodes["generate_answer"], GenerateAnswerNode)
        assert isinstance(inner._nodes["output_format"], OutputFormatNode)

    def test_inner_graph_name_and_schema(self):
        inner = DomainWorkflowGraph()
        assert inner.name == "ret_c2_087_loyalty_offer_qa_workflow"
        assert inner.state_schema is State

    def test_initialize_finalize_are_not_registered(self):
        # Outer backbone concerns must not leak into the inner topology.
        inner = DomainWorkflowGraph()
        inner.register_nodes()
        assert "initialize" not in inner._nodes
        assert "finalize" not in inner._nodes


class TestConfigForwarding:
    def test_int_02_extra_initial_state_republishes_retrieval_block(self):
        inner = DomainWorkflowGraph(config={"configurable": {"retrieval": {"top_k": 2}}})
        extra = inner._extra_initial_state()
        assert set(extra.keys()) == {"retrieval_config", "input_context"}
        # Structured state fields travel as JSON strings, never bare containers.
        assert isinstance(extra["retrieval_config"], str)
        seeded = from_json(extra["retrieval_config"])
        assert seeded["top_k"] == 2
        # The block is completed from the declared defaults, so the nodes always
        # receive a full, validated set rather than a fragment.
        assert seeded["score_threshold"] == 0.25
        assert seeded["kb_path"] == "config/kb/loyalty_kb.json"

    def test_extra_initial_state_with_no_config_falls_back_to_the_runtime_file(self):
        seeded = from_json(DomainWorkflowGraph()._extra_initial_state()["retrieval_config"])
        assert seeded["top_k"] == 4
        assert seeded["score_threshold"] == 0.25

    def test_a_non_finite_declared_threshold_stops_compilation(self):
        # A clamp would turn NaN into the strictest possible floor and refuse
        # every covered question. Compilation fails instead.
        import pytest
        from framework.errors import ConfigError

        inner = DomainWorkflowGraph(config={"configurable": {"retrieval": {"score_threshold": float("nan")}}})
        with pytest.raises(ConfigError):
            inner._validate_config()


class TestOutputShape:
    def test_int_03_get_output_shapes_the_merge_contract(self):
        inner = DomainWorkflowGraph()
        out = inner.get_output(
            {
                "formatted_answer": "ANSWER",
                "citations": "[]",
                "answer_category": "redemption_catalog",
                "status": AgentStatus.SUCCESS.value,
                "node_history": ["InputValidateNode"],
            }
        )
        assert out["formatted_answer"] == "ANSWER"
        assert out["citations"] == "[]"
        assert out["answer_category"] == "redemption_catalog"
        assert out["status"] == AgentStatus.SUCCESS.value
        assert out["node_history"] == ["InputValidateNode"]

    def test_route_returns_end_on_error(self):
        inner = DomainWorkflowGraph()
        assert inner.route({"status": AgentStatus.ERROR.value}) == END
        assert inner.route({"status": AgentStatus.SUCCESS.value}) == "output_format"


class TestInnerEndToEnd:
    def _invoke(self, payload: str) -> dict:
        # Same construction path the outer GraphNode uses: manifest-derived
        # config via _parent_config(); domain nodes take NO ctor args.
        inner = DomainWorkflowGraph(config=LoyaltyOfferQAGraphNode()._parent_config())
        return inner.invoke(payload, session_id="inner-e2e")

    def test_int_04_full_inner_run_produces_the_formatted_answer(self):
        result = self._invoke(_REDEMPTION_QUERY)
        assert result["status"] == AgentStatus.SUCCESS.value
        answer = result["formatted_answer"]
        assert answer.startswith("# Loyalty Program Q&A Result")
        assert "[1]" in answer
        assert "does not reflect your live points balance or account history" in answer
        citations = from_json(result["citations"])
        assert citations and citations[0]["id"] == "kb-011"
        assert result["answer_category"] == "redemption_catalog"

    def test_int_04_inner_node_history_is_the_linear_topology(self):
        history = self._invoke(_REDEMPTION_QUERY)["node_history"]
        assert history == [
            "InputValidateNode",
            "RetrieveNode",
            "RerankFilterNode",
            "GenerateAnswerNode",
            "OutputFormatNode",
        ]

    def test_no_coverage_query_still_terminates_success(self):
        result = self._invoke("quantum telepathy sandwich recipes")
        assert result["status"] == AgentStatus.SUCCESS.value
        assert "does not contain sufficient coverage" in result["formatted_answer"]
        assert result["answer_category"] is None
