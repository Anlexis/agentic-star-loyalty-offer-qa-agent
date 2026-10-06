"""AgentCore Platform v1.0"""

# RET-C2-087 - DomainWorkflowGraph (inner BaseGraph)
#
# The inner graph of the two-layer nested architecture. It encapsulates the
# whole loyalty-program Q&A workflow:
#
#   START -> input_validate -> retrieve -> rerank_filter
#         -> generate_answer -> output_format -> END
#
# Called by LoyaltyOfferQAGraphNode.get_subgraph() (graph.py).
# get_output() shapes the sub_result dict consumed by merge_output() there.
#
# Rules enforced:
#   - Inherits BaseGraph (fully custom topology - no forced backbone)
#   - register_nodes() does NOT call super() (abstract in BaseGraph)
#   - register_nodes() instantiates every domain node with NO ctor args
#   - Does NOT register initialize / finalize (outer backbone concerns)
#   - get_output() designed together with LoyaltyOfferQAGraphNode.merge_output()

from typing import Any, Dict

from langgraph.graph import END, START

from framework.errors import ConfigError
from framework.graph.base_graph import BaseGraph
from framework.schemas.agent_state import AgentState
from framework.schemas.agent_status import AgentStatus
from src.graph.context_bridge import get_caller_input_context, get_resolved_retrieval_config
from src.graph.runtime_config import resolve_retrieval, validate_retrieval_block
from src.nodes.generate_answer_node import GenerateAnswerNode
from src.nodes.input_validate_node import InputValidateNode
from src.nodes.output_format_node import OutputFormatNode
from src.nodes.rerank_filter_node import RerankFilterNode
from src.nodes.retrieve_node import RetrieveNode
from src.schemas.caller_contract import CallerFieldError
from src.schemas.state import State, from_json, to_json


class DomainWorkflowGraph(BaseGraph):
    """Inner domain workflow graph for RET-C2-087.

    Inherits BaseGraph directly for a fully custom node topology. Called by
    LoyaltyOfferQAGraphNode.get_subgraph() in graph.py, which passes the
    resolved runtime config into the constructor.

    Pipeline (linear):
        START
          -> input_validate  (InputValidateNode)  - parse + normalise the query
          -> retrieve        (RetrieveNode)       - keyword-score the seeded KB
          -> rerank_filter   (RerankFilterNode)   - boost / threshold / top_k cut
          -> generate_answer (GenerateAnswerNode) - grounded answer + citations
          -> output_format   (OutputFormatNode)   - final format + disclaimer
          -> END

    All nodes are FunctionNode subclasses returning partial-dict state updates.
    initialize / finalize are outer backbone concerns - not registered here.
    """

    # -- Identity --------------------------------------------------------------

    @property
    def name(self) -> str:
        """Unique identifier for this inner graph."""
        return "ret_c2_087_loyalty_offer_qa_workflow"

    @property
    def state_schema(self) -> type:
        """TypedDict subclass shared across inner and outer graph."""
        return State

    # -- Config validation -----------------------------------------------------

    def _validate_config(self) -> None:
        """Reject a malformed retrieval block before the graph is compiled.

        The declared relevance floor and candidate cap decide which passages
        can ever reach an answer, so a value that is out of range - or not a
        finite number at all - must stop the deployment rather than quietly
        change every answer. A non-finite threshold is the dangerous case: it
        compares False against every score, and the usual clamp turns it into
        the strictest possible floor, so the agent refuses questions the
        knowledge base covers.
        """
        block = (self.config or {}).get("configurable", {}).get("retrieval")
        if block is None:
            return
        try:
            validate_retrieval_block(block)
        except CallerFieldError as exc:
            raise ConfigError(f"[{self.__class__.__name__}] invalid retrieval configuration: {exc}") from None

    # -- Config + caller context forwarding into state --------------------------

    def _extra_initial_state(self) -> Dict[str, Any]:
        """Seed the inner state with the retrieval block and the caller context.

        Two hand-offs happen here, both because the framework's GraphNode
        invokes a subgraph with the user input alone:

        `retrieval_config` - the effective retrieval block as a JSON string.
        The live values are the ones the outer graph published into outer state
        from config/config.yaml and the context bridge carried across; the
        constructor config is the fallback for a directly-invoked inner graph.
        RetrieveNode and RerankFilterNode read this state field, because
        execute() takes only `state` - there is no `config` parameter.

        `input_context` - the caller's own retrieval overrides. Without this
        line every inner read of state["input_context"] would see {} no matter
        what the caller sent.
        """
        retrieval = from_json(get_resolved_retrieval_config(), None)
        if not isinstance(retrieval, dict) or not retrieval:
            retrieval = resolve_retrieval({"retrieval": (self.config or {}).get("configurable", {}).get("retrieval")})

        return {
            "retrieval_config": to_json(retrieval),
            "input_context": get_caller_input_context(),
        }

    # -- Node registration -----------------------------------------------------

    def register_nodes(self) -> None:
        """Register all 5 domain nodes.

        No super() call - BaseGraph.register_nodes() is abstract.
        Do NOT register initialize or finalize; those are outer backbone
        concerns handled by AgentBaseGraph in graph.py.

        Every node is instantiated with NO constructor arguments; configuration
        flows in via the state field seeded above, never a per-call execute()
        parameter.
        """
        self._nodes["input_validate"] = InputValidateNode()
        self._nodes["retrieve"] = RetrieveNode()
        self._nodes["rerank_filter"] = RerankFilterNode()
        self._nodes["generate_answer"] = GenerateAnswerNode()
        self._nodes["output_format"] = OutputFormatNode()

    # -- Edge wiring -----------------------------------------------------------

    def add_edges(self) -> None:
        """Wire the linear loyalty Q&A domain topology.

        Each step passes its partial-dict output into the shared State. The
        topology is intentionally linear - no conditional branching between
        domain nodes. route() is implemented as the base class requires, but
        add_conditional_edges() is not used.
        """
        self._sg.add_edge(START, "input_validate")
        self._sg.add_edge("input_validate", "retrieve")
        self._sg.add_edge("retrieve", "rerank_filter")
        self._sg.add_edge("rerank_filter", "generate_answer")
        self._sg.add_edge("generate_answer", "output_format")
        self._sg.add_edge("output_format", END)

    # -- Routing ---------------------------------------------------------------

    def route(self, state: AgentState) -> str:
        """Conditional routing - required by the BaseGraph interface.

        For this linear topology add_conditional_edges() is not used, so this
        method is never called at runtime. It returns END on error so an
        unexpected call cannot re-enter a processing node.
        """
        if state.get("status") == AgentStatus.ERROR.value:
            return END
        return "output_format"

    # -- Output shape ----------------------------------------------------------

    def get_output(self, state: AgentState) -> Dict[str, Any]:
        """Shape the output dict returned to the outer graph as sub_result.

        This dict is received by LoyaltyOfferQAGraphNode.merge_output() in
        graph.py as its `sub_result` argument. Both methods are designed
        together so the field names agree:

            Inner get_output()   emits: "formatted_answer", "citations",
                                        "answer_category", "grounding_excerpts",
                                        "status", ...
            Outer merge_output() reads: the same names.

        `grounding_excerpts` carries the passage text the answer was assembled
        from, so the outer output gate can verify that every figure in the
        rendered answer came from the knowledge base.
        """
        return {
            "formatted_answer": state.get("formatted_answer"),
            "citations": state.get("citations"),
            "answer_category": state.get("answer_category"),
            "grounding_excerpts": state.get("grounding_excerpts"),
            "status": state.get("status"),
            "intake_notes": state.get("intake_notes"),
            "trace_id": state.get("trace_id"),
            "correlation_id": state.get("correlation_id"),
            "node_history": state.get("node_history", []),
        }
