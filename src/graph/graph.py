"""AgentCore Platform v1.0"""

# RET-C2-087 - outer graph (Category 2, two-layer nested architecture)
#
# Loyalty Offer Q&A Agent.
#
# Architecture:
#
#   Outer backbone (fixed - do NOT override add_edges()):
#     START -> initialize -> pre_process -> main -> {route} -> post_process -> finalize -> END
#                                             |  (retry, max 3)
#                                             -> pre_process
#
#   The `main` slot is a GraphNode subclass (LoyaltyOfferQAGraphNode) that
#   delegates the full loyalty Q&A domain workflow to DomainWorkflowGraph
#   (inner BaseGraph: input_validate -> retrieve -> rerank_filter ->
#   generate_answer -> output_format).
#
#   Domain complexity is fully encapsulated inside the inner graph. The outer
#   backbone is never modified.
#
# Directory layout:
#   src/graph/graph.py                 <- outer graph (this file)
#   src/graph/domain_workflow_graph.py <- inner graph (multi-step topology)
#   src/graph/context_bridge.py        <- carries input_context across the boundary
#   src/graph/runtime_config.py        <- config/config.yaml loader + validation
#
# Class-name contract:
#   graph.py class:           LoyaltyOfferQAAgent (this file)
#   config/agent.yaml class:  "src.graph.graph.LoyaltyOfferQAAgent"  <- must match
#   src/api/server.py import: from src.graph.graph import LoyaltyOfferQAAgent

from typing import Any, ClassVar, Dict, cast

from framework.graph.agent_base_graph import AgentBaseGraph
from framework.nodes.graph_node import GraphNode
from framework.schemas.agent_state import AgentState
from framework.schemas.agent_status import AgentStatus
from src.graph.context_bridge import set_caller_input_context, set_resolved_retrieval_config
from src.graph.runtime_config import resolve_retrieval
from src.nodes.post_process_node import (
    PostProcessNode,
    _enforce_numeric_fidelity,
    _security_gate_output,
    grounding_text,
)
from src.nodes.pre_process_node import PreProcessNode
from src.schemas.state import State, from_json, to_json


class LoyaltyOfferQAGraphNode(GraphNode):
    """GraphNode subclass assigned to the `main` slot of the outer agent.

    Wraps DomainWorkflowGraph (the inner loyalty-Q&A pipeline). Called by the
    backbone after pre_process and before post_process.

    Contracts:
      get_subgraph()    - instantiate DomainWorkflowGraph with the resolved
                          runtime config (_parent_config())
      extract_input()   - pull validated_input (identifier-stripped) from outer
                          state, and stash the caller's input_context so the
                          inner graph can read it
      merge_output()    - map sub_result fields into the outer state delta
                          (changed keys only)
      error_strategy    - "propagate": re-raise inner errors as SubgraphError
    """

    # "propagate": re-raise inner graph exceptions as SubgraphError (fail fast).
    # "handle": call on_subgraph_error() instead - for graceful degradation.
    error_strategy: ClassVar[str] = "propagate"

    # False: human-in-the-loop interrupts stay inside the inner graph. This
    # template does not use them at all.
    propagate_hitl: ClassVar[bool] = False

    def _parent_config(self) -> Dict[str, Any]:
        """Return the retrieval block the inner graph is constructed with.

        Values come from config/config.yaml (see src/graph/runtime_config.py),
        validated on the way through, and are never empty. This is the
        construction-time fallback; when the agent runs as a whole graph the
        live values arrive through the context bridge instead, which lets a
        registry-supplied config override the file on disk.
        """
        return {"configurable": {"retrieval": resolve_retrieval()}}

    def get_subgraph(self) -> Any:
        """Instantiate and return the inner domain workflow graph.

        DomainWorkflowGraph is imported inside the method to keep module load
        order independent of the inner graph.

        The inner graph receives its configuration through the BaseGraph
        constructor - graph-level injection of immutable config, which is
        distinct from the per-node execute() contract. The domain nodes
        themselves take no constructor arguments and read configuration
        exclusively from the state field seeded by the inner graph.
        """
        from src.graph.domain_workflow_graph import DomainWorkflowGraph

        return DomainWorkflowGraph(config=self._parent_config())

    def extract_input(self, state: AgentState) -> str:
        """Return the string input passed into inner_graph.invoke().

        Also stashes, for the inner graph, the two values the framework's
        GraphNode does not carry across the subgraph call: the caller's
        input_context, and the retrieval block the outer graph resolved from
        config/config.yaml. Without this hand-off every inner read of
        input_context would see an empty mapping, and the declared retrieval
        values would never take effect (see src/graph/context_bridge.py).

        PreProcessNode validates and identifier-strips the raw user_input and
        writes the result to validated_input. Prefer that; fall back to
        user_input when validated_input is absent (a bare unit-test state).
        """
        set_caller_input_context(state.get("input_context") or {})
        set_resolved_retrieval_config(state.get("retrieval_config"))
        return cast(str, state.get("validated_input") or state.get("user_input", ""))

    def merge_output(self, state: AgentState, sub_result: Dict[str, Any]) -> Dict[str, Any]:
        """Map the inner graph's sub_result back into the outer state delta.

        Returns ONLY changed keys - never the full state.

        Key coupling (designed together with DomainWorkflowGraph.get_output()):
          Inner get_output() emits  -> "formatted_answer", "citations",
                                       "answer_category", "grounding_excerpts",
                                       "status", ...
          This merge_output() reads -> the same names.

        loyalty_answer: the rendered loyalty Q&A answer, written by
          OutputFormatNode inside the inner graph.
        result: PostProcessNode reads state["result"], so the rendered answer
          is mapped there as well; otherwise the output the output gate sees
          is always empty.
        grounding_excerpts: the passage text the answer was assembled from.
          The output gate needs it to check that every figure in the rendered
          answer came from the knowledge base.
        answer_category: re-surfaced at the outer layer as a plain scalar so
          get_output() can read it without reaching into the inner graph.
        status: terminal status value from the inner graph run.
        """
        return {
            "loyalty_answer": sub_result.get("formatted_answer"),
            "result": sub_result.get("formatted_answer"),
            "citations": sub_result.get("citations"),
            "answer_category": sub_result.get("answer_category"),
            "grounding_excerpts": sub_result.get("grounding_excerpts"),
            "status": sub_result.get("status"),
        }


class LoyaltyOfferQAAgent(AgentBaseGraph):
    """Outer graph for RET-C2-087.

    Inherits AgentBaseGraph directly. Domain logic is fully encapsulated in
    LoyaltyOfferQAGraphNode (main slot), which delegates to DomainWorkflowGraph.

    Backbone (fixed):
        START -> initialize -> pre_process -> main -> post_process -> finalize -> END

    register_nodes(), _extra_initial_state() and get_output() are the only
    overrides:
      - super().register_nodes() fills initialize and finalize
      - pre_process:  PreProcessNode (trust gate, caller-contract validation,
                      identifier screen)
      - main:         LoyaltyOfferQAGraphNode (delegates to DomainWorkflowGraph)
      - post_process: PostProcessNode (output gate)
      - _extra_initial_state(): publishes the resolved retrieval block so the
        values declared in config/config.yaml reach the domain nodes
      - get_output(): extends the base envelope with the structured
        citations / answer_category fields, on SUCCESS only, fail-closed,
        and withholds `output` entirely on any non-success status

    add_edges() is NOT overridden - backbone wiring belongs to the framework.
    """

    @property
    def name(self) -> str:
        """Agent identifier registered with the agent registry."""
        return "LoyaltyOfferQAAgent"

    @property
    def state_schema(self) -> type:
        return State

    def _extra_initial_state(self) -> Dict[str, Any]:
        """Publish the resolved retrieval block into the initial outer state.

        `self.config` is what the registry (or src/api/server.py) loaded from
        config/config.yaml; when the agent is constructed bare, the resolver
        reads the same file directly. Either way the values a deployment
        DECLARES are the values the domain nodes run with - the block travels
        outer state -> context bridge -> inner state, and RetrieveNode /
        RerankFilterNode read it from there.
        """
        return {"retrieval_config": to_json(resolve_retrieval(self.config))}

    def register_nodes(self) -> None:
        """Fill all 5 backbone slots.

        super().register_nodes() MUST be called first - it injects the
        framework's default InitializeNode (schema_version, session_id,
        trust_level) and FinalizeNode (response_metadata, total_time_ms).
        """
        super().register_nodes()  # fills: initialize, finalize

        self._nodes["pre_process"] = PreProcessNode()
        self._nodes["main"] = LoyaltyOfferQAGraphNode()
        self._nodes["post_process"] = PostProcessNode()

    # add_edges() is NOT overridden - backbone wiring belongs to the framework.

    def get_output(self, state: AgentState) -> Dict[str, Any]:
        """Extend the base envelope with the structured fields.

        The grounding citations and the matched knowledge-base category ARE
        the product, not a side note, so they are surfaced as real (decoded)
        Python values on top of the base output / status / trace_id /
        correlation_id / node_history envelope.

        Fail-closed, three times over:
          1. `output` is surfaced only on a SUCCESS terminal status. The base
             envelope resolves it as `formatted_output or result` with no
             status check of its own, so any non-success outcome - a trust
             denial, a blocked output, a subgraph error - would otherwise be
             free to carry whichever of those two the state still holds. A
             caller told the request failed receives no payload.
          2. The structured fields are attached only when the terminal status
             is SUCCESS - any other outcome returns the base envelope alone.
          3. Even on SUCCESS the fields go through the SAME two gates
             PostProcessNode applies, which covered only `result`; this second
             pass covers the fields added here, because an invariant that holds
             for one representation of the product and not the other does not
             hold. A violation flips the status to ERROR and withholds the
             structured fields AND `output` - never a partial payload. This is
             the one refusal PostProcessNode cannot contain on its own: it
             passed, so it cleared nothing, and the rendered answer is still
             the live `formatted_output` when the second pass refuses.

        Both fields are knowledge-base-derived (citations reference seeded
        entries; answer_category is the top-ranked entry's category), never an
        echo of the caller's query, so no member-supplied value can reach this
        envelope.
        """
        base: Dict[str, Any] = super().get_output(state)

        if state.get("status") != AgentStatus.SUCCESS.value:
            return self._withhold_output(base)

        answer_category = state.get("answer_category")
        citations = from_json(state.get("citations"), []) or []

        structured = {"answer_category": answer_category, "citations": citations}
        if _security_gate_output(structured) or _enforce_numeric_fidelity(
            structured, grounding_text(state), len(citations)
        ):
            base["status"] = AgentStatus.ERROR.value
            return self._withhold_output(base)

        base["answer_category"] = answer_category
        base["citations"] = citations
        return base

    @staticmethod
    def _withhold_output(base: Dict[str, Any]) -> Dict[str, Any]:
        """Drop the payload from a non-success envelope.

        Re-resolves `output` rather than trusting what the base class put
        there. The structured fields are never added on this path, so nothing
        else has to be removed.
        """
        base["output"] = None
        return base


# Back-compat alias - config/agent.yaml declares
# class: "src.graph.graph.LoyaltyOfferQAAgent", and src/api/server.py imports
# the class directly. Keep both names pointing at the agent.
Graph = LoyaltyOfferQAAgent
