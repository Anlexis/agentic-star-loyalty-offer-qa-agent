"""AgentCore Platform v1.0"""

# ADR-005: State must be a flat TypedDict - never Pydantic BaseModel.
# LangGraph checkpoints use msgpack serialization; Pydantic objects
# cause silent corruption.  Extend AgentState with agent-specific
# fields only.  Do NOT add credentials, secrets, or Pydantic models.
#
# ADR-005 (msgpack safety): structured fields (dict / list[dict]) are stored
# as JSON STRINGS, not bare Python containers - a bare dict/list in a
# checkpointed State field is a CoE gate-state-safety violation. Producers
# serialize with to_json() on write; consumers deserialize with from_json()
# on read.
#
# RET-C2-087 - LoyaltyOfferQAAgent (Cat 2 RAG, nested).
# Two-layer nested Cat 2 graph: outer backbone (AgentBaseGraph) + inner
# domain workflow (BaseGraph).  Fields below cover both layers.
#
# PII / confidentiality note: this agent handles loyalty POINTS BALANCES and
# member identity. A member id, loyalty-card number, or claimed points
# balance in the query payload is surface-stripped by PreProcessNode (S-1/S-2)
# before any field is written to State. Only the normalised search query, KB
# passage summaries, and the final grounded answer are persisted - never raw
# member/CSR-supplied identifiers, and never a live points balance (v1 has no
# loyalty-platform/POS integration - docs/01_proposal.md "Scope Revision").

import json
from typing import Any, NotRequired, Optional

from framework.schemas.agent_state import AgentState


def to_json(value: Any) -> Optional[str]:
    """Serialize a dict/list State field to a JSON string (ADR-005 msgpack safety).

    None passes through unchanged so an 'unset' field stays distinguishable
    from an empty container.
    """
    if value is None:
        return None
    return json.dumps(value, ensure_ascii=False)


def from_json(value: Optional[str], default: Any = None) -> Any:
    """Deserialize a JSON-string State field back to its dict/list.

    None / empty / malformed input -> the supplied ``default`` so a missing or
    corrupt field is non-fatal for the consuming node.
    """
    if not value:
        return default
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return default


class State(AgentState):
    """Flat TypedDict for RET-C2-087.

    All shared fields (user_input, status, session_id, node_history,
    error_log, hitl_*, etc.) are inherited from AgentState.
    Domain fields are NotRequired so the TypedDict is valid at graph
    initialisation, before any node has written a value.
    """

    # ------------------------------------------------------------------
    # Outer layer - set by PreProcessNode / LoyaltyOfferQAGraphNode.merge_output
    # ------------------------------------------------------------------

    # Identifier-stripped, validated query payload produced by PreProcessNode
    # (S-1/S-2).  Raw input is NOT persisted beyond PreProcessNode.
    validated_input: NotRequired[str]

    # Final loyalty-program Q&A answer, mapped from the inner graph's
    # formatted_answer output via merge_output.
    loyalty_answer: NotRequired[str]

    # KB category the answer was grounded in - one of "program_rules",
    # "offer_catalog", "redemption_catalog", or None when no KB passage
    # cleared the relevance threshold. Set by the inner GenerateAnswerNode
    # from the top-ranked document's category, re-surfaced at the outer
    # layer by merge_output() so get_output() (RULES B10) can read it
    # without reaching into the inner graph.
    answer_category: NotRequired[Optional[str]]

    # ------------------------------------------------------------------
    # Inner layer - domain nodes (DomainWorkflowGraph)
    # ------------------------------------------------------------------

    # InputValidateNode outputs
    # Normalised free-text search query (whitespace-collapsed, length-capped).
    search_query: NotRequired[str]

    # JSON STRING (to_json) of parsed structured query params. Deserialised
    # dict shape: {"category": str | None, "top_k": int | None}.
    # Consumers (RetrieveNode, RerankFilterNode) read it back via from_json().
    query_filters: NotRequired[Optional[str]]

    # Manifest `retrieval` block forwarded by LoyaltyOfferQAGraphNode.
    # _parent_config() -> DomainWorkflowGraph._extra_initial_state().
    # JSON STRING (to_json) of {"top_k": int, "score_threshold": float,
    # "kb_path": str}. Consumers (RetrieveNode, RerankFilterNode) read it
    # back via from_json() - execute() takes no `config` parameter (SDK
    # v1.0.0rc1 / CoE 2026-07-27 contract), so this state field is the ONLY
    # route config knobs reach those nodes.
    retrieval_config: NotRequired[Optional[str]]

    # RetrieveNode output
    # JSON STRING (to_json) of scored KB candidates. Deserialised shape:
    # list[dict], each entry {"id": str, "title": str, "category": str,
    # "source": str, "score": float, "excerpt": str}.
    # Consumers (RerankFilterNode) read it back via from_json().
    retrieved_documents: NotRequired[Optional[str]]

    # RerankFilterNode output
    # JSON STRING (to_json) of reranked + threshold-filtered passages, capped
    # at top_k. Same entry shape as retrieved_documents.
    # Consumers (GenerateAnswerNode) read it back via from_json().
    ranked_documents: NotRequired[Optional[str]]

    # GenerateAnswerNode outputs
    # Rule-assembled grounded answer body with numbered citation markers.
    # Never contains a verbatim copy of the caller's query (see
    # docs/02_design.md "Security Gates" - S-2 design decision).
    grounded_answer: NotRequired[str]

    # JSON STRING (to_json) of the knowledge-base passages the answer was
    # assembled from (id + title + excerpt + source per entry). The output
    # boundary reads it to confirm that every figure in the rendered answer
    # appears in the corpus - see src/nodes/post_process_node.py. Also
    # re-surfaced at the outer layer by merge_output().
    grounding_excerpts: NotRequired[Optional[str]]

    # JSON STRING (to_json) of citations. Deserialised shape: list[dict],
    # each entry {"ref": int, "id": str, "title": str, "source": str}.
    # Consumers (OutputFormatNode) read it back via from_json(). Also
    # re-surfaced at the outer layer by merge_output() (RULES B10).
    citations: NotRequired[Optional[str]]

    # OutputFormatNode output
    # Final formatted answer (body + sources + v1-scope disclaimer). Written
    # by OutputFormatNode; surfaced to the outer graph via get_output() ->
    # merge_output().
    formatted_answer: NotRequired[str]

    # Validation / parse notes accumulated during intake (no PII).
    # JSON STRING (to_json) of list[str].
    intake_notes: NotRequired[Optional[str]]

    # ------------------------------------------------------------------
    # Tracing / audit - framework-managed; do NOT write from node code
    # ------------------------------------------------------------------

    trace_id: Optional[str]
    correlation_id: Optional[str]
    # node_history inherited from AgentState; listed here for clarity
    # node_history: Optional[List[str]]
