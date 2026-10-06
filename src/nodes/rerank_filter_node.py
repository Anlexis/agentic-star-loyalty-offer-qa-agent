"""AgentCore Platform v1.0"""

# RET-C2-087 - RerankFilterNode
# Domain node 3: rerank the retrieval candidates and enforce the relevance
# floor. Deterministic: a small category-match boost on top of the retrieval
# score, drop everything below `score_threshold`, cap the survivors at
# `top_k`.
#
# Config: reads `top_k` / `score_threshold` EXCLUSIVELY from the state-seeded
# `retrieval_config` field, published by DomainWorkflowGraph
# ._extra_initial_state() from config/config.yaml. execute() takes no `config`
# parameter, so this state field is the only route configuration reaches the
# node.
#
# A caller override (query_filters) applies only when it is STRICTER than the
# declared value - a smaller candidate cap, a higher relevance floor. A caller
# can narrow what an answer is built from; it cannot widen it below the floor
# the deployment declared.
#
# Neither number is clamped. `max(0.0, min(1.0, x))` looks like a safe guard
# and is not: with a non-finite x every comparison is False, the clamp returns
# 1.0, and the strictest possible floor is installed silently - so the agent
# refuses every question the knowledge base does cover. Values are validated,
# and a bad one refuses the request.
#
# Wired by the inner graph (DomainWorkflowGraph).
# Returns only changed state keys (partial dict).

from typing import Any, ClassVar, Dict, List

from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_state import AgentState
from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel
from shared.utils.audit_logger import emit_trace_event
from src.graph.runtime_config import DEFAULT_RETRIEVAL, validate_retrieval_block
from src.schemas.caller_contract import (
    SCORE_THRESHOLD_MAX,
    SCORE_THRESHOLD_MIN,
    TOP_K_MAX,
    TOP_K_MIN,
    CallerFieldError,
    finite_in_range,
)
from src.schemas.state import from_json, to_json

# Boost applied when a candidate's category matches the caller's filter.
_CATEGORY_BOOST = 0.1


def _resolve_retrieval_config(state: AgentState) -> Dict[str, Any]:
    """Effective retrieval config: the state-seeded block, validated.

    No `config` parameter is read here - execute() takes only `state`. Raises
    CallerFieldError when the seeded block carries a value outside its bounds.
    """
    effective = dict(DEFAULT_RETRIEVAL)  # local copy - never mutate the module default
    from_state = from_json(state.get("retrieval_config"), None)
    if isinstance(from_state, dict):
        effective.update(from_state)
    return validate_retrieval_block(effective)


class RerankFilterNode(FunctionNode):
    """Rerank candidates, apply the score threshold, cap at top_k.

    Input state keys:
        retrieved_documents: JSON list of scored candidates (from RetrieveNode)
        query_filters:       JSON dict with optional category / top_k override
        retrieval_config:    forwarded manifest retrieval block (JSON)

    Output state keys (partial dict):
        ranked_documents: JSON list of surviving passages (score desc, <= top_k)
    """

    required_trust_level: ClassVar[TrustLevel] = TrustLevel.ANONYMOUS

    def execute(self, state: AgentState) -> Dict[str, Any]:
        candidates: List[Dict[str, Any]] = from_json(state.get("retrieved_documents"), []) or []
        filters = from_json(state.get("query_filters"), {}) or {}
        try:
            retrieval_cfg = _resolve_retrieval_config(state)
        except CallerFieldError as exc:
            return {
                "status": AgentStatus.ERROR.value,
                "error_log": [f"RerankFilterNode: {exc}"],
            }

        top_k = int(retrieval_cfg["top_k"])
        score_threshold = float(retrieval_cfg["score_threshold"])

        # A caller override applies only when it is stricter than the declared
        # value. The bounds are re-checked here rather than assumed: this node
        # can also be driven from a state assembled directly, and a non-finite
        # threshold arriving that way would otherwise pass straight through.
        try:
            if filters.get("top_k") is not None:
                top_k = min(
                    top_k,
                    int(
                        finite_in_range(
                            filters["top_k"],
                            field="top_k",
                            minimum=TOP_K_MIN,
                            maximum=TOP_K_MAX,
                            integer=True,
                        )
                    ),
                )
            if filters.get("score_threshold") is not None:
                score_threshold = max(
                    score_threshold,
                    finite_in_range(
                        filters["score_threshold"],
                        field="score_threshold",
                        minimum=SCORE_THRESHOLD_MIN,
                        maximum=SCORE_THRESHOLD_MAX,
                    ),
                )
        except CallerFieldError as exc:
            return {
                "status": AgentStatus.ERROR.value,
                "error_log": [f"RerankFilterNode: {exc}"],
            }

        category = filters.get("category")

        reranked: List[Dict[str, Any]] = []
        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue
            entry = dict(candidate)  # local copy - inputs stay immutable
            try:
                score = float(entry.get("score", 0.0))
            except (TypeError, ValueError):
                score = 0.0
            if category and str(entry.get("category", "")).lower() == str(category).lower():
                score = min(1.0, score + _CATEGORY_BOOST)
            entry["score"] = round(score, 4)
            reranked.append(entry)

        # Deterministic ordering: score desc, then id asc for stable ties.
        reranked.sort(key=lambda c: (-c.get("score", 0.0), str(c.get("id", ""))))

        kept = [c for c in reranked if c.get("score", 0.0) >= score_threshold][:top_k]
        dropped = len(reranked) - len(kept)

        # S-4 domain audit: rerank + relevance floor applied.
        emit_trace_event(
            "rerank_filter_complete",
            {
                "kept": len(kept),
                "dropped": dropped,
                "score_threshold": score_threshold,
                "top_k": top_k,
            },
            state,
        )

        return {"ranked_documents": to_json(kept)}
