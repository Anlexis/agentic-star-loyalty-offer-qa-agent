"""AgentCore Platform v1.0"""

# RET-C2-087 - carries values across the outer -> inner graph boundary.
#
# Why this exists: GraphNode.execute() invokes the inner graph as
# `subgraph.invoke(user_input, session_id=..., ctx=...)`. Only the user input
# crosses that call, so anything else the inner nodes need - the caller's
# input_context, and the retrieval block the outer graph resolved from
# config/config.yaml - would be lost. An inner read of state["input_context"]
# would always see {} no matter what the caller sent, and the inner graph would
# silently fall back to its own defaults instead of the declared configuration.
#
# The sanctioned subclass hooks hand the values over:
#
#   LoyaltyOfferQAGraphNode.extract_input(state)   [runs BEFORE subgraph.invoke]
#       -> set_caller_input_context(state["input_context"])
#       -> set_resolved_retrieval_config(state["retrieval_config"])
#   DomainWorkflowGraph._extra_initial_state()     [runs INSIDE subgraph.invoke]
#       -> seeds both into the inner initial state
#
# ContextVars keep each hand-off correct per thread and per async task, so
# concurrent invocations in one process cannot read each other's values.

from contextvars import ContextVar
from typing import Any, Dict, Optional

_CALLER_INPUT_CONTEXT: ContextVar[Optional[Dict[str, Any]]] = ContextVar(
    "ret_c2_087_caller_input_context", default=None
)

_RESOLVED_RETRIEVAL_CONFIG: ContextVar[Optional[str]] = ContextVar("ret_c2_087_resolved_retrieval_config", default=None)


def set_caller_input_context(input_context: Optional[Dict[str, Any]]) -> None:
    """Stash the outer graph's input_context for the imminent inner-graph invoke."""
    _CALLER_INPUT_CONTEXT.set(dict(input_context) if input_context else {})


def get_caller_input_context() -> Dict[str, Any]:
    """Read (without consuming) the stashed input_context; {} when none was set."""
    return _CALLER_INPUT_CONTEXT.get() or {}


def set_resolved_retrieval_config(retrieval_config: Optional[str]) -> None:
    """Stash the outer graph's resolved retrieval block (a JSON string)."""
    _RESOLVED_RETRIEVAL_CONFIG.set(retrieval_config or None)


def get_resolved_retrieval_config() -> Optional[str]:
    """Read the stashed retrieval block; None when the outer graph set none."""
    return _RESOLVED_RETRIEVAL_CONFIG.get()


def clear_boundary_values() -> None:
    """Drop everything stashed - used by tests to prove one run cannot see another's."""
    _CALLER_INPUT_CONTEXT.set(None)
    _RESOLVED_RETRIEVAL_CONFIG.set(None)
