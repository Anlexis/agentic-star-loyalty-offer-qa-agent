"""AgentCore Platform v1.0"""

# RET-C2-087 - InputValidateNode
# Domain node 1: parse and normalise the incoming loyalty Q&A request.
#
# Retrieval parameters can arrive through two channels, and BOTH are caller
# data, so both are validated against the same bounds:
#
#   input_context   the first-class channel - {"category": ..., "top_k": ...,
#                   "score_threshold": ...} passed alongside the question and
#                   carried into this graph by the context bridge.
#   a JSON envelope the question string itself may be
#                   {"query": "...", "category": "...", "top_k": N}, which the
#                   outer graph passes through verbatim and this node parses back.
#
# Where both supply the same field, input_context wins - one stated rule rather
# than whichever happens to be read last.
#
# A value that fails its bounds REFUSES the request. It is not clamped and not
# dropped: silently substituting a different number answers a question the
# caller did not ask, and the caller has no way to tell that happened.
#
# `category` is matched against the knowledge base's own category vocabulary as
# a whole value. Substring matching here would be a quiet correctness bug -
# a filter that matches the wrong category discards exactly the passages that
# answer the question, and the agent then reports no coverage.
#
# Wired by the inner graph (DomainWorkflowGraph).
# Returns only changed state keys (partial dict).

import json
import re
from typing import Any, ClassVar, Dict, List

from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_state import AgentState
from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel
from shared.utils.audit_logger import emit_trace_event
from src.schemas.caller_contract import CallerFieldError, validate_caller_context
from src.schemas.state import to_json

# Hard cap on the normalised query length (defence in depth on input size).
_MAX_QUERY_CHARS = 2000

_WHITESPACE_RE = re.compile(r"\s+")

# Fields the in-question JSON envelope may carry alongside the query text.
_ENVELOPE_QUERY_KEYS = ("query", "question")
_ENVELOPE_PARAM_KEYS = ("category", "top_k", "score_threshold")


class InputValidateNode(FunctionNode):
    """Parse the (possibly JSON-enveloped) request into a normalised query.

    Input state keys:
        validated_input | user_input: identifier-stripped request payload
        input_context:                caller retrieval overrides

    Output state keys (partial dict):
        search_query:  normalised free-text search query
        query_filters: JSON dict {"category": str|None, "top_k": int|None,
                       "score_threshold": float|None}
        intake_notes:  (when anomalies were seen) JSON list[str]
    """

    required_trust_level: ClassVar[TrustLevel] = TrustLevel.ANONYMOUS

    def execute(self, state: AgentState) -> Dict[str, Any]:
        raw = state.get("validated_input") or state.get("user_input", "")
        notes: List[str] = []

        query = ""
        envelope_params: Dict[str, Any] = {}

        if isinstance(raw, str) and raw.strip():
            payload: Any = None
            text = raw.strip()
            if text.startswith("{"):
                try:
                    payload = json.loads(text)
                except (json.JSONDecodeError, ValueError):
                    notes.append(
                        "InputValidateNode: JSON-looking input did not parse - " "treated as plain text query."
                    )
            if isinstance(payload, dict):
                for key in _ENVELOPE_QUERY_KEYS:
                    if payload.get(key):
                        query = str(payload.get(key))
                        break
                envelope_params = {key: payload[key] for key in _ENVELOPE_PARAM_KEYS if key in payload}
            else:
                query = text
        else:
            notes.append("InputValidateNode: empty request - no query to search.")

        # Both channels through the same bounds. input_context wins on a clash.
        try:
            merged = dict(envelope_params)
            merged.update({k: v for k, v in (state.get("input_context") or {}).items() if k in _ENVELOPE_PARAM_KEYS})
            params = validate_caller_context(merged)
        except CallerFieldError as exc:
            emit_trace_event("input_validate_refused", {"field": exc.field}, state)
            return {
                "status": AgentStatus.ERROR.value,
                "error_log": [f"InputValidateNode: {exc}"],
            }

        # Normalise whitespace and cap length.
        query = _WHITESPACE_RE.sub(" ", query).strip()
        if len(query) > _MAX_QUERY_CHARS:
            query = query[:_MAX_QUERY_CHARS]
            notes.append(f"InputValidateNode: query truncated to {_MAX_QUERY_CHARS} chars.")

        filters: Dict[str, Any] = {
            "category": params.get("category"),
            "top_k": params.get("top_k"),
            "score_threshold": params.get("score_threshold"),
        }

        # Domain audit: request parsed and normalised. No query text, no
        # category value, and no digit content (points / member numbers) in the
        # payload - counts and booleans only.
        emit_trace_event(
            "input_validate_complete",
            {
                "query_chars": len(query),
                "has_category_filter": filters["category"] is not None,
                "has_top_k_override": filters["top_k"] is not None,
                "has_threshold_override": filters["score_threshold"] is not None,
            },
            state,
        )

        out: Dict[str, Any] = {
            "search_query": query,
            "query_filters": to_json(filters),
        }
        if notes:
            out["intake_notes"] = to_json(notes)
        return out
