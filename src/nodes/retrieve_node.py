"""AgentCore Platform v1.0"""

# RET-C2-087 - RetrieveNode
# Domain node 2: deterministic keyword retrieval over the seeded loyalty
# knowledge base (config/kb/loyalty_kb.json). v1 is fully deterministic - no
# embedding model, no vector store, no network call (no live loyalty-platform
# or POS integration anywhere - docs/01_proposal.md "Scope Revision"); the
# retrieval contract (retrieved_documents JSON) is store-agnostic so a later
# vector-store upgrade only swaps this node's internals.
#
# Config: reads `top_k` and `kb_path` EXCLUSIVELY from the state-seeded
# `retrieval_config` field, which DomainWorkflowGraph._extra_initial_state()
# publishes from the values declared in config/config.yaml. execute() takes no
# `config` parameter, so this state field is the only route configuration
# reaches the node. The block is re-validated here rather than trusted: a
# threshold or candidate cap that is out of range - or not a finite number -
# refuses the request instead of quietly changing every answer.
#
# Wired by the inner graph (DomainWorkflowGraph).
# Returns only changed state keys (partial dict).

import json
import re
from pathlib import Path
from typing import Any, ClassVar, Dict, List

from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_state import AgentState
from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel
from shared.utils.audit_logger import emit_trace_event
from src.graph.runtime_config import DEFAULT_RETRIEVAL, validate_retrieval_block
from src.schemas.caller_contract import CallerFieldError
from src.schemas.state import from_json, to_json

# Repo root: src/nodes/retrieve_node.py -> parents[2].
_REPO_ROOT = Path(__file__).resolve().parents[2]

# Minimal stopword set for query tokenisation (deterministic, no NLP deps).
_STOPWORDS = frozenset(
    {
        "the",
        "a",
        "an",
        "and",
        "or",
        "of",
        "to",
        "in",
        "on",
        "for",
        "is",
        "are",
        "be",
        "with",
        "under",
        "what",
        "which",
        "when",
        "how",
        "do",
        "does",
        "must",
        "should",
        "before",
        "after",
        "by",
        "at",
        "from",
        "that",
        "this",
        "it",
        "as",
        "was",
        "were",
        "can",
        "may",
        "any",
        "my",
        "i",
    }
)

_TOKEN_RE = re.compile(r"[a-z0-9]+")

# Per-field match weights: a query token found in the title counts more than
# one found only in the body content.
_TITLE_WEIGHT = 1.0
_TAG_WEIGHT = 0.8
_CONTENT_WEIGHT = 0.5

# Excerpt length carried into retrieved_documents (keeps State small).
_EXCERPT_CHARS = 400


def _tokenize(text: str) -> List[str]:
    """Lowercase alphanumeric tokens, stopwords and 1-2 char noise removed."""
    return [t for t in _TOKEN_RE.findall(text.lower()) if len(t) > 2 and t not in _STOPWORDS]


def _resolve_retrieval_config(state: AgentState) -> Dict[str, Any]:
    """Effective retrieval config: the state-seeded block, validated.

    No `config` parameter is read here - execute() takes only `state`. Raises
    CallerFieldError when the seeded block carries a value outside its bounds,
    which the caller turns into a refusal.
    """
    effective = dict(DEFAULT_RETRIEVAL)  # local copy - never mutate the module default
    from_state = from_json(state.get("retrieval_config"), None)
    if isinstance(from_state, dict):
        effective.update(from_state)
    return validate_retrieval_block(effective)


def _load_kb(kb_path: str) -> "tuple[List[Dict[str, Any]], List[str]]":
    """Load the seeded KB JSON. Missing / malformed file degrades gracefully."""
    notes: List[str] = []
    path = Path(kb_path)
    if not path.is_absolute():
        path = _REPO_ROOT / path
    try:
        entries = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, ValueError):
        notes.append(f"RetrieveNode: knowledge base not readable at {kb_path}.")
        return [], notes
    if not isinstance(entries, list):
        notes.append("RetrieveNode: knowledge base root must be a JSON list.")
        return [], notes
    return [e for e in entries if isinstance(e, dict)], notes


def _score_entry(entry: Dict[str, Any], query_tokens: List[str]) -> float:
    """Per-entry relevance: best field-weight per query token, averaged."""
    if not query_tokens:
        return 0.0
    title_tokens = set(_tokenize(str(entry.get("title", ""))))
    tag_tokens = set(_tokenize(" ".join(str(t) for t in entry.get("tags", []))))
    content_tokens = set(_tokenize(str(entry.get("content", ""))))
    total = 0.0
    for token in query_tokens:
        if token in title_tokens:
            total += _TITLE_WEIGHT
        elif token in tag_tokens:
            total += _TAG_WEIGHT
        elif token in content_tokens:
            total += _CONTENT_WEIGHT
    return round(total / len(query_tokens), 4)


class RetrieveNode(FunctionNode):
    """Score the seeded loyalty KB against the search query and emit candidates.

    Input state keys:
        search_query:     normalised query (from InputValidateNode)
        query_filters:    JSON dict with optional category filter
        retrieval_config: forwarded manifest retrieval block (JSON)

    Output state keys (partial dict):
        retrieved_documents: JSON list of scored candidates (score desc)
        intake_notes:        (on KB anomalies) JSON list[str]
    """

    required_trust_level: ClassVar[TrustLevel] = TrustLevel.ANONYMOUS

    def execute(self, state: AgentState) -> Dict[str, Any]:
        query = state.get("search_query") or state.get("validated_input") or state.get("user_input", "")
        filters = from_json(state.get("query_filters"), {}) or {}
        try:
            retrieval_cfg = _resolve_retrieval_config(state)
        except CallerFieldError as exc:
            return {
                "status": AgentStatus.ERROR.value,
                "error_log": [f"RetrieveNode: {exc}"],
            }

        top_k = int(retrieval_cfg["top_k"])
        entries, notes = _load_kb(str(retrieval_cfg["kb_path"]))

        category = filters.get("category")
        if category:
            entries = [e for e in entries if str(e.get("category", "")).lower() == str(category).lower()]

        query_tokens = _tokenize(query if isinstance(query, str) else "")

        candidates: List[Dict[str, Any]] = []
        for entry in entries:
            score = _score_entry(entry, query_tokens)
            if score <= 0.0:
                continue
            candidates.append(
                {
                    "id": str(entry.get("id", "")),
                    "title": str(entry.get("title", "")),
                    "category": str(entry.get("category", "")),
                    "source": str(entry.get("source", "")),
                    "score": score,
                    "excerpt": str(entry.get("content", ""))[:_EXCERPT_CHARS],
                }
            )

        # Deterministic ordering: score desc, then id asc for stable ties.
        candidates.sort(key=lambda c: (-c["score"], c["id"]))
        # Keep a candidate pool wider than top_k - RerankFilterNode makes
        # the final cut after the category boost + threshold.
        pool_size = max(top_k * 3, 10)
        candidates = candidates[:pool_size]

        # S-4 domain audit: retrieval pass completed. Counts and config
        # knobs only - no query text, no category value from the caller.
        emit_trace_event(
            "retrieve_complete",
            {
                "candidates": len(candidates),
                "kb_entries": len(entries),
                "query_tokens": len(query_tokens),
                "top_k": top_k,
            },
            state,
        )

        out: Dict[str, Any] = {"retrieved_documents": to_json(candidates)}
        if notes:
            # Append to (never clobber) the notes accumulated upstream.
            prior = from_json(state.get("intake_notes"), []) or []
            out["intake_notes"] = to_json(list(prior) + notes)
        return out
