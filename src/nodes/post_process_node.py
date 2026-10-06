"""AgentCore Platform v1.0"""

# RET-C2-087 - PostProcessNode (outer post_process slot; the output boundary)
#
# Two INDEPENDENT layers run over everything this agent is about to return, and
# each raises its own audit event so a block is attributable to one of them:
#
#   1. Credential redaction - a credential-shaped value must never reach a
#      caller. The scan walks nested dict / list / tuple structures, not just
#      top-level strings: a token one level down inside a returned payload
#      escapes a top-level-only scan entirely.
#
#   2. Numeric fidelity - the invariant this template actually states. Every
#      figure in an answer must be a figure from the knowledge base, reproduced
#      exactly.
#
# On the second layer, and why it is not the usual rounding grid: several
# templates in this family render computed monetary aggregates, and enforce a
# published schema by SNAPPING every amount to the nearest 1,000. That is the
# wrong instrument here, and shipping it would break the product. This agent
# renders no aggregates at all - it answers policy questions by quoting the
# programme's own text, where the exact figure IS the answer. "The minimum
# redemption amount is 500 points" rounded to the nearest 1,000 becomes a
# statement that is false, and a member acting on it is misinformed. The same
# goes for "1 point per 100 yen", the tier thresholds, and the 1.5x and 2x
# earning multipliers.
#
# So the invariant is fidelity, not rounding, and the gate VERIFIES rather than
# rewrites: every numeric run in the rendered answer must appear in the
# passages the answer was assembled from. A rewriting gate can corrupt what it
# touches - decimal fractions and identifiers being the usual casualties - and
# this one cannot, because it never modifies the text. What it does catch is a
# number that came from anywhere other than the knowledge base, which is the
# leak the "never echo the caller's question" rule exists to prevent.
#
# The module-level `_security_gate_output()` below is also imported and reused
# by `LoyaltyOfferQAAgent.get_output()` (src/graph/graph.py) before it surfaces
# the structured `citations` / `answer_category` fields - the same gates
# applied a second time, at the outer envelope boundary, fail-closed.

import json
import re
from typing import Any, ClassVar, Dict, List

from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel
from shared.utils.audit_logger import emit_trace_event

# Credential-shaped strings that must never reach the caller.
#
# This set is the UNION of the framework's own detector and the shapes it does
# not know, never a subset of it. A local set narrower than the framework's is
# not merely weaker - it is a bypass. A value this scan misses reaches the
# returned delta, the framework's @final S-3 gate then RAISES on it, and
# BaseNode.__call__ discards the node's whole delta, this node's clearing with
# it. Catching it here instead means the block is contained in state as well as
# in the envelope.
#
# The framework's six patterns (framework/security/credential_detector.py) are
# mirrored in the first six alternatives. The seventh is ours - any scheme
# carrying inline credentials, not only the four database schemes it knows - as
# is the keyword-assignment form below. Widening is safe; narrowing is the
# bypass. tests/unit/test_post_process_node.py pins one sample per framework
# credential type against this scan.
_CREDENTIAL_LIKE_RE = re.compile(
    r"sk_(?:live|test)_[A-Za-z0-9]{16,}"
    r"|sk-[A-Za-z0-9]{20,}"
    r"|eyJ[A-Za-z0-9._-]{10,}"
    r"|AKIA[A-Z0-9]{16}"
    r"|Bearer\s+[A-Za-z0-9._-]{16,}"
    r"|(?:postgresql|mysql|mongodb|redis)://[^\s]{8,}"
    # A connection string carries its credentials inline. This template used to
    # leave the form to the surrounding framework's own scan; owning it here
    # means the guarantee holds wherever the agent runs, and it keeps the
    # ordering below honest - a credential is attributed to the credential
    # layer rather than to whichever later check happens to notice it first.
    r"|[a-z][a-z0-9+.-]*://[^\s/@:]+:[^\s/@]+@"
)

# Keyword-assignment secrets (`password=...`, `api_key: ...`) match neither the
# patterns above nor the framework's own detector. The value arm requires 6+
# non-space characters so prose such as "password rules" never trips it.
_SECRET_ASSIGNMENT_RE = re.compile(
    r"\b(?:password|passwd|pwd|secret|api[_-]?key|access[_-]?key)\s*[:=]\s*\S{6,}",
    re.IGNORECASE,
)

# A numeric run, together with the grouping punctuation that belongs to it.
#
# The character class is the alphabet this template's output actually renders
# numbers in - comma-grouped thresholds ("5,000"), decimal multipliers ("1.5x"),
# hyphenated ranges ("5-7 business days"), and knowledge-base entry identifiers
# ("kb-001", "PTR-2026-014"). A narrower class splits those into fragments and
# then reports the fragments; read the alphabet the output uses rather than
# assuming one.
_NUMBER_RUN_RE = re.compile(r"\d[\d,._-]*\d|\d")


def _security_gate_output(content: Any) -> "list[str]":
    """Scan an outgoing payload for credential-shaped content.

    Module-level and called inline from PostProcessNode.execute(); the
    framework's own `_security_gate_output` / `_security_gate_input` are final
    and cannot be overridden (see
    tests/unit/test_framework_compliance_tc06_tc07.py), and an
    `_extra_security_gate_output` instance method would be auto-wrapped into
    the graph chain, where a clean (None) return breaks the invoke chain.

    Recurses into dict / list / tuple, so a credential-shaped value nested
    inside a structured response - not only a top-level string - is still
    caught.
    """
    violations: list[str] = []

    def _scan(node: Any, path: str) -> None:
        if isinstance(node, str):
            if _CREDENTIAL_LIKE_RE.search(node) or _SECRET_ASSIGNMENT_RE.search(node):
                violations.append(f"output gate: credential-like value at {path}")
        elif isinstance(node, dict):
            for key, value in node.items():
                _scan(value, f"{path}.{key}")
        elif isinstance(node, (list, tuple)):
            for i, value in enumerate(node):
                _scan(value, f"{path}[{i}]")
        # other scalar types (int / float / bool / None) carry no credential risk

    _scan(content, "output")
    return violations


def _enforce_numeric_fidelity(content: Any, grounding: str, citation_count: int = 0) -> "list[str]":
    """Check that every figure in an outgoing payload came from the corpus.

    `grounding` is the concatenation of the knowledge-base passages the answer
    was assembled from. A numeric run in the payload passes when it appears
    verbatim in that text, or when it is one of the citation markers this
    template generates ([1], [2], ...). Anything else is a figure with no
    source - a caller value that reached the output, or a number the pipeline
    invented - and the answer is withheld.

    Recurses the same way the credential scan does, so the structured fields
    are covered as well as the rendered string. This function never modifies
    the payload; it only reports.
    """
    violations: list[str] = []
    allowed_refs = {str(n) for n in range(1, citation_count + 1)}

    def _scan(node: Any, path: str) -> None:
        if isinstance(node, str):
            for run in _NUMBER_RUN_RE.findall(node):
                if run in grounding or run in allowed_refs:
                    continue
                # The run itself is withheld from the message: an ungrounded
                # figure is exactly the sort of value that must not be echoed.
                violations.append(f"output gate: ungrounded figure at {path}")
                return
        elif isinstance(node, dict):
            for key, value in node.items():
                _scan(value, f"{path}.{key}")
        elif isinstance(node, (list, tuple)):
            for i, value in enumerate(node):
                _scan(value, f"{path}[{i}]")

    _scan(content, "output")
    return violations


# A blocked answer must not survive anywhere the caller can reach it.
#
# AgentBaseGraph.get_output() resolves the caller-facing payload as
# `state.get("formatted_output") or state.get("result")`. It applies no status
# check, so returning ERROR while leaving the pre-gate answer in state["result"]
# ships that answer inside the error envelope. Omitting `formatted_output` does
# not help - and neither does an empty string, which is falsy and therefore
# selects the same fallback. The replacement below is TRUTHY on purpose.
#
# The clearing has to be explicit for the same reason: LangGraph merges a node's
# partial dict into state, so a key that is merely absent from the returned
# delta keeps its previous value. Every field that carried answer text is
# returned with an emptied value, not left out.
BLOCKED_OUTPUT_NOTICE = "The response was withheld by the output gate."

# The outer-state fields that carry answer text or a payload assembled from it.
# These are exactly the caller-facing keys LoyaltyOfferQAGraphNode.merge_output()
# writes (src/graph/graph.py) minus `status`; tests/unit/test_post_process_node.py
# pins that correspondence so a field added to the merge cannot quietly skip the
# clearing. Everything else in outer state is either framework-managed or the
# caller's own request text, neither of which is gate-refused output.
_OUTPUT_BEARING_FIELDS = (
    "result",
    "loyalty_answer",
    "citations",
    "answer_category",
    "grounding_excerpts",
)


def blocked_output_state() -> Dict[str, Any]:
    """Return the cleared state delta that replaces a refused answer.

    Every output-bearing field is present and emptied - present because an
    omitted key leaves the old value in state, emptied because the value is
    the thing being withheld. `formatted_output` carries a fixed notice that
    is constant text, never assembled from the state just cleared.
    """
    cleared: Dict[str, Any] = {field: None for field in _OUTPUT_BEARING_FIELDS}
    cleared["formatted_output"] = BLOCKED_OUTPUT_NOTICE
    return cleared


def grounding_text(state: Any) -> str:
    """Return the corpus text an answer in *state* was assembled from."""
    raw = state.get("grounding_excerpts")
    if not raw:
        return ""
    try:
        excerpts = json.loads(raw) if isinstance(raw, str) else raw
    except (json.JSONDecodeError, TypeError, ValueError):
        return ""
    if not isinstance(excerpts, list):
        return ""
    return " ".join(str(part) for part in excerpts)


class PostProcessNode(FunctionNode):
    """The output boundary: scan the final answer before it reaches the caller.

    Reads state["result"] - the rendered answer merged out of the inner graph -
    and applies both gates. On a pass, `formatted_output` is the flat rendered
    answer string; the structured `citations` / `answer_category` fields are
    surfaced separately by `LoyaltyOfferQAAgent.get_output()`
    (src/graph/graph.py), which runs the same two gates over them.

    On a block, the node does not merely decline to write `formatted_output`:
    it returns blocked_output_state(), which clears every output-bearing field
    and substitutes a fixed notice. Refusing to write the field is not
    containment, because the base envelope falls back to the ungated
    state["result"] whenever `formatted_output` is falsy or absent.

    required_trust_level: VERIFIED_EXTERNAL - an outer backbone gate slot, same
    posture as pre_process.
    """

    required_trust_level: ClassVar[TrustLevel] = TrustLevel.VERIFIED_EXTERNAL

    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        result = state.get("result", "")
        formatted_output = result

        violations: List[str] = _security_gate_output(formatted_output)
        if violations:
            emit_trace_event(
                "post_process_credential_blocked",
                {"violation_count": len(violations)},
                state,
            )
            return {
                **blocked_output_state(),
                "status": AgentStatus.ERROR.value,
                "error_log": violations,
            }

        citations = state.get("citations")
        try:
            citation_count = len(json.loads(citations)) if isinstance(citations, str) else 0
        except (json.JSONDecodeError, TypeError, ValueError):
            citation_count = 0

        ungrounded = _enforce_numeric_fidelity(formatted_output, grounding_text(state), citation_count)
        if ungrounded:
            emit_trace_event(
                "post_process_fidelity_blocked",
                {"violation_count": len(ungrounded)},
                state,
            )
            return {
                **blocked_output_state(),
                "status": AgentStatus.ERROR.value,
                "error_log": ungrounded,
            }

        # Audit that a response was shaped - no answer content in the payload.
        emit_trace_event(
            "post_process_complete",
            {"has_output": bool(result)},
            state,
        )

        return {
            "formatted_output": formatted_output,
            "status": AgentStatus.SUCCESS.value,
        }
