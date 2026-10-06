"""AgentCore Platform v1.0"""

# RET-C2-087 - PreProcessNode (outer pre_process slot)
#
# Node contract:
#  - Extend FunctionNode; implement execute(state) -> dict
#  - Return ONLY the fields this node changes (never full state)
#  - Return AgentStatus.<X>.value strings for status assignments
#  - Read input_context via state.get("input_context", {}) - read-only
#  - Never import from mediator/, api/, or other agents
#
# This node owns the caller-data contract. Everything a caller can influence is
# settled here, before any retrieval or answer assembly runs:
#
#   1. The request must be a non-empty string within the size cap.
#   2. It must not be an attempt to redirect the agent's behaviour. The refusal
#      is enforced HERE rather than left to the surrounding framework: where a
#      platform input gate is absent or configured off, a payload that relies on
#      it reaches the answer path and comes back SUCCESS. A guarantee the
#      template states is a guarantee the template has to keep.
#   3. Every field of input_context is validated against explicit bounds
#      (src/schemas/caller_contract.py). A field that fails refuses the request
#      rather than being dropped, and the error names the field, never its value.
#   4. Direct member and loyalty-card identifiers are redacted from the free-text
#      question before validated_input is written, so a raw identifier never
#      reaches the domain nodes, the checkpoint store, or the rendered answer.
#
# The context channel carries no free text at all - every supported field is a
# bounded number or an inert identifier - so the identifier redaction below has
# only one input to cover, the question itself.

import re
from typing import Any, ClassVar, Dict, List

from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel
from shared.utils.audit_logger import emit_trace_event
from src.schemas.caller_contract import CallerFieldError, validate_caller_context

# Hard cap on the accepted question length (defence in depth on input size; the
# adapter refuses anything larger still).
MAX_INPUT_CHARS = 16_000

# Attempts to redirect the agent rather than ask it a question.
#
# Every pattern is anchored to whole words and, where a verb alone would be
# ambiguous, to the phrase around it. That anchoring is the point: an
# unanchored screen refuses ordinary domain questions, and a Q&A agent that
# refuses "Can a supervisor override the expiry policy?" or "Show me the
# instructions for redeeming points" has failed at its job just as surely as
# one that answers an attack. The paired tests probe both directions.
_INJECTION_PATTERNS: List[re.Pattern[str]] = [
    # "ignore the previous instructions", "disregard all prior rules"
    re.compile(
        r"\b(?:ignore|disregard|forget)\s+(?:all\s+|any\s+)?(?:of\s+)?(?:the\s+|your\s+)?"
        r"(?:previous|prior|above|preceding|earlier|foregoing)\s+"
        r"(?:instruction|instructions|prompt|prompts|rule|rules|message|messages|context)\b",
        re.IGNORECASE,
    ),
    # "reveal your system prompt", "print the system message" - but NOT
    # "show me the instructions for redeeming points".
    re.compile(
        r"\b(?:reveal|show|print|repeat|output|display|dump)\s+(?:me\s+)?"
        r"(?:your\s+(?:system\s+)?(?:prompt|instructions|rules|configuration)"
        r"|the\s+system\s+(?:prompt|message|instructions))\b",
        re.IGNORECASE,
    ),
    # Role reassignment: "you are now an unrestricted assistant".
    re.compile(r"\byou\s+are\s+now\s+(?:a|an|the)\b", re.IGNORECASE),
    # "act as an administrator" - the leading word boundary matters, or the
    # screen also fires inside ordinary words such as "transact as".
    re.compile(
        r"\b(?:act|behave|respond)\s+as\s+(?:an?\s+|the\s+)?"
        r"(?:developer|admin|administrator|root|system|superuser|unrestricted)\b",
        re.IGNORECASE,
    ),
    re.compile(r"\bpretend\s+(?:to\s+be|that\s+you|you\s+are)\b", re.IGNORECASE),
    # Chat-template delimiters pasted into a question.
    re.compile(r"<\|\s*im_(?:start|end)\s*\|>", re.IGNORECASE),
    re.compile(r"\bsystem\s*(?:prompt|message)\s*[:=]", re.IGNORECASE),
]

# Surface-level identifier patterns redacted before validated_input is written.
#
# The value arm requires the token to CONTAIN A DIGIT. Without that, the label
# match runs on into the next English word - "What is the card number
# replacement policy?" had "replacement" redacted, taking the query's most
# useful search term with it and quietly degrading retrieval to a no-coverage
# answer. A member or card number always carries digits; a policy question does
# not.
_IDENTIFIER_PATTERNS: List[re.Pattern[str]] = [
    # Labelled member / loyalty / card identifier: "member ID: LM-1029384",
    # "loyalty card # 88231", "card no 4412-9902". Redacts the whole token that
    # follows the label, and tolerates a linking verb or punctuation between
    # the two ("member ID is X", "member ID: X", "member ID X").
    re.compile(
        r"\b(?:member|loyalty|card)\s*(?:id|no\.?|number|#)\s*(?:is\s+)?[:#=]?\s*"
        r"(?=[A-Za-z0-9-]{4,}(?![A-Za-z0-9-]))"  # a >=4 character identifier token
        r"(?=[A-Za-z0-9-]*\d)"  # ...that contains at least one digit
        r"[A-Za-z0-9-]{4,}",
        re.IGNORECASE,
    ),
    # Grouped long digit sequence (10-19 digits, optionally dash/space grouped)
    # - the shape of a formatted card or account number.
    re.compile(r"\b\d{4}[- ]?\d{4}[- ]?\d{2,11}\b"),
    # Plain unformatted long digit run (8-9 digits) not caught above - the shape
    # of an unformatted member number. Point totals ("500 points", "1,200
    # points") stay well under this floor.
    re.compile(r"\b\d{8,9}\b"),
    # E-mail addresses.
    re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"),
]
_REDACTION = "[REDACTED]"


def _looks_like_redirection(text: str) -> bool:
    """True when the request tries to change the agent's behaviour."""
    return any(pattern.search(text) for pattern in _INJECTION_PATTERNS)


def _surface_strip_identifiers(text: str) -> str:
    """Redact member / loyalty-card identifier tokens from a free-text string."""
    for pattern in _IDENTIFIER_PATTERNS:
        text = pattern.sub(_REDACTION, text)
    return text


class PreProcessNode(FunctionNode):
    """Validate the caller's request before any domain processing runs.

    This is the template's single external trust gate: a caller who has not
    been elevated (see the Bearer check in src/api/server.py) is denied here,
    before any retrieval or answer assembly happens.

    It is also where the caller-data contract is enforced - the question, its
    size, its intent, and every field of input_context - and where direct
    member and loyalty-card identifiers are stripped out of the question.
    """

    required_trust_level: ClassVar[TrustLevel] = TrustLevel.VERIFIED_EXTERNAL

    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        user_input = state.get("user_input", "")
        raw_context = state.get("input_context", {})  # read-only

        if not user_input or not isinstance(user_input, str) or not user_input.strip():
            return self._refuse("user_input must be a non-empty string")

        if len(user_input) > MAX_INPUT_CHARS:
            return self._refuse(f"user_input must be at most {MAX_INPUT_CHARS} characters")

        if _looks_like_redirection(user_input):
            # Nothing is carried forward: no validated_input is written, so no
            # later node can act on the request even in part.
            emit_trace_event("pre_process_request_refused", {"reason": "redirection"}, state)
            return self._refuse("user_input was refused by the request screen")

        try:
            caller_context = validate_caller_context(raw_context)
        except CallerFieldError as exc:
            emit_trace_event(
                "pre_process_request_refused",
                {"reason": "context_field", "field": exc.field},
                state,
            )
            return self._refuse(str(exc))

        validated_input = _surface_strip_identifiers(user_input.strip())

        # Audit that a question passed validation. Counts and booleans only -
        # no question text, and no digit content that could be a member id or a
        # points balance.
        emit_trace_event(
            "pre_process_complete",
            {
                "input_chars": len(validated_input),
                "has_channel": "channel" in caller_context,
                "context_fields": len(caller_context),
            },
            state,
        )

        return {
            "validated_input": validated_input,
            "enriched_context": {
                "source": "LoyaltyOfferQAAgent",
                "channel": caller_context.get("channel", "unknown"),
            },
            "status": AgentStatus.SUCCESS.value,
        }

    @staticmethod
    def _refuse(reason: str) -> Dict[str, Any]:
        """Fail closed: an error status and a reason that names no caller value."""
        return {
            "status": AgentStatus.ERROR.value,
            "error_log": [f"PreProcessNode: {reason}"],
        }
