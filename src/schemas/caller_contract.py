"""AgentCore Platform v1.0"""

# RET-C2-087 - the caller-data contract.
#
# Every value a caller can influence - the free-text question, the retrieval
# overrides carried in `input_context`, and the structured parameters that may
# travel inside the request payload - is validated HERE, in one place, so the
# nodes that consume them cannot each invent their own idea of "valid".
#
# Two rules the whole module exists to enforce:
#
#   1. A caller-supplied NUMBER must be finite and inside an explicit range.
#      `float("nan")` and `float("inf")` parse successfully and arrive through
#      plain JSON, and every comparison against NaN is False - so a NaN
#      relevance threshold does not error, it silently changes the answer.
#      Clamping is not a defence either: `max(0.0, min(1.0, nan))` evaluates to
#      1.0, which installs the STRICTEST possible floor and refuses questions
#      the knowledge base actually covers. Validation fails CLOSED instead.
#
#   2. A caller-supplied STRING that can influence output is locked to an inert
#      identifier shape (lowercase letters, digits, underscore, at most 32
#      characters) and, where a fixed vocabulary exists, to that vocabulary.
#
# Errors name the FIELD and never repeat the value - an error message is an
# output channel, and echoing a rejected value back to the caller hands them
# one.

import math
import re
from typing import Any, Dict, Mapping, Optional, Tuple

# Retrieval bounds. `top_k` mirrors the candidate cap the ranking step applies;
# `score_threshold` is a relevance fraction, so it is bounded by its own units.
TOP_K_MIN = 1
TOP_K_MAX = 20
SCORE_THRESHOLD_MIN = 0.0
SCORE_THRESHOLD_MAX = 1.0

# The knowledge base's own category vocabulary (config/kb/loyalty_kb.json).
# A caller filter is matched against these values as a WHOLE value - never as a
# substring, which would let "offer" silently select "offer_catalog" and
# discard the passages that answer the question.
SUPPORTED_CATEGORIES = frozenset({"program_rules", "offer_catalog", "redemption_catalog"})

# Context fields this template consumes. Anything else is refused rather than
# ignored: a silently dropped "top-k" typo looks identical to a value that was
# honoured, and the caller has no way to tell the difference.
SUPPORTED_CONTEXT_FIELDS: Tuple[str, ...] = (
    "channel",
    "category",
    "top_k",
    "score_threshold",
)

# Inert identifier shape for any caller string that can reach an audit payload
# or the rendered answer.
_INERT_IDENTIFIER_RE = re.compile(r"^[a-z0-9_]{1,32}$")


class CallerFieldError(ValueError):
    """A caller-supplied field failed validation.

    Carries the field NAME so a caller can fix the request, and never the
    rejected value.
    """

    def __init__(self, field: str, reason: str) -> None:
        super().__init__(f"{field} {reason}")
        self.field = field
        self.reason = reason


def finite_in_range(
    value: Any,
    *,
    field: str,
    minimum: float,
    maximum: float,
    integer: bool = False,
) -> float:
    """Return *value* as a finite number inside [minimum, maximum], or raise.

    Rejects, in order: booleans (``True`` is an ``int`` in Python and would
    silently become 1), values that are not numbers at all, NaN and +/-Infinity,
    non-whole values when an integer is required, and anything outside the
    stated range.
    """
    if isinstance(value, bool):
        raise CallerFieldError(field, "must be a number, not a true/false value")

    if isinstance(value, str):
        text = value.strip()
        if not text:
            raise CallerFieldError(field, "must not be blank")
        try:
            number = float(text)
        except ValueError:
            raise CallerFieldError(field, "must be a number") from None
    elif isinstance(value, (int, float)):
        number = float(value)
    else:
        raise CallerFieldError(field, "must be a number")

    if not math.isfinite(number):
        raise CallerFieldError(field, "must be a finite number")

    if integer:
        if number != int(number):
            raise CallerFieldError(field, "must be a whole number")
        whole = int(number)
        if whole < minimum or whole > maximum:
            raise CallerFieldError(field, f"must be between {int(minimum)} and {int(maximum)}")
        return float(whole)

    if number < minimum or number > maximum:
        raise CallerFieldError(field, f"must be between {minimum} and {maximum}")
    return number


def inert_identifier(value: Any, *, field: str, allowed: Optional[frozenset[str]] = None) -> str:
    """Return *value* as a lowercase inert identifier, or raise.

    The accepted shape is ``[a-z0-9_]{1,32}``. When *allowed* is supplied the
    value must equal one of its members - a whole-value comparison, so a
    fragment of a supported name is refused rather than quietly selecting it.
    """
    if not isinstance(value, str):
        raise CallerFieldError(field, "must be text")
    text = value.strip().lower()
    if not _INERT_IDENTIFIER_RE.match(text):
        raise CallerFieldError(field, "must be 1-32 characters of a-z, 0-9 or underscore")
    if allowed is not None and text not in allowed:
        raise CallerFieldError(field, "is not one of the supported values")
    return text


# Keys the platform itself puts into input_context, not the caller. The Marketplace
# runner invokes every agent as
#     agent.invoke(message, ctx=ctx, input_context={"conversation_history": history})
# (agenticstar-agentcore, shared/bootstrap/marketplace_app.py), whatever the user typed.
# Refusing it as an unknown field refused every chat request before the question was
# read. Discarded, not validated: nothing in this pipeline reads prior turns, and
# screening a transcript would let one earlier message refuse every later one. Discarding
# adds no exposure — the backbone's first node has already copied the raw input_context
# into state before this contract runs.
PLATFORM_RESERVED_KEYS = frozenset({"conversation_history"})


def validate_caller_context(raw: Any) -> Dict[str, Any]:
    """Validate the caller's ``input_context`` mapping.

    Returns a dict holding only the fields that were supplied, each already
    coerced to its validated form. An absent context is not an error - the
    agent answers from its declared defaults - but a PRESENT field that fails
    its bounds refuses the request.
    """
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise CallerFieldError("input_context", "must be an object")
    raw = {k: v for k, v in raw.items() if k not in PLATFORM_RESERVED_KEYS}

    unsupported = [k for k in raw if k not in SUPPORTED_CONTEXT_FIELDS]
    if unsupported:
        # The count only: a context KEY is caller-controlled text as much as a
        # value is, so repeating it back would reopen the hole this closes.
        raise CallerFieldError(
            "input_context",
            f"carries {len(unsupported)} unsupported field(s); supported fields are "
            + ", ".join(SUPPORTED_CONTEXT_FIELDS),
        )

    validated: Dict[str, Any] = {}

    if "channel" in raw:
        validated["channel"] = inert_identifier(raw["channel"], field="channel")

    if "category" in raw:
        validated["category"] = inert_identifier(raw["category"], field="category", allowed=SUPPORTED_CATEGORIES)

    if "top_k" in raw:
        validated["top_k"] = int(
            finite_in_range(
                raw["top_k"],
                field="top_k",
                minimum=TOP_K_MIN,
                maximum=TOP_K_MAX,
                integer=True,
            )
        )

    if "score_threshold" in raw:
        validated["score_threshold"] = finite_in_range(
            raw["score_threshold"],
            field="score_threshold",
            minimum=SCORE_THRESHOLD_MIN,
            maximum=SCORE_THRESHOLD_MAX,
        )

    return validated
