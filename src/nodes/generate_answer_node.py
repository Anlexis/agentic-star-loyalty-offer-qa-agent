"""AgentCore Platform v1.0"""

# RET-C2-087 - GenerateAnswerNode
# Domain node 4: assemble the grounded answer from the ranked KB passages.
#
# v1 is DETERMINISTIC (no live LLM call): the answer is rule-assembled from
# the ranked passages only - a lead sentence plus one cited point per
# passage, each carrying a numbered citation marker [n]. Nothing outside the
# ranked_documents input reaches the answer body, so the output is grounded
# by construction. The LLM synthesis upgrade seam is documented in
# docs/02_design.md ("v1 Implementation Note - LLM synthesis") and
# config/prompts/answer_synthesis_prompt.md: a v2 node swaps the assembly
# for an LLM call over the same input and emits the same state contract.
#
# Design decision: this node does NOT interpolate the caller's question into
# the answer body. A loyalty question can carry a member id or a claimed points
# balance that the identifier screen redacted imperfectly; never echoing the
# question at all removes that leak path rather than narrowing it.
#
# The same decision is what makes the output boundary checkable. Because every
# sentence and every figure in the answer comes from a retrieved passage, the
# node publishes those passages as `grounding_excerpts`, and the output gate
# can then verify that no number in the rendered answer came from anywhere
# else. See src/nodes/post_process_node.py.
#
# This node also records `answer_category` - the knowledge-base category of the
# top-ranked passage - so the outer graph can surface it as a structured field
# without re-deriving it.
#
# Wired by the inner graph (DomainWorkflowGraph).
# Returns only changed state keys (partial dict).

from typing import Any, ClassVar, Dict, List, Optional

from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_state import AgentState
from framework.schemas.trust_level import TrustLevel
from shared.utils.audit_logger import emit_trace_event
from src.schemas.state import from_json, to_json

# Answer body used when no KB passage cleared the relevance threshold.
_NO_COVERAGE_ANSWER = (
    "The loyalty program knowledge base does not contain sufficient coverage "
    "to answer this question. Rephrase the query with more specific terms "
    "about points, redemption, expiry, or promotions, or escalate to member "
    "services for a manual review."
)

# Generic lead line - deliberately does NOT quote the caller's query (see the
# S-2 design decision above).
_LEAD_LINE = "Based on the seeded loyalty-program knowledge base, the following " "passages answer your question:"

# Cited excerpt length per passage inside the answer body.
_POINT_EXCERPT_CHARS = 240


def _first_sentences(text: str, limit: int) -> str:
    """Trim an excerpt at a sentence boundary where possible, else hard-cap."""
    text = text.strip()
    if len(text) <= limit:
        return text
    cut = text[:limit]
    period = cut.rfind(". ")
    if period > limit // 2:
        return cut[: period + 1]
    return cut.rstrip() + "..."


class GenerateAnswerNode(FunctionNode):
    """Rule-based grounded answer assembly with numbered citations.

    Input state keys:
        ranked_documents: JSON list of surviving passages (from RerankFilterNode)

    Output state keys (partial dict):
        grounded_answer:    answer body with [n] citation markers
        citations:          JSON list [{ref, id, title, source}]
        answer_category:    knowledge-base category of the top passage, or None
        grounding_excerpts: JSON list of the passage strings the answer was
                            assembled from - the evidence the output gate
                            checks the rendered figures against
    """

    required_trust_level: ClassVar[TrustLevel] = TrustLevel.ANONYMOUS

    def execute(self, state: AgentState) -> Dict[str, Any]:
        ranked: List[Dict[str, Any]] = from_json(state.get("ranked_documents"), []) or []

        citations: List[Dict[str, Any]] = []
        grounding: List[str] = []
        answer_category: Optional[str] = None

        if not ranked:
            grounded_answer = _NO_COVERAGE_ANSWER
        else:
            top = ranked[0]
            if isinstance(top, dict):
                raw_category = top.get("category")
                if isinstance(raw_category, str) and raw_category.strip():
                    answer_category = raw_category.strip()

            lines: List[str] = [_LEAD_LINE, ""]
            for ref, doc in enumerate(ranked, start=1):
                if not isinstance(doc, dict):
                    continue
                title = str(doc.get("title", "")).strip()
                excerpt = _first_sentences(str(doc.get("excerpt", "")), _POINT_EXCERPT_CHARS)
                lines.append(f"[{ref}] {title}: {excerpt}")
                grounding.append(f"{doc.get('id', '')} {title} {excerpt} {doc.get('source', '')}")
                citations.append(
                    {
                        "ref": ref,
                        "id": str(doc.get("id", "")),
                        "title": title,
                        "source": str(doc.get("source", "")),
                    }
                )
            grounded_answer = "\n".join(lines)

        # S-4 domain audit: grounded answer assembled. Counts and booleans
        # only - no query text, no answer body content.
        emit_trace_event(
            "generate_answer_complete",
            {
                "citation_count": len(citations),
                "answer_chars": len(grounded_answer),
                "no_coverage": not ranked,
            },
            state,
        )

        return {
            "grounded_answer": grounded_answer,
            "citations": to_json(citations),
            "answer_category": answer_category,
            "grounding_excerpts": to_json(grounding),
        }
