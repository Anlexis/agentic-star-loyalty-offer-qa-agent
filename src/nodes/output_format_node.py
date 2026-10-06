"""AgentCore Platform v1.0"""

# RET-C2-087 - OutputFormatNode
# Domain node 5 (terminal): compose the final formatted answer - the grounded
# answer body, the Sources list, and the standing v1-scope disclaimer. The
# disclaimer is part of THIS node's domain output contract, not of the outer
# post_process slot (post_process only gates, it does not compose).
#
# Wired by the inner graph (DomainWorkflowGraph). get_output() of the inner
# graph surfaces formatted_answer + status (+ citations / answer_category) to
# the outer merge_output().
# Returns only changed state keys (partial dict).

from typing import Any, ClassVar, Dict, List

from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_state import AgentState
from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel
from shared.utils.audit_logger import emit_trace_event
from src.schemas.state import from_json

# Standing v1-scope disclaimer - appended to EVERY answer this template
# emits (docs/01_proposal.md "Scope Revision" / docs/02_design.md "v1 Scope
# Disclaimer"): no live account/points-balance lookup happens in v1, and
# promotional terms can change, so the reader must be told to verify both.
_SCOPE_DISCLAIMER = (
    "This answer is generated from the seeded loyalty-program knowledge base "
    "for informational purposes only. It does not reflect your live points "
    "balance or account history - no account lookup is performed in this "
    "version. Verify redemption eligibility and current promotion terms with "
    "member services before redeeming; all promotional offers are subject to "
    "change and their published terms and conditions."
)


class OutputFormatNode(FunctionNode):
    """Compose the final answer: body + sources + v1-scope disclaimer.

    Input state keys:
        grounded_answer: answer body with [n] citation markers
        citations:       JSON list [{ref, id, title, source}]

    Output state keys (partial dict):
        formatted_answer: final rendered answer string
        status:           AgentStatus.SUCCESS.value (plain string — review finding 15:
                          never write the bare enum to State)
    """

    required_trust_level: ClassVar[TrustLevel] = TrustLevel.ANONYMOUS

    def execute(self, state: AgentState) -> Dict[str, Any]:
        grounded_answer = state.get("grounded_answer") or ("No answer is available for this request.")
        citations: List[Dict[str, Any]] = from_json(state.get("citations"), []) or []

        lines: List[str] = []
        lines.append("# Loyalty Program Q&A Result")
        lines.append("")
        lines.append(grounded_answer)
        lines.append("")
        lines.append("## Sources")
        if citations:
            for citation in citations:
                if not isinstance(citation, dict):
                    continue
                ref = citation.get("ref", "?")
                title = str(citation.get("title", "")).strip()
                source = str(citation.get("source", "")).strip()
                suffix = f" ({source})" if source else ""
                lines.append(f"- [{ref}] {title}{suffix}")
        else:
            lines.append("- none (no knowledge-base passage cleared the relevance threshold)")
        lines.append("")
        lines.append("---")
        lines.append("")
        lines.append(f"*{_SCOPE_DISCLAIMER}*")

        formatted_answer = "\n".join(lines)

        # S-4 domain audit: final answer composed (disclaimer attached).
        emit_trace_event(
            "output_format_complete",
            {
                "answer_chars": len(formatted_answer),
                "citation_count": len(citations),
            },
            state,
        )

        return {
            "formatted_answer": formatted_answer,
            "status": AgentStatus.SUCCESS.value,
        }
