# Answer Synthesis Prompt — RET-C2-087 (v2 LLM upgrade seam)

> **v1 does NOT use this prompt at runtime.** v1 of `GenerateAnswerNode` is
> deterministic (rule-based grounded assembly over `ranked_documents`); no
> node reads this file. It documents the synthesis contract for the v2 LLM
> upgrade described in `docs/02_design.md` ("v1 Implementation Note — LLM
> synthesis"), so the v2 swap changes only the inside of
> `GenerateAnswerNode.execute()`.

## Contract (v2 GenerateAnswerNode)

- **Input:** the same `ranked_documents` JSON (id / title / category / source /
  score / excerpt) the v1 node reads. The v2 node must NOT be given the raw
  caller query text as a free-form prompt input beyond what is already
  present in `ranked_documents` — this template's queries may carry a member
  ID or a claimed points balance that the S-2 surface screen imperfectly
  redacted (`docs/02_design.md` "Security Gates"), so the synthesis prompt
  stays grounded in KB passages only, never in the raw question.
- **Output:** the same state contract — `grounded_answer` (str, with numbered
  `[n]` citation markers), `citations` (JSON list of
  `{ref, id, title, source}`), and `answer_category` (str | None, the
  top-cited passage's KB category).
- **Grounding rule:** every factual statement in the answer must be traceable
  to one of the supplied passages via a `[n]` marker; content not present in
  the passages must not be asserted. This applies with extra force to
  promotional claims — an offer's terms, eligibility, or expiry must never be
  asserted beyond what the cited KB passage states (misleading-representation
  risk on promotional claims).
- **No-coverage rule:** when no passage supports the question, say so and
  recommend rephrasing or escalating to member services — never answer from
  parametric knowledge, and never assert or estimate a live points balance
  (v1 and v2 both have no account/POS integration).
- **Tone:** neutral, member-service-appropriate, no individualized redemption
  promises (the v1-scope disclaimer is appended downstream by
  `OutputFormatNode`).

## Prompt template

```
You answer retail loyalty-program questions strictly from the knowledge-base
passages provided below (program rules, offer catalog, redemption catalog).

Passages (each with a reference number):
{ranked_documents}

Rules:
1. Use ONLY the passages above. If they do not answer the question, say the
   knowledge base has insufficient coverage and stop.
2. Mark every factual statement with the [n] reference of its passage.
3. Never state or estimate a live points balance, account status, or
   transaction history — no account lookup exists in this system.
4. Never assert a promotional offer's terms, eligibility, or expiry beyond
   what the cited passage states.
5. Keep the answer under 300 words.
```

## Manifest coupling

The `llm` block in `config/agent.yaml` (`temperature`, `max_tokens`) is already
forwarded to the inner graph via
`LoyaltyOfferQAGraphNode._parent_config()` under
`config["configurable"]["llm"]`; the v2 node reads it from there.
