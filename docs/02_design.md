# Template Design Specification — RET-C2-087

**Template ID:** RET-C2-087
**Template Name:** LoyaltyOfferQAAgent
**Category:** Cat 2 (multi-step domain workflow — RAG pattern)
**Industry:** RET

## Position in AgentCore Architecture

- **Agent Class:** `LoyaltyOfferQAAgent` (alias `Graph`)
- **L1 Base:** `AgentBaseGraph` (outer graph) — direct L1 inheritance, no L2 class names
- **Inner graph base:** `BaseGraph` (L1) — `DomainWorkflowGraph`
- **Pattern:** Cat 2 two-layer nested architecture (outer fixed 5-node backbone +
  `GraphNode` in the `main` slot wrapping an inner `BaseGraph` domain workflow)
- **Three-Layer Separation:**
  - State: flat `TypedDict` composition (no Pydantic — msgpack incompatible);
    structured fields stored as JSON strings via `to_json()` / `from_json()`
  - Node: L1 inheritance via `FunctionNode` (override `execute(self, state) -> dict`
    ONLY — no `config` parameter; SDK v1.0.0rc1 / CoE 2026-07-27 contract)
  - Graph: composition (`register_nodes()` for node substitution); outer
    `add_edges()` is NOT overridden

## Purpose

Retail loyalty-program Q&A agent: member (and CSR-on-behalf-of-member) questions
about points-balance rules, redemption options, expiry, and active promotions are
answered over a seeded loyalty-program knowledge base — retrieve → rerank/filter →
grounded answer with citations + a standing v1-scope disclaimer. v1 is fully
deterministic (keyword retrieval + rule-based grounded answer assembly; no live
LLM call — see the v1 Implementation Note below) and network-free: no live
loyalty-platform or POS integration is called anywhere, and the agent never
surfaces an individual member's live account balance (no account integration in
v1 — see `docs/01_proposal.md` "Scope Revision").

## Architecture Overview

### Outer backbone (AgentBaseGraph)

```
START → initialize → pre_process → main → {route} → post_process → finalize → END
                                     ↓ (retry, max 3)
                                   pre_process
```

| Slot | Class | Responsibility | required_trust_level | S-gate |
|------|-------|----------------|----------------------|--------|
| initialize | InitializeNode (framework default) | session_id, trust_level, schema_version | — (framework) | — |
| pre_process | `PreProcessNode` | S-1 validate non-empty input; S-2 surface-strip direct identifiers (member/loyalty-card IDs, long digit runs, e-mail) → `validated_input` | `TrustLevel.VERIFIED_EXTERNAL` | S-1, S-2 |
| main | `LoyaltyOfferQAGraphNode` (`GraphNode`) | delegates to inner `DomainWorkflowGraph`; maps inner `formatted_answer` → outer `result` / `loyalty_answer` | — (GraphNode delegation) | — (delegates) |
| post_process | `PostProcessNode` | S-3 output gate — module-level `_security_gate_output()` + `_enforce_numeric_fidelity()` scan `result` (recursively) → on a hit, ERROR with every output-bearing field cleared and a fixed notice in `formatted_output` | `TrustLevel.VERIFIED_EXTERNAL` | S-3 |
| finalize | FinalizeNode (framework default) | response_metadata, total_time_ms | — (framework) | — |

### Inner graph (DomainWorkflowGraph — BaseGraph, linear)

```
START → input_validate → retrieve → rerank_filter → generate_answer → output_format → END
```

All five inner domain nodes declare `required_trust_level = TrustLevel.ANONYMOUS`
(the external trust gate lives on the outer backbone gate node; a stricter inner
level would deny a real VERIFIED_EXTERNAL invoke at runtime).

| Node | Responsibility | required_trust_level | Input State | Output State |
|------|----------------|----------------------|-------------|---------------|
| `InputValidateNode` | Parse the (possibly JSON-enveloped) query; normalise whitespace; cap length; guard numeric params (`top_k` 1–20); extract structured filters | `TrustLevel.ANONYMOUS` | `validated_input` \| `user_input` | `search_query`, `query_filters`, `intake_notes` |
| `RetrieveNode` | Deterministic keyword retrieval over the seeded KB (`config/kb/loyalty_kb.json`): tokenise query, score title/tags/content overlap, apply category filter | `TrustLevel.ANONYMOUS` | `search_query`, `query_filters`, `retrieval_config` | `retrieved_documents`, `intake_notes` |
| `RerankFilterNode` | Rerank candidates (category-match boost), drop entries below `score_threshold`, cap at `top_k` | `TrustLevel.ANONYMOUS` | `retrieved_documents`, `query_filters`, `retrieval_config` | `ranked_documents` |
| `GenerateAnswerNode` | Rule-based grounded answer assembly from the ranked KB passages only, with numbered citation markers (v1 deterministic — LLM synthesis seam documented below); records the top-matched KB category | `TrustLevel.ANONYMOUS` | `ranked_documents` | `grounded_answer`, `citations`, `answer_category` |
| `OutputFormatNode` | Compose the final answer: body + Sources list + the standing v1-scope disclaimer (disclaimer is part of this node, NOT post_process) | `TrustLevel.ANONYMOUS` | `grounded_answer`, `citations` | `formatted_answer`, `status` |

### Data Flow

```
user_input
  → PreProcessNode (S-1/S-2)                     → validated_input
  → LoyaltyOfferQAGraphNode.extract_input        → inner DomainWorkflowGraph.invoke(validated_input)
        → input_validate                         → search_query / query_filters
        → retrieve                                → retrieved_documents
        → rerank_filter                           → ranked_documents
        → generate_answer                         → grounded_answer / citations / answer_category
        → output_format                           → formatted_answer (+ v1-scope disclaimer)
     get_output() → {formatted_answer, citations, answer_category, status, ...}
  → LoyaltyOfferQAGraphNode.merge_output          → result = formatted_answer, loyalty_answer
  → PostProcessNode (S-3)                          → pass:  formatted_output (gated)
                                                     block: every output-bearing field
                                                            cleared + fixed notice, ERROR
  → LoyaltyOfferQAAgent.get_output() (B10)         → base envelope + citations / answer_category
                                                       (SUCCESS-only, S-3 re-gated;
                                                        non-SUCCESS → output withheld)
```

Structured parameters travel as JSON through `extract_input()`: when the caller
supplies a JSON envelope (`{"query": ..., "category": ..., "top_k": ...}`), it
passes through `validated_input` as a string and the FIRST inner node
(`InputValidateNode`) parses it back. Inner nodes read their input via
`state.get("validated_input") or state.get("user_input", "")`.

### Runtime config forwarding (`_parent_config`)

Two files sit in `config/` and they are not interchangeable:

| File | Holds | Read by |
|------|-------|---------|
| `config/agent.yaml` | the **flat manifest** — `id`, `name`, `namespace`, entry-point `class`, `required_trust_level`, `requires` | `AgentRegistry`; `src/api/server.py` for `namespace` / `name` |
| `config/config.yaml` | the **runtime parameters** — `max_retry`, `timeout_s`, and the `retrieval` block | `AgentRegistry`, passed in as `Graph(config=...)` |

The `retrieval` block lives in `config/config.yaml`, NOT in the manifest. Reading
a runtime value out of the manifest is a silent no-op — the lookup succeeds,
returns nothing, and the code falls back to its module defaults while every test
stays green. `src/graph/runtime_config.py` is the single module that knows which
file holds what.

`LoyaltyOfferQAGraphNode._parent_config()` calls `resolve_retrieval()`, which
reads the runtime config (constructor config first, then `config/config.yaml` on
disk, then the module defaults), validates it, and forwards **only** `retrieval`
under `config["configurable"]` (never `{}`):

```
{"configurable": {"retrieval": {top_k, score_threshold, kb_path}}}
```

`get_subgraph()` passes this into `DomainWorkflowGraph(config=...)` (graph-level
constructor injection of immutable config — this is distinct from the per-node
`execute()` contract). At runtime the live values reach the inner graph the other
way round: `LoyaltyOfferQAAgent._extra_initial_state()` publishes the resolved
block into outer state as `retrieval_config`, `extract_input()` stashes it on the
context bridge, and `DomainWorkflowGraph._extra_initial_state()` seeds it into the
inner state. The constructor block is the fallback for a directly-invoked inner
graph. `RetrieveNode` and `RerankFilterNode` read `top_k` / `score_threshold`
**exclusively from the state-seeded `retrieval_config` field** — `execute()` takes
no `config` parameter (SDK v1.0.0rc1 / CoE 2026-07-27 contract) — falling back to
safe module defaults that mirror the declared values when the field is absent
(e.g. bare unit-test state).

A declared value out of range does not get clamped: `validate_retrieval_block()`
raises `ConfigError` and the deployment stops. A misconfigured relevance floor
that silently answered every question differently would be the worse outcome.

### State Definition

| Field | Type | Purpose | Layer |
|-------|------|---------|-------|
| `validated_input` | `NotRequired[str]` | identifier-stripped query payload | outer |
| `loyalty_answer` | `NotRequired[str]` | final answer, mapped from inner `formatted_answer` | outer |
| `answer_category` | `NotRequired[Optional[str]]` | KB category the answer was grounded in (`program_rules` \| `offer_catalog` \| `redemption_catalog`), or `None` on no-coverage | outer + inner |
| `search_query` | `NotRequired[str]` | normalised search query | inner |
| `query_filters` | `NotRequired[Optional[str]]` (JSON) | parsed structured params (`category`, `top_k`) | inner |
| `retrieval_config` | `NotRequired[Optional[str]]` (JSON) | resolved `retrieval` block from `config/config.yaml` | outer + inner |
| `retrieved_documents` | `NotRequired[Optional[str]]` (JSON) | scored KB candidates | inner |
| `ranked_documents` | `NotRequired[Optional[str]]` (JSON) | reranked + threshold-filtered passages | inner |
| `grounded_answer` | `NotRequired[str]` | rule-assembled grounded answer body | inner |
| `citations` | `NotRequired[Optional[str]]` (JSON) | `[{ref, id, title, source}]` | inner + outer |
| `formatted_answer` | `NotRequired[str]` | final answer + sources + disclaimer | inner |
| `intake_notes` | `NotRequired[Optional[str]]` (JSON) | validation / parse notes (no PII) | inner |
| `trace_id` / `correlation_id` | `Optional[str]` | framework-managed tracing | both |

**State Constraints (mandatory):**
- Flat `TypedDict` only (primitives + JSON-serialisable types).
- Structured fields (dict / list[dict]) stored as JSON STRINGS via `to_json()` /
  `from_json()` — used consistently by every producer AND consumer (ADR-005
  msgpack safety).
- Domain fields are `NotRequired[...]` (valid TypedDict before any node writes).
- `formatted_output` is NOT re-declared (backbone field stays framework-owned).
- No JWT, API keys, credentials, or raw personal identifiers in State.
- **No live points balance, no live member-account data of any kind** — v1 has
  no loyalty-platform/POS integration; the agent answers policy/catalog
  questions only (`docs/01_proposal.md` "Scope Revision").
- `InvocationContext` via `config["configurable"]` only (never in State).
- No Pydantic models / dataclasses / arbitrary Python objects.

## Security Gates

- **S-1 (trust gate / input validation):** every node declares
  `required_trust_level` (see tables above); `PreProcessNode`
  (VERIFIED_EXTERNAL) rejects empty / non-string `user_input` before the inner
  workflow runs. The standalone server elevates authenticated Bearer callers to
  VERIFIED_EXTERNAL (`INVOKE_AUTH_TOKEN`).
- **S-2 (identifier screen) — deliberate handling for this template:** this
  agent handles loyalty POINTS BALANCES and member identity in the free-text
  query (a member or a CSR relaying a member's question may paste a member ID,
  loyalty-card number, or a claimed points balance into `user_input`).
  `PreProcessNode._surface_strip_identifiers()` redacts labelled
  member/loyalty/card ID tokens, long digit runs (8+, the shape of a loyalty
  card / member number), and e-mail addresses from the payload **before**
  `validated_input` is written, so nothing beyond that surface-stripped text
  ever reaches the inner workflow or the checkpoint DB. The framework
  `FunctionNode` default PII scan additionally masks `user_input` /
  `validated_input` at every node boundary.
  **Design decision (domain-specific, deviates from the peer template golden):**
  `GenerateAnswerNode` does NOT interpolate the raw query text into the
  rendered answer (the golden's `GenerateAnswerNode` quotes the question back
  in its lead sentence — FIN's compliance-question domain carries low PII
  risk; this agent's domain does not, so any surface-stripped residual is kept
  out of the rendered output entirely as a second layer of defence).
- **S-3 (output gate):** `PostProcessNode` calls the module-level
  `_security_gate_output()` and `_enforce_numeric_fidelity()` scans from
  `execute()` — API keys / JWT / Bearer tokens / credential assignments /
  connection strings, and any figure in the answer that is not in the passages
  the answer was assembled from. The scans walk dict/list/tuple structures
  RECURSIVELY, not just top-level strings (molt HIGH ×2, 2026-07-24 —
  A peer template / a peer template). Violation messages name the LOCATION only, never
  the matched value. No `_extra_security_gate_input` /
  `_extra_security_gate_output` instance methods are defined on any node (the
  real SDK auto-wraps such hooks — prohibited).

  The template's credential pattern set is the **union** of the framework's own
  `detect_credentials()` list and the shapes it does not know (any scheme
  carrying inline credentials, and `password=` / `api_key:` keyword
  assignments) — never a subset. A shape only the framework catches is a bypass
  of the clearing below, not a second line of defence: the framework's `@final`
  S-3 gate raises on the returned delta and `BaseNode.__call__` discards the
  node's whole dict, the clearing with it. Widening the local set is safe;
  narrowing it is the bypass.

  **Containment contract on a block — what a refusal actually returns.**
  `AgentBaseGraph.get_output()` resolves the caller-facing payload as
  `state.get("formatted_output") or state.get("result")`, with **no status
  check**. Returning `AgentStatus.ERROR` without touching `result` therefore
  ships the pre-gate answer inside the error envelope, and setting
  `formatted_output` to `""` or omitting it selects that same fallback, because
  both are falsy. This template contains a block in two independent places:

  1. `PostProcessNode.execute()` returns `blocked_output_state()` — every
     output-bearing outer field (`result`, `loyalty_answer`, `citations`,
     `answer_category`, `grounding_excerpts`) **present and emptied**, and
     `formatted_output` set to a fixed, TRUTHY notice. Present-and-emptied
     rather than omitted because LangGraph merges partial deltas: a key left
     out of the returned dict keeps its previous value in state, so a
     checkpoint or any later reader would still hold the refused answer.
     `tests/unit/test_post_process_node.py` pins the cleared set against the
     keys `merge_output()` writes, so a new caller-facing field cannot skip it.
  2. `LoyaltyOfferQAAgent.get_output()` withholds `output` entirely on any
     non-SUCCESS status. This is not redundant with (1): where the framework's
     own `@final` credential scan raises on the returned delta, `BaseNode`
     discards the node's whole dict — the clearing with it — and where the
     SUCCESS-path second pass over the structured fields refuses, the gate
     passed and cleared nothing at all. Only the envelope check covers those.

  `tests/integration/test_output_containment_e2e.py` drives both gate layers
  and the structured second pass through the full outer graph and through the
  real `/invoke` handler, and asserts on the complete response: `status=error`,
  no `output`, and no part of the refused answer anywhere in it. A clean-path
  control in the same module proves the gate still answers.
  **B10 structured output (STRUCTURED_PRODUCT=yes):** `LoyaltyOfferQAAgent.get_output()`
  extends `super().get_output()` and surfaces `citations` + `answer_category`
  as real (decoded) values on top of the base envelope — fail-closed twice:
  (1) only attached when the terminal `status` is SUCCESS; (2) even on
  SUCCESS, the two fields are re-scanned through the SAME
  `_security_gate_output()` used by `PostProcessNode` before being surfaced —
  a violation flips `status` to ERROR and withholds both fields. Both fields
  are KB-derived (never a raw echo of the caller's query or a raw
  request/response object), so no member-supplied value can reach them.
- **S-4 (audit logging):** every node's `execute()` emits exactly ONE
  domain-specific `emit_trace_event("<node>_complete", {small non-PII payload},
  state)` (free function, positional args) on its success path. Payloads carry
  ONLY counts / lengths / booleans / config knobs (`top_k`, `score_threshold`)
  — **never** query text, a category value copied from the caller, or any
  digit sequence that could be a member ID or a points balance. Nodes do NOT
  emit `node_start` / `node_complete` / `node_error` — `BaseNode.__call__()`
  emits those. Domain event names (these are the names
  `docs/07_operation_guide.md` will document):
  - `pre_process_complete`
  - `input_validate_complete`
  - `retrieve_complete`
  - `rerank_filter_complete`
  - `generate_answer_complete`
  - `output_format_complete`
  - `post_process_complete`

## v1 Scope Disclaimer

Every answer carries a standing v1-scope disclaimer (informational only; not a
live account balance; verify redemption eligibility and current promotion terms
with member services before redeeming). It is appended by `OutputFormatNode` as
part of the domain output contract — NOT injected by `post_process` (post_process
only gates). See `docs/01_proposal.md` "Scope Revision" for the full v1
boundary (no live loyalty-platform/POS integration, no per-member live
points-balance lookup, static engineer-seeded KB snapshot).

## v1 Implementation Note — LLM synthesis

v1 of this template is **deterministic end-to-end**: retrieval is keyword
scoring over the seeded KB and `GenerateAnswerNode` assembles the grounded
answer rule-based from the ranked passages (lead sentence + cited passage
excerpts). There is NO live LLM call and no LLM client dependency in v1, and no
`system_prompt` is read at runtime.

`config/config.yaml` still carries an `llm` block (`temperature`, `max_tokens`),
placed there by the real-stg manifest migration. **It has no consumer.** It is
not forwarded by `_parent_config()`, and no module under `src/` constructs a
model client or reads it. Treat it as an inert placeholder for the v2 seam
below, not as live configuration: changing its values changes nothing.

The LLM synthesis upgrade seam is
documented in `config/prompts/answer_synthesis_prompt.md`: a v2
`GenerateAnswerNode` swaps the rule-based assembly for an LLM call that
synthesises over the same `ranked_documents` input and emits the same
`grounded_answer` / `citations` state contract, so no other node changes.

## Composition Pattern

- **Pattern:** `GraphNode` (subgraph) in the outer `main` slot.
- **Composition target:** `DomainWorkflowGraph` (inner `BaseGraph`).
- **Error propagation strategy:** `propagate` (inner errors re-raised as `SubgraphError`).
- Inner domain nodes run at `TrustLevel.ANONYMOUS`; outer pre/post_process run
  at `TrustLevel.VERIFIED_EXTERNAL`.

## Import Isolation Confirmation
- [x] Template does not import the Level 0 platform SDK.
- [x] Import targets: `framework/` and `shared/` only (no `agents/base/` required).
- [x] No L2 class names (`VectorRAGAgent`, `ChatAgent`, …) in any base position.

## Design Decision Record

| Decision | Option A | Option B | Chosen | Rationale |
|----------|----------|----------|--------|-----------|
| L1 base type | AgentBaseGraph | AutonomousBaseGraph | **AgentBaseGraph** | Fixed multi-step RAG workflow, no autonomous loop |
| Composition pattern | Standalone Cat 1 slots | GraphNode → inner BaseGraph | **GraphNode → inner BaseGraph** | 5-step domain workflow exceeds a single `main` node; nested keeps the outer backbone untouched |
| Config delivery to nodes | `execute(state, config)` | State-seeded (`_extra_initial_state`) | **State-seeded** | 2026-07-27 CoE contract: `execute(self, state) -> dict` only; keeps node ctors no-arg |
| Query echo in answer body | Quote the caller's query | Never echo caller input | **Never echo** | This template's queries may carry a member ID / points balance the S-2 screen imperfectly redacts; the golden's query-quoting habit is not safe to copy here |
| Answer synthesis | Rule-based assembly | LLM call | **Rule-based (v1)** | SDK v1.0.0rc1 ships no LLM client; deterministic assembly is testable; v2 swaps the LLM in at the documented seam |
| KB storage | External vector store | Seeded JSON KB | **Seeded JSON KB (v1)** | Self-contained, deterministic CI; the retrieval contract (`retrieved_documents` JSON) is store-agnostic for a later vector-store upgrade; also matches `docs/01`'s "static, engineer-seeded" v1 scope |
