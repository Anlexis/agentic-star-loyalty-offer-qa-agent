# Test Specification — RET-C2-087

**Template ID:** RET-C2-087
**Template Name:** LoyaltyOfferQAAgent
**Category:** Cat 2 (nested RAG)

This document defines the test cases for the Wave-1 implementation (state,
nodes, inner/outer graphs, manifest, server). The test code is delivered in
Wave-2 (`tests/unit/` + `tests/proof_of_boundary/`); this spec is the contract
those tests implement.

## 1. Scope & Invocation Conventions

- Per-node unit tests for the 5 inner domain nodes + the 2 outer S-gate nodes.
- Manifest/config consistency (`config/agent.yaml` ↔ code) and seeded-KB integrity.
- Retrieval quality (golden queries over `config/kb/loyalty_kb.json`).
- Inner-graph (`DomainWorkflowGraph`) and outer-graph (`LoyaltyOfferQAAgent`)
  composition / integration.
- Proof-of-Boundary (PoB): import isolation, State msgpack safety, invoke
  order (PB-6), HITL propagation (PB-7, conditional), server boot.

**Trust-gate invocation canon (CoE 2026-07-16).** Every per-node test invokes
the node via `node(state)` — through `BaseNode.__call__`, which runs
S-1 trust → S-2 PII mask → `execute()` → S-3 credential gate — never a bare
`node.execute(state)`. The state builder sets `caller_trust_level` to
`TrustLevel.VERIFIED_EXTERNAL.value` for the two outer S-gate slots
(PreProcessNode / PostProcessNode — the manifest's declared caller level) and
`TrustLevel.ANONYMOUS.value` for the five inner domain nodes.
**No config-parameter carve-out** (RULES C2 ⛔ retired 2026-07-27): `execute(self,
state) -> dict` takes no second argument — a call passing one would `TypeError`
against the real node. RetrieveNode / RerankFilterNode config knobs (`kb_path`,
`top_k`, `score_threshold`) are exercised exclusively via the state-seeded
`retrieval_config` JSON field, always through `node(state)`.

**S-2 masking expectations.** The framework input gate masks
`user_input`/`validated_input`/`llm_response` (e-mail, 12-digit runs, SSN/
phone/credit-card digit groups, Title-Case name bigrams) to `[MASKED]` before
`execute()` runs. `PreProcessNode` additionally runs its own
`_surface_strip_identifiers()` screen for labelled member/loyalty/card
identifiers and unformatted 8-9-digit runs the framework patterns do not
catch, replacing them with `[REDACTED]`. Every masking claim in §2.1 was
verified empirically against the installed real SDK's `detect_pii()` before
being written down (RULES C3 — "measure, don't reason"), including the
Wave-2 STG sign-off payload itself: its `1,000-yen` digit group is
comma-separated, not a 12-digit / credit-card / phone / SSN shape, so it is
**not** masked — the SUCCESS path is unaffected end to end. Positive-path
payloads are otherwise lowercase, PII-free domain phrasing. Domain fields
(`grounded_answer`, `formatted_answer`, `retrieved_documents`, …) are not S-2
scan targets.

**Audit muting.** `shared.*` is never sys.modules-stubbed (the framework
imports `shared.security` at load time). The domain S-4 emitter is muted via
an autouse fixture patching `src.nodes.<mod>.emit_trace_event`; the audit
assertion test re-patches the same attribute with a spy and asserts on
`call.args[1]` (the event payload).

## 2. Unit Test Cases

### 2.1 PreProcessNode (outer pre_process slot; S-1/S-2) — `test_pre_process_node.py`

| ID | Case | Input | Expected |
|----|------|-------|----------|
| PRE-01 | Valid query (STG sign-off payload) | `"How many points ... 1,000-yen digital gift card ..."` | `status=SUCCESS`, `validated_input` byte-equal to the input (empirically unmasked), `enriched_context` carries channel/source |
| PRE-02 | Empty input | `""` / whitespace | `status=ERROR`, `error_log` non-empty, no `validated_input` |
| PRE-03 | Missing / non-string input | `user_input` absent; dict payload | `status=ERROR` |
| PRE-04 | Labelled member ID | `"member ID is LM-1029384"` | node screen: raw id absent, `[REDACTED]` present |
| PRE-05 | Labelled loyalty card number | `"loyalty card # 88231"` | node screen: raw number absent, `[REDACTED]` present |
| PRE-06 | E-mail | `"jane.doe@example.com"` | framework S-2: raw address absent, `[MASKED]` present |
| PRE-07 | Grouped 12-digit run | `"1234 5678 9012"` | framework S-2 `my_number` pattern: `[MASKED]` present (caught before the node's own screen sees it) |
| PRE-08 | Unformatted 8-digit run | `"12345678"` | below every framework digit shape; node's own `\b\d{8,9}\b` screen: `[REDACTED]` present |
| PRE-09 | Unformatted 7-digit run | `"1234567"` | below the node's 8-digit floor: NOT redacted (boundary case) |
| PRE-10 | Name bigram | `"Jane Doe"` | framework S-2 `name` heuristic: `[MASKED]` present |
| PRE-11 | S-4 audit | valid query | `pre_process_complete` emitted; payload (`call.args[1]`) carries `input_chars` |

### 2.2 InputValidateNode (inner node 1) — `test_input_validate_node.py`

| ID | Case | Input | Expected |
|----|------|-------|----------|
| VAL-01 | Plain text | free-text query | whole string becomes `search_query`; filters `{category: None, top_k: None}` |
| VAL-02 | Whitespace | ragged spacing/newlines | collapsed to single spaces |
| VAL-03 | JSON envelope | `{"query","category","top_k"}` | all three parsed; `question` alias accepted; category lower-cased/stripped |
| VAL-04 | Malformed JSON | `{`-prefixed non-JSON | treated as plain-text query + parse note |
| VAL-05 | top_k out of range | 99 / −5 / `"many"` | **refused**: `status=ERROR`, `error_log` names the field `top_k` — not clamped, not dropped |
| VAL-06 | top_k non-finite | `NaN` / `inf` | **refused** the same way (a clamp turns a non-finite value into the strictest bound and silently changes every answer) |
| VAL-07 | Refused value is not echoed | out-of-range value | the rejected value never appears in `error_log` |
| — | Valid override | `top_k=9` via either channel | applied; `input_context` wins over the in-question envelope on a clash |
| — | Unknown category | `"offer"` | **refused** with the field name — matched whole-value against the KB vocabulary, never as a substring |
| VAL-08 | Oversize query | > 2000 chars | truncated to 2000 + note |
| VAL-09 | Empty request | `""` | `search_query=""` + "empty request" note (non-fatal) |
| — | Fallback | `validated_input` absent | falls back to `user_input` |
| — | ADR-005 | any | `query_filters` is a JSON string, never a bare dict |

### 2.3 RetrieveNode (inner node 2) — `test_retrieve_node.py`

| ID | Case | Input | Expected |
|----|------|-------|----------|
| RET-01 | Happy path | redemption query | top-1 candidate is `kb-011` |
| RET-02 | Ordering | redemption query | scores strictly sorted desc; all > 0 |
| RET-03 | Entry shape | any hit | keys `{id,title,category,source,score,excerpt}`; excerpt ≤ 400 chars |
| RET-04 | Category filter | `query_filters.category="offer_catalog"` | only offer_catalog entries; top-1 `kb-008` |
| RET-05 | Empty query | `""` | no candidates |
| RET-06 | State-seeded config: kb_path override | `retrieval_config.kb_path` bogus | `[]` + "not readable" note (no config= parameter exists — RULES C2 retired) |
| RET-07 | State-seeded config: top_k shapes pool | `retrieval_config.top_k=1` vs default 4, query="points" (11 KB hits) | pool = max(top_k·3,10): top_k=1 → 10 kept; default → 11 kept |
| RET-08 | Missing retrieval_config | no `retrieval_config` key at all | falls back to module defaults; still resolves the seeded KB |
| RET-09 | Notes accumulation | prior `intake_notes` | appended, never clobbered |

### 2.4 RerankFilterNode (inner node 3) — `test_rerank_filter_node.py`

| ID | Case | Input | Expected |
|----|------|-------|----------|
| RRF-01 | Relevance floor | scores 0.9 / 0.1 | 0.1 dropped (default 0.25 floor) |
| RRF-02 | State-seeded score_threshold | `retrieval_config.score_threshold=0.5` | 0.3 dropped |
| RRF-03 | State-seeded top_k | `retrieval_config.top_k=1` | one survivor, highest score |
| RRF-04 | Category boost | matching category | +0.1, re-ranked ahead |
| RRF-05 | Boost cap | 0.95 + boost | capped at 1.0 |
| RRF-06 | Caller top_k | stricter (1) wins; looser (10) does not widen | enforced |
| RRF-07 | Garbage entries | non-dict / uncoercible score | skipped / coerced to 0.0 and dropped |
| RRF-08 | Tie-break | equal scores | deterministic id-ascending order |

### 2.5 GenerateAnswerNode (inner node 4) — `test_generate_answer_node.py`

| ID | Case | Input | Expected |
|----|------|-------|----------|
| GEN-01 | Citation markers | 2 ranked passages | `[1]`/`[2]` markers with titles |
| GEN-02 | **Query never echoed** *(docs/02 deviation from the golden)* | query with a distinctive marker token | marker token absent from `grounded_answer` regardless of ranked content; lead line is the fixed generic sentence, not a quote of the query |
| GEN-03 | Citations list | ranked passages | refs 1..n mirror ranked order; id/title/source carried |
| GEN-04 | Groundedness | single passage | answer body traces to ranked passages only |
| GEN-05 | `answer_category` | multiple ranked passages | equals the top-ranked passage's category |
| GEN-06 | No coverage | empty/missing `ranked_documents` | escalation answer; `citations=[]`; `answer_category=None`; query still not echoed |

### 2.6 OutputFormatNode (inner node 5, terminal) — `test_output_format_node.py`

| ID | Case | Input | Expected |
|----|------|-------|----------|
| FMT-01 | Full compose | body + citations | header + body + `## Sources` rows + v1-scope disclaimer; `status=SUCCESS` |
| FMT-02 | Blank source | citation without source | no `()` suffix |
| FMT-03 | Disclaimer | every input | disclaimer (+ "verify ... with member services") rides with every answer |
| FMT-04 | No citations | empty list | explicit "- none (…)" sources line |
| FMT-05 | Missing body | no `grounded_answer` | fallback text; `status=SUCCESS` |

### 2.7 PostProcessNode (outer post_process slot; S-3) — `test_post_process_node.py`

Two independent scans, verified empirically. On a block the node returns
`blocked_output_state()`: every output-bearing field present and emptied, plus a
fixed TRUTHY `formatted_output` notice. Omitting `formatted_output` is NOT the
contract and must never be asserted as one — the base envelope resolves
`formatted_output or result`, so an absent or empty value re-selects the very
answer the gate refused. See docs/02 "Containment contract on a block".

| ID | Case | Input (`result`) | Expected |
|----|------|------------------|----------|
| POST-01 | Clean output | normal loyalty Q&A answer | `formatted_output=result`, `status=SUCCESS` |
| POST-02 | Empty result | `""` | forwarded as-is, `status=SUCCESS` (non-fatal) |
| POST-03..05b | Domain gate hit (node's own regex: `eyJ…`/`sk-…`/`Bearer …`/db URI/`password=…`) | JWT / OpenAI-style key / Bearer token / keyword assignment | `status=ERROR`; `result`, `loyalty_answer`, `citations`, `answer_category`, `grounding_excerpts` all **present and empty**; `formatted_output` = the fixed notice (truthy); secret absent from the whole returned dict |
| POST-05c | Fidelity gate hit | a figure absent from `grounding_excerpts` | same cleared shape; the ungrounded figure is not echoed back |
| POST-05d | Cleared-set inventory guard | — | the cleared set equals the caller-facing keys `merge_output()` writes, so a new field cannot silently skip the clearing |
| POST-06 | AWS access key | `AKIA…` | caught by the node's own scan (see POST-06b), so the cleared shape applies as in POST-03..05b |
| POST-06b | Local scan ⊇ framework scan | one sample per framework credential type (`sk_live_`, `sk-`, `eyJ`, `AKIA`, `Bearer`, `postgresql://`, `mongodb://`) | each is caught by the node's own `_security_gate_output()`. A shape only `detect_credentials()` catches is a **bypass of the clearing**, not a second line of defence: the framework's `@final` gate raises on the returned delta and `BaseNode.__call__` discards the whole dict, clearing included. Widening the local set is safe; narrowing it is the bypass |
| POST-07 | Connection string with inline credentials | a `postgresql` URI carrying `user`/`password` before the `@` | cleared shape as in POST-03..05b |
| POST-08 | S-4 audit | clean + blocked cases | `post_process_complete` (success) / `post_process_credential_blocked` / `post_process_fidelity_blocked` (violation, with `violation_count`) |

> **Fixed 2026-07-30 (found while authoring this spec):** a generic
> `password=<value>` assignment matched neither the node's own
> `_CREDENTIAL_LIKE_RE` (`eyJ…`/`sk-…`/`Bearer …`) nor the framework's
> `detect_credentials()`. The domain gate now also scans keyword-assignment
> shapes via `_SECRET_ASSIGNMENT_RE` (`password`/`passwd`/`pwd`/`secret`/
> `api_key`/`access_key` + `:`/`=` + a 6+ char value, case-insensitive) —
> pinned by POST-05b, and reused by `get_output()`'s second pass.

### 2.8 Manifest / config consistency — `test_config_manifest.py`

| ID | Case | Expected |
|----|------|----------|
| CFG-01 | Template ids | `agent_id` = `template_id` = `agent.id` = `RET-C2-087` |
| CFG-02 | Class-name contract | manifest `agent.class`/`agent.name` == `LoyaltyOfferQAAgent` (graph.py class); `module=src.graph` |
| CFG-03 | Classification | Cat 2 / RET / RAGAgent |
| CFG-04 | Trust level | manifest `VERIFIED_EXTERNAL` == PreProcessNode & PostProcessNode `required_trust_level` |
| CFG-05 | max_retry | int, `0 ≤ v < 10` (framework ceiling); hitl not enabled (PB-7 waiver contract), `LoyaltyOfferQAGraphNode.propagate_hitl is False` |
| CFG-06 | Retrieval block | declared in `config/config.yaml` (NOT the manifest — a manifest copy would be read by nobody); `top_k`/`score_threshold` mirror the node module defaults; `kb_path` exists |
| CFG-07 | `_parent_config()` | forwards the **runtime-config `retrieval` block only**; never `{}`. The `llm` block in `config/config.yaml` has no consumer and is not forwarded (see docs/02 "v1 Implementation Note") |
| CFG-07b | Declared value reaches the nodes | a `retrieval` block handed in as `Graph(config=...)` arrives in the state field `RetrieveNode` / `RerankFilterNode` read |
| CFG-08 | KB integrity | JSON list ≥ 5 entries; unique ids; required keys per entry; categories ⊆ {program_rules, offer_catalog, redemption_catalog} |

### 2.9 Retrieval quality (golden queries) — `test_retrieval_quality.py`

| ID | Case | Expected |
|----|------|----------|
| QUAL-01 | 8 golden domain queries (one per category, plus spread) | expected KB entry is top-1 (kb-001/002/003/006/007/009/010/011) |
| QUAL-02 | Relevance floor | every survivor ≥ 0.25 |
| QUAL-03 | Citation integrity | every survivor id exists in the seeded KB |
| QUAL-04 | Precision | membership-tiers query keeps ONLY `kb-003` |
| QUAL-05 | Category filter | offer_catalog filter → only offer_catalog entries, top-1 `kb-008` |
| QUAL-06 | No coverage | out-of-domain query → zero survivors |
| QUAL-07 | Escalation answer | no-coverage → explicit escalation text, no citations |

## 3. Integration / Composition

### 3.1 Inner graph — `test_domain_workflow_graph.py`

| ID | Case | Expected |
|----|------|----------|
| INT-01 | Composition | inherits `BaseGraph`; registers exactly the 5 domain nodes; no initialize/finalize |
| INT-02 | Config forwarding | `_extra_initial_state()` republishes the retrieval block as the JSON-string `retrieval_config` |
| INT-03 | Output shape | `get_output()` emits `formatted_answer`/`citations`/`answer_category`/`status`/… (the merge contract); `route()` → END on error |
| INT-04 | Inner e2e | full inner `invoke()` → SUCCESS; formatted answer + disclaimer + kb-011 citation + `answer_category=redemption_catalog`; inner `node_history` = the 5 domain nodes in linear order; no-coverage query still SUCCEEDs with the escalation text |

### 3.2 Outer graph + e2e — `test_graph_composition.py`

| ID | Case | Expected |
|----|------|----------|
| INT-05 | Outer composition | inherits `AgentBaseGraph` (L1 direct); `Graph` alias; `add_edges()` NOT overridden |
| INT-06 | Backbone slots | compile() fills all 5; pre/main/post are PreProcessNode / LoyaltyOfferQAGraphNode / PostProcessNode |
| INT-07 | `get_subgraph()` | returns `DomainWorkflowGraph` carrying the forwarded retrieval config |
| INT-08 | `extract_input()` | prefers `validated_input`, falls back to `user_input` |
| INT-09 | `merge_output()` | inner `formatted_answer` → outer `loyalty_answer` AND `result`; `citations`/`answer_category`/`status` mapped; changed keys only |
| INT-10 | Runtime-config fallback | `_parent_config()` never `{}` even with an unreadable `config/config.yaml` — it falls back to the module defaults that mirror the shipped file |
| INT-11 | e2e happy path | VERIFIED_EXTERNAL invoke on the STG sign-off payload → SUCCESS; `output` = gated formatted answer; PostProcessNode traversed |
| INT-12 | e2e structured fields | `answer_category`/`citations` surfaced on SUCCESS (RULES B10) |
| INT-13 | e2e S-1 denial | ANONYMOUS invoke → ERROR; empty `output`, `citations`, `answer_category`; `LoyaltyOfferQAGraphNode` runs (records itself in `node_history`) but its OWN generic "incoming state already ERROR" input-gate short-circuit skips `execute()` — the inner subgraph is never invoked (verified by instrumented trace); PostProcessNode NOT traversed |
| — | ADR-005 helpers | `to_json`/`from_json` round-trip; None/malformed handling |

### 3.3 Output containment (e2e) — `tests/integration/test_output_containment_e2e.py`

Every case here reads the COMPLETE response from a full `agent.invoke()` (or the
real `/invoke` handler) and asserts behaviour, not message wording. Faults are
injected on the DATA path — the knowledge base is the agent's retrieval
transport, so a KB entry carrying a credential is a transport returning one; the
pipeline, the gates and the envelope code are the shipped ones.

| ID | Case | Expected |
|----|------|----------|
| CON-01 | Credential block, e2e | KB passage carries `password=<value>` (a shape the template gate catches and the framework detector does not, so it reaches post_process) → `status=error`, no `output`, no `citations`/`answer_category`, and neither the secret nor the answer text anywhere in the response; `PostProcessNode` in `node_history` (the block happened AT the gate) |
| CON-02 | Credential block, persisted state | after the same invoke, `result` / `loyalty_answer` / `citations` / `answer_category` / `grounding_excerpts` are all **present and empty** in the state the pipeline built, `formatted_output` is truthy, and the secret appears nowhere in state |
| CON-03 | Credential block, `/invoke` | the same request through the real ASGI handler (Bearer-elevated to VERIFIED_EXTERNAL) returns the same contained envelope |
| CON-04 | Ungrounded-figure block, e2e | `merge_output` drift stops carrying `grounding_excerpts`, so every figure in the answer is unsourced → same contained envelope |
| CON-05 | Structured second pass, e2e | KB entry whose `category` carries a figure absent from the grounding text (category never enters `grounding_excerpts`, and is never rendered into the answer, so post_process passes and cleared nothing) → `status=error`, no `output` |
| CON-06 | Already-contained refusal paths | empty request / request screen / caller-contract refusal / inner-graph refusal / S-1 denial → `status=error`, no payload. **Regression pins, not fixes** — these short-circuit before `merge_output` writes `result` and were contained already |
| CON-07 | Clean-path control | the same clean request still returns its real answer, `answer_category` and `citations`; a refuse-everything gate cannot pass |

> **Why these are not node-level tests.** The node-level POST-* cases proved
> `status=ERROR` on a block and stopped there. They never drove the outer graph
> with a gate-triggering `result`, so they could not see that
> `AgentBaseGraph.get_output()` resolves `formatted_output or result` with no
> status check and was handing the caller the un-gated answer. The mutation
> matrix for these tests is recorded on the fix MR: each containment layer has a
> mutant only it fails, and reverting to the original shipped `src/` fails all
> five containment cases.

## 4. Proof-of-Boundary

| ID | Case | Expected |
|----|------|----------|
| PB-IMPORT | `test_import_isolation.py` | no `agenticstar` / Level-0 import anywhere under `src/` |
| PB-STATE | `test_state_safety.py` | `State` has no credential-named fields and no `BaseModel` / `InvocationContext` annotations |
| PB-6 | `test_pb_invoke_order.py` | full `Graph().invoke()` with `InvocationContext(caller_trust_level=VERIFIED_EXTERNAL)` (never `for_internal()`) over the payload byte-equal to `deploy/invoke_payload.json`'s `input` → SUCCESS with outer `node_history` exactly `[InitializeNode, PreProcessNode, LoyaltyOfferQAGraphNode, PostProcessNode, FinalizeNode]` |
| PB-7 | `test_pb7_hitl_interrupt_propagation.py` | **Auto-waived — non-HITL** (no graph class declares `propagate_hitl=True`); canonical conditional skip-stub (a peer template convention) retained |
| PB-BOOT | `test_server_boot.py` | `import src.api.server` does not raise; module-level agent is this template's class, compiled; fresh ctor→`compile()` fills the 5 backbone slots; `/health` reports the agent |

> **Pre-CoE gate checklist:** PB-IMPORT, PB-STATE, PB-6 and PB-BOOT are
> mandatory. PB-7 applies only to HITL-enabled templates — this template is
> non-HITL, so PB-7 is **Auto-waived — non-HITL** and its skip must not block
> the gate.

## 5. Test Execution Summary

- Execution date: 2026-09-08
- Runner: real SDK wheel (`agenticstar-agentcore` 1.0.2), no stub
- Main (`pytest tests/`): 334 passed, 1 skipped (335 collected)
- PB (`pytest tests/proof_of_boundary/`): 28 passed, 1 skipped (29 collected)
- Pass: 334 / Fail: 0 / Skip: 1 (PB-7 conditional stub — auto-waived, non-HITL)
- Determinism: no LLM, no network; retrieval + answer assembly are rule-based
- Prior recorded run: 2026-07-30, 158 passed / 1 skipped on `agenticstar-agentcore==1.0.0`
