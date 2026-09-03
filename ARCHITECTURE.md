# System Architecture & Technical Design

This document details the architectural design, state schema, policy engine,
evidence/citation model, and Human-in-the-Loop (HITL) lifecycle for the
**Merchant Onboarding Risk Auditor**.

---

## 1. High-Level Architecture Diagram

```
                     ┌───────────────────────────┐
                     │   MERCHANT INPUT PAYLOAD    │
                     └─────────────┬─────────────┘
                                   │
                                   ▼
                     ┌───────────────────────────┐
                     │  Document Ingestion Agent    │  sanitizes untrusted text
                     │  (sanitize_untrusted_text)   │  against prompt injection
                     └─────────────┬─────────────┘
                                   │
                                   ▼
                     ┌───────────────────────────┐
                     │  Data Extraction Agent       │  produces ExtractedField
                     │  (Gemini 3.5 Flash-Lite +    │  (value, confidence,
                     │   deterministic field copy)  │   SourceCitation) per field
                     └─────────────┬─────────────┘
                                   │
                                   ▼
                     ┌───────────────────────────┐
                     │  Evidence Retrieval Agent    │──┬──► External Tool:
                     │                               │  │    query_business_registry
                     │                               │  └──► External Tool:
                     │                               │       check_aml_sanctions_database
                     │  → EvidenceItem[] (citable)   │
                     └─────────────┬─────────────┘
                                   │
                                   ▼
                     ┌───────────────────────────┐
                     │  Fraud Detection Agent       │  Gemini-scored digital
                     │  (Gemini 3.5 Flash-Lite)     │  footprint risk (0.0–1.0)
                     └─────────────┬─────────────┘
                                   │
                                   ▼
                     ┌───────────────────────────┐
                     │  Policy Check Agent          │  runs src/policy.py rules:
                     │  (deterministic, rule-based) │  registry status, AML,
                     │                               │  confidence, fraud-risk
                     │  + Discrepancy Detection      │  threshold + cross-field
                     └─────────────┬─────────────┘  discrepancy checks
                                   │
                                   ▼
                     ┌───────────────────────────┐
                     │  Risk Dossier Agent          │  aggregates checks +
                     │  (deterministic decision)    │  discrepancies + evidence
                     │                               │  into RiskDossier;
                     │                               │  decides escalation
                     └─────────────┬─────────────┘
                                   │
                     ┌─────────────┴─────────────┐
                     ▼                             ▼
          requires_human_review=False    requires_human_review=True
                     │                             │
                     ▼                             ▼
                 [ END ]              [ interrupt_before=["human_review"] ]
           (auto-approved/rejected)                │
                                                     ▼
                                      ┌───────────────────────────┐
                                      │  Compliance Officer Review   │──► Saved durably to
                                      │  & Override (Streamlit / API)│    `checkpoints.sqlite`
                                      └─────────────┬─────────────┘
                                                     │
                                                     ▼
                                      ┌───────────────────────────┐
                                      │    Final Execution & End     │
                                      └───────────────────────────┘
```

---

## 2. Core Components & Technical Mechanics

### 2.1 Shared State Schema (`OnboardingState`, `src/workflow.py`)

| Field | Type | Description |
|---|---|---|
| `merchant_name` | `str` | Registered business name (sanitized after ingestion). |
| `registration_number` | `str` | Company identifier (sanitized after ingestion). |
| `website_url` | `str` | Digital storefront URL (sanitized after ingestion). |
| `extracted_data` | `dict` | Serialized `ExtractedMerchantData` — per-field value, confidence, and `SourceCitation`. |
| `verification_status` | `str` | Combined registry & AML screening status (`passed`, `failed`, `needs_review`). |
| `fraud_risk_score` | `float` | Normalized digital risk metric (0.0–1.0). |
| `evidence` | `Annotated[List[dict], operator.add]` | Serialized `EvidenceItem[]` — raw + summarized tool outputs, citable by source. |
| `policy_checks` | `Annotated[List[dict], operator.add]` | Serialized `PolicyCheckResult[]` from `src/policy.py`. |
| `discrepancies` | `Annotated[List[dict], operator.add]` | Serialized `Discrepancy[]` from cross-field consistency checks. |
| `escalation_reasons` | `Annotated[List[str], operator.add]` | Human-readable reasons the case was escalated (empty if not). |
| `requires_human_review` | `bool` | Set by `risk_dossier_agent`; drives the conditional edge to `human_review`. |
| `dossier` | `dict` | Serialized `RiskDossier` — the final auditable output (see §2.4). |
| `final_decision` | `str` | Final disposition (`approved`, `rejected`, `manual_review`). |
| `audit_trail` | `Annotated[List[str], operator.add]` | Immutable append-only log of every agent action and tool output. |
| `override_reviewer_id` | `str` | Set by the caller (via `update_state`) before resuming past `human_review`; identity of the compliance officer making the decision. |
| `override_reason` | `str` | Optional free-text justification for the override, set alongside `override_reviewer_id`. |
| `review_audit` | `Annotated[List[dict], operator.add]` | Serialized `ReviewAuditEntry[]` — append-only accountability trail of every human decision (who, what, when, why). |

### 2.2 External Tool Function Bindings

The Evidence Retrieval Agent calls two native LangChain tools (`@tool` decorator):

- **`query_business_registry(registration_number)`** — Simulates a government registry lookup (`STATUS_ACTIVE`, `STATUS_PENDING`, `STATUS_INVALID`).
- **`check_aml_sanctions_database(merchant_name)`** — Screens against watchlists/PEP/OFAC (`CLEAR`, `WATCH`, `CRITICAL ALERT`).

Both raw outputs are wrapped into `EvidenceItem` objects (`src/schemas.py`) with `source`, `summary`, `raw_result`, and `retrieved_at`, so every policy check can cite exactly what it relied on.

### 2.3 Policy Engine & Discrepancy Detection (`src/policy.py`)

Policy checks are small, independently unit-tested functions rather than a single LLM judgment call — this is what makes the decision auditable and injection-resistant:

- `check_registry_status(evidence)` → critical fail on `STATUS_INVALID`, medium on `STATUS_PENDING`.
- `check_aml_sanctions(evidence)` → critical fail on `CRITICAL ALERT`, medium on `WATCH`.
- `check_extraction_confidence(extracted)` → fails (high severity) if any field's confidence is below `EXTRACTION_CONFIDENCE_THRESHOLD` (0.75).
- `check_fraud_risk_score(score)` → fails (critical) if the score exceeds `FRAUD_RISK_ESCALATION_THRESHOLD` (0.7).
- `check_fraud_score_consistency(score, registry_evidence, aml_evidence)` → **LLM-output guardrail**: fails (high severity) if deterministic evidence (registry/AML) indicates critical risk but the LLM-generated `fraud_risk_score` claims low risk (below `LLM_LOW_RISK_CLAIM_THRESHOLD`, 0.5). This exists because the fraud score comes from a free-text LLM response parsed as a float, with only a bare `try/except ValueError` as a parsing safety net — nothing previously caught a *plausible-looking but wrong* score (e.g. the model returning `0.1` on an application with an invalid registry and a sanctions match). This check does not fire in the other direction (LLM says high risk on clean evidence); that case already escalates via `check_fraud_risk_score`.
- `detect_discrepancies(extracted)` → rule-based cross-field checks (e.g. UK registration number paired with a non-UK domain; placeholder/test-like registration numbers).

`POLICY_VERSION` (`src/policy.py`) is a version string bumped whenever a rule's logic or a threshold above changes. It's stamped onto every `RiskDossier` (see §2.4) so a later threshold change never leaves a historical decision unexplainable.

### 2.4 The Risk Dossier (`RiskDossier`, `src/schemas.py`)

`risk_dossier_agent` is the only node that decides `final_decision` and `requires_human_review`, and it does so **purely from structured data** (`policy_checks`, `discrepancies`, `fraud_risk_score`, `min_extraction_confidence`) — never from LLM free text. This is the core defense against prompt injection: even if an injected instruction reached an LLM-generated field (e.g. the business summary), it cannot influence the decision because the decision logic never reads that field.

`RiskDossier.escalation_report()` renders a human-readable answer to *"which checks caused escalation, and what evidence supports each"* by walking `policy_checks` (any failed/high/critical), `discrepancies`, the confidence threshold, and (if present) the `review_audit` history — this is surfaced directly in the Streamlit UI and in `eval/run_eval.py`.

Every dossier also carries governance/provenance metadata: `policy_version` (from `src/policy.py`), `model_name` (the pinned `MODEL_NAME` constant in `src/workflow.py`, currently `"gemini-3.5-flash-lite"`), and `evaluated_at` (ISO timestamp). None of these are optional or defaulted away in practice — `risk_dossier_agent` always sets them explicitly.

### 2.5 Governance: Override Accountability (`ReviewAuditEntry`)

A decision alone ("approved") is not an auditable compliance record without knowing *who* made it, *when*, and *why*. Every human-in-the-loop resolution is captured as a `ReviewAuditEntry`:

- A caller resuming the graph past the `human_review` interrupt is expected to set `override_reviewer_id` (required, identity of the reviewer) and `override_reason` (optional, free-text justification) via `update_state`, alongside `final_decision`.
- `human_review_agent` reads these, builds a `ReviewAuditEntry` (reviewer, decision, reason, timestamp), and appends it to **both** the state's `review_audit` channel (`Annotated[List[dict], operator.add]` — append-only, survives across multiple resumes on the same thread) **and** the dossier's own `review_audit` list, so the dossier stays a self-contained, exportable record independent of the raw graph state.
- If no `override_reviewer_id` is set, the entry records `"unspecified_reviewer"` explicitly — accountability gaps are visible in the record rather than silently absent.
- The Streamlit UI enforces this at the input layer: the Approve/Reject buttons are `disabled` until a non-empty reviewer ID is entered.
- Covered by `tests/test_graph_e2e.py::test_override_records_reviewer_identity_and_reason` and `test_override_without_reviewer_id_falls_back_to_unspecified`.

### 2.6 Prompt-Injection Defense

- `sanitize_untrusted_text()` (`src/workflow.py`) redacts known injection patterns (e.g. "ignore previous instructions", "system prompt", "approve this application") from merchant-supplied text **before** it's interpolated into any LLM prompt. Applied in `document_ingestion_agent` (first node) and again to the LLM-generated summary in `data_extraction_agent`.
- Even a missed/novel injection payload cannot flip the outcome, because §2.4's decision logic is deterministic and structured-data-only.
- Covered by `tests/test_prompt_injection.py`, including an end-to-end test that feeds a malicious "APPROVED, risk 0.0" summary into `risk_dossier_agent` alongside real critical policy failures and asserts the decision is still `rejected`.

### 2.7 Durable Persistence & SQLite Checkpointer

State is managed via LangGraph's `SqliteSaver`, connected to a local `checkpoints.sqlite` file. Thread state snapshots, graph execution history, and checkpoint channels persist across restarts, supporting asynchronous resumption after a human-review interrupt.

### 2.8 Human-in-the-Loop (HITL) & Observability

- **Graph Interruption** (`interrupt_before=["human_review"]`) — Halts execution *before* the `human_review` node whenever `risk_dossier_agent` sets `requires_human_review=True`. Clean cases route straight to `END` and never pause.
- **Compliance State Override** — `app.update_state(config, {"final_decision": ...})` followed by `app.invoke(None, config)` resumes the graph past the interrupt with the officer's decision applied. `human_review_agent` is *not* a no-op: it compares the officer's `final_decision` against `dossier["decision"]` and, if they differ, updates the dossier's `decision` field to match and logs the override in `audit_trail`. This matters because `dossier` is otherwise a snapshot frozen at the moment `risk_dossier_agent` first ran — without this sync step, any consumer reading `dossier.decision` (the Streamlit UI, an export, `eval/run_eval.py`) would keep showing the pre-override decision even after resolution. Covered by `tests/test_graph_e2e.py`.
- **LangSmith Integration** — Traces stream automatically when `LANGCHAIN_TRACING_V2=true`, covering LLM latencies, token consumption, and tool execution trees across all seven nodes.

---

## 3. Evaluation (`eval/`)

`eval/dataset.jsonl` holds labeled scenarios (clean approval, critical rejection, medium-severity pending-registry escalation, AML name-match, discrepancy case, and two prompt-injection attempts — one benign, one paired with real fraud signals), each with an `expected_decision` and `expected_requires_human_review`. `eval/run_eval.py` runs each through the compiled graph, diffs the actual outcome against the label, and prints `RiskDossier.escalation_report()` for any escalated case.
