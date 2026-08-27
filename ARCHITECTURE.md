# System Architecture & Technical Design

This document details the architectural design, agent interactions, multi-tool function bindings, local SQLite persistence, and Human-in-the-Loop (HITL) state lifecycle for the **Merchant Onboarding & Risk Workflow**.

---

## 1. High-Level Architecture Diagram

```
                     ┌───────────────────────────┐
                     │   MERCHANT INPUT PAYLOAD    │
                     └─────────────┬─────────────┘
                                   │
                                   ▼
                     ┌───────────────────────────┐
                     │   State Initialization      │
                     │   (OnboardingState)         │
                     └─────────────┬─────────────┘
                                   │
                                   ▼
                     ┌───────────────────────────┐
                     │  Data Extraction Agent       │
                     │  (Gemini 3.5 Flash-Lite)     │
                     └─────────────┬─────────────┘
                                   │
                                   ▼
                     ┌───────────────────────────┐
                     │   Verification Agent         │──┬──► External Tool:
                     │                               │  │    query_business_registry
                     │                               │  └──► External Tool:
                     │                               │       check_aml_sanctions_database
                     └─────────────┬─────────────┘
                                   │
                                   ▼
                     ┌───────────────────────────┐
                     │  Fraud & Risk Agent          │
                     │  (Gemini 3.5 Flash-Lite)     │
                     └─────────────┬─────────────┘
                                   │
                                   ▼
                     ┌───────────────────────────┐
                     │  Orchestrator Agent          │
                     └─────────────┬─────────────┘
                                   │
                     [ HITL Checkpoint / Interrupt ]
                             (interrupt_after)
                                   │
                                   ▼
                     ┌───────────────────────────┐
                     │  Compliance Review & State   │──► Saved durably to
                     │  Override (Streamlit / API)  │    `checkpoints.sqlite`
                     └─────────────┬─────────────┘
                                   │
                                   ▼
                     ┌───────────────────────────┐
                     │    Final Execution & End     │
                     └───────────────────────────┘
```

---

## 2. Core Components & Technical Mechanics

### 2.1 Shared State Schema (`OnboardingState`)

The system passes a strongly-typed dictionary object across the computational graph nodes:

| Field | Type | Description |
|---|---|---|
| `merchant_name` | `str` | Registered business name. |
| `registration_number` | `str` | Company identifier. |
| `website_url` | `str` | Digital storefront URL. |
| `extracted_data` | `dict` | Normalized entity attributes. |
| `verification_status` | `str` | Combined registry & AML screening status (`passed`, `failed`, `needs_review`). |
| `fraud_risk_score` | `float` | Normalized digital risk metric (0.0 to 1.0). |
| `final_decision` | `str` | Final disposition (`approved`, `rejected`, `manual_review`). |
| `audit_trail` | `Annotated[List[str], operator.add]` | Immutable append-only log recording actions, multi-tool outputs, and decisions. |

### 2.2 External Tool Function Bindings

The Verification Agent is equipped with two native LangChain tools (`@tool` decorator):

- **`query_business_registry(registration_number)`** — Simulates enterprise database lookups against corporate registries (`STATUS_ACTIVE`, `STATUS_PENDING`, `STATUS_INVALID`).
- **`check_aml_sanctions_database(merchant_name)`** — Screens entities against global watchlists, PEP databases, and international sanctions (`CLEAR`, `WATCH`, `CRITICAL ALERT`).

### 2.3 Durable Persistence & SQLite Checkpointer

State is managed via LangGraph's `SqliteSaver`:

- Connects to a local SQLite database (`checkpoints.sqlite`).
- Persists thread state snapshots, graph execution history, and checkpoint channels to disk, surviving script restarts and supporting asynchronous workflow resumption.

### 2.4 Human-in-the-Loop (HITL) & Observability

- **Graph Interruption** (`interrupt_after=["orchestrator"]`) — Halts execution immediately after the orchestrator renders its initial recommendation on ambiguous applications.
- **Compliance State Override** — Allows manual updates via `app.update_state()` and graph continuation via `app.invoke(None, config)`.
- **LangSmith Integration** — Automatically streams traces (`LANGCHAIN_TRACING_V2=true`) to monitor LLM latencies, token consumption, and tool execution trees.