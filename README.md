# Autonomous Multi-Agent Merchant Onboarding & Risk Workflow

An enterprise-grade, stateful multi-agent AI workflow built with **LangGraph** and **Google Gemini 3.5 Flash-Lite** featuring **External Tool Function Calling (Registry & AML Screening)**, **Durable SQLite Persistence**, **LangSmith Observability**, **Human-in-the-Loop (HITL) Checkpoints**, **Automated Unit Testing (`pytest`)**, **Containerization (`Docker`)**, and an **Interactive Streamlit Web UI**.

---

## 📌 Overview

Traditional B2B merchant onboarding requires balancing automated velocity with stringent risk mitigation. This project implements a stateful orchestration architecture where specialized AI agents collaborate to evaluate merchant applications, query business registries, screen global watchlists, quantify digital risk, and pause for compliance officer intervention when edge cases or policy overrides are required.

---

## 🚀 Key Features

- **Multi-Agent Orchestration** — Decoupled specialized agent nodes managed through a deterministic computational graph via LangGraph.
- **LLM-Powered Extraction** — Dynamically parses and normalizes unstructured business inputs.
- **Multi-Tool Function Calling** — Agents utilize native tool bindings (`@tool`) to concurrently query official government business registries and global **AML / Sanctions Watchlists / PEP databases**.
- **Digital Risk & Fraud Scoring** — Conducts semantic evaluation of website footprints to calculate precise risk metrics.
- **Human-in-the-Loop (HITL) Interruption** — Persists graph states via local `SqliteSaver`, allowing compliance officers to inspect, pause, and override decisions mid-execution.
- **Full Observability & Tracing** — Out-of-the-box integration with **LangSmith** to monitor LLM token usage, latencies, and node execution graphs.
- **Interactive Frontend UI** — Built with **Streamlit** for real-time application submission, audit trail visualization, and interactive compliance override controls.
- **Automated Testing Suite** — Robust test coverage via `pytest` verifying tool mock behavior and orchestrator decision matrices.
- **Containerized Deployment** — Fully dockerized setup (`Dockerfile` & `docker-compose.yml`) for seamless cross-platform execution.

---

## 🛠️ Tech Stack

- **Language:** Python 3.10+
- **LLM:** Google Gemini 3.5 Flash-Lite (`gemini-3.5-flash-lite`)
- **Orchestration Framework:** [LangGraph](https://github.com/langchain-ai/langgraph) / LangChain Core
- **Persistence & State:** LangGraph SQLite Checkpointer (`langgraph-checkpoint-sqlite`)
- **Observability:** LangSmith Tracing (`langsmith`)
- **Interactive UI:** Streamlit (`streamlit`)
- **Testing:** Pytest (`pytest`)
- **Containerization:** Docker & Docker Compose

---

## 📂 Project Structure

```text
merchant-onboarding-agent/
├── src/
│   ├── __init__.py
│   └── workflow.py         # Core LangGraph definition, tools, agent nodes & multi-scenario runner
├── tests/
│   └── test_workflow.py    # Automated unit tests for tools and orchestrator decision matrix
├── app.py                   # Streamlit Interactive Web Dashboard
├── checkpoints.sqlite         # Durable local SQLite state database
├── .env                        # Environment variables (Gemini & LangSmith API keys)
├── langgraph.json               # LangGraph Studio dev configuration
├── Dockerfile                    # Container build instructions
├── docker-compose.yml             # Multi-service orchestration configuration
├── pytest.ini                      # Pytest configuration settings
├── requirements.txt                 # Project dependencies
├── ARCHITECTURE.md                   # Technical design & state lifecycle documentation
└── README.md                          # Project documentation
```

---

## ⚡ Quickstart Guide

### Option A: Run Locally (Python Virtual Environment)

**1. Clone & Set Up Virtual Environment**

```bash
git clone <your-repo-url>
cd merchant-onboarding-agent

python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

**2. Configure API Credentials**

Create a `.env` file in the root directory:

```
GEMINI_API_KEY=your_gemini_api_key_here
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_langsmith_api_key_here
LANGCHAIN_PROJECT=merchant-onboarding-agent
```

**3. Run Unit Tests**

```bash
pytest
```

**4. Launch the Streamlit UI**

```bash
streamlit run app.py
```

### Option B: Run via Docker (Recommended for Zero-Config Deployment)

1. Make sure your `.env` file is configured in the root directory.
2. Build and run containers using Docker Compose:

```bash
docker compose up --build
```

3. Open your browser and navigate to `http://localhost:8501`.

---

## 🔮 Future Roadmap

- [ ] Upgrade persistence layer to PostgreSQL (`AsyncPostgresSaver`) for multi-tenant server scaling.
- [ ] Integrate live external REST APIs (e.g., Companies House, OpenCorporates) into tool definitions.
- [ ] Add automated document OCR parsing for uploaded corporate identity proofs.