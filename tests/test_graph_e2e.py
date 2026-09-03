"""
End-to-end tests for the compiled LangGraph `workflow`, using a mocked LLM
so these run offline/in CI without an API key or network access.

Covers three things unit tests on individual nodes can't:
1. The full node sequence actually wires together correctly.
2. A clean case really does skip the human_review interrupt and reach END.
3. An escalated case really does pause before human_review, and - this is
   the regression test for a real bug we hit - overriding the decision via
   `update_state` + resuming actually syncs `dossier.decision`, not just
   the top-level `final_decision` field.
"""
from typing import List

import pytest
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver

from src.workflow import workflow, _empty_state


class _FakeResponse:
    def __init__(self, content: str):
        self.content = content


class _FakeLLM:
    """
    Minimal stand-in for ChatGoogleGenerativeAI. Distinguishes the two
    prompt shapes used in the graph (fraud-risk scoring vs. business
    summary) by content, so both nodes get plausible canned responses.
    """

    def invoke(self, messages: List[HumanMessage]):
        text = messages[0].content
        if "Respond with ONLY a single decimal number" in text:
            # Only inspect the merchant-supplied data lines, not the
            # instructional boilerplate (which itself contains words like
            # "scam"/"fraudulent" as part of the scoring rules).
            data_lines = "\n".join(
                line for line in text.splitlines()
                if line.strip().startswith(("Merchant Name:", "Registration Number:", "Website URL:"))
            ).lower()
            if any(k in data_lines for k in ("scam", "fake", "sanctioned", "shadow")):
                return _FakeResponse("0.9")
            return _FakeResponse("0.1")
        return _FakeResponse("Mock business summary for testing.")


@pytest.fixture
def stub_llm(monkeypatch):
    monkeypatch.setattr("src.workflow.llm", _FakeLLM())


@pytest.fixture
def graph_app():
    checkpointer = MemorySaver()
    return workflow.compile(checkpointer=checkpointer, interrupt_before=["human_review"])


def test_clean_case_auto_approves_without_pausing(stub_llm, graph_app):
    config = {"configurable": {"thread_id": "e2e_clean"}}
    initial_state = _empty_state("Global Tech Solutions Ltd", "UK-9843210", "https://www.globaltechsolutions.co.uk")

    list(graph_app.stream(initial_state, config))
    snapshot = graph_app.get_state(config)

    assert snapshot.next == ()  # graph ran to completion, never paused
    assert snapshot.values["final_decision"] == "approved"
    assert snapshot.values["requires_human_review"] is False
    assert snapshot.values["dossier"]["decision"] == "approved"


def test_dossier_is_stamped_with_policy_version_and_model_name(stub_llm, graph_app):
    """Governance requirement: every dossier must record which policy ruleset and model produced it."""
    config = {"configurable": {"thread_id": "e2e_stamping"}}
    initial_state = _empty_state("Global Tech Solutions Ltd", "UK-9843210", "https://www.globaltechsolutions.co.uk")

    list(graph_app.stream(initial_state, config))
    dossier = graph_app.get_state(config).values["dossier"]

    assert dossier["policy_version"] and dossier["policy_version"] != "unspecified"
    assert dossier["model_name"] and dossier["model_name"] != "unspecified"
    assert dossier["evaluated_at"]  # non-empty ISO timestamp


def test_critical_case_rejects_and_still_pauses_for_review(stub_llm, graph_app):
    config = {"configurable": {"thread_id": "e2e_critical"}}
    initial_state = _empty_state("Quick Cash Scam Corp", "FAKE-999999", "https://get-rich-quick-free-money.biz")

    list(graph_app.stream(initial_state, config))
    snapshot = graph_app.get_state(config)

    # Critical failures escalate (requires_human_review=True) even though
    # the preliminary decision is already "rejected" - the graph still
    # pauses so a human can review the dossier before it's final.
    assert snapshot.next == ("human_review",)
    assert snapshot.values["final_decision"] == "rejected"
    assert snapshot.values["requires_human_review"] is True


def test_human_review_override_syncs_dossier_decision(stub_llm, graph_app):
    """
    Regression test: previously, overriding `final_decision` via
    update_state + resuming past the human_review interrupt updated the
    top-level state field but left `dossier["decision"]` frozen at its
    original (pre-override) value, which made the Streamlit UI (and any
    other consumer reading the dossier) show stale information.
    """
    config = {"configurable": {"thread_id": "e2e_override"}}
    initial_state = _empty_state("Borderline Ventures LLC", "PENDING-4455", "https://www.borderlineventures.io")

    list(graph_app.stream(initial_state, config))
    snapshot = graph_app.get_state(config)
    assert snapshot.next == ("human_review",)
    original_decision = snapshot.values["dossier"]["decision"]
    assert original_decision == "manual_review"

    # Compliance officer overrides the decision and resumes the graph.
    graph_app.update_state(config, {"final_decision": "approved"})
    graph_app.invoke(None, config)

    final_snapshot = graph_app.get_state(config)
    assert final_snapshot.next == ()  # resolved, no longer paused
    assert final_snapshot.values["final_decision"] == "approved"
    assert final_snapshot.values["dossier"]["decision"] == "approved"  # synced, not stale


def test_human_review_confirmation_without_override_keeps_decision(stub_llm, graph_app):
    """If the officer just confirms the recommended decision (no change), the dossier should stay consistent too."""
    config = {"configurable": {"thread_id": "e2e_confirm"}}
    initial_state = _empty_state("Borderline Ventures LLC", "PENDING-4455", "https://www.borderlineventures.io")

    list(graph_app.stream(initial_state, config))
    snapshot = graph_app.get_state(config)
    recommended = snapshot.values["final_decision"]

    graph_app.update_state(config, {"final_decision": recommended})
    graph_app.invoke(None, config)

    final_snapshot = graph_app.get_state(config)
    assert final_snapshot.values["final_decision"] == recommended
    assert final_snapshot.values["dossier"]["decision"] == recommended


def test_override_records_reviewer_identity_and_reason(stub_llm, graph_app):
    """
    Governance requirement: an override must be attributable to a specific
    reviewer, with an optional reason, recorded in both the state's
    review_audit channel and the dossier's own review_audit list (so the
    dossier remains self-contained/exportable).
    """
    config = {"configurable": {"thread_id": "e2e_review_audit"}}
    initial_state = _empty_state("Borderline Ventures LLC", "PENDING-4455", "https://www.borderlineventures.io")

    list(graph_app.stream(initial_state, config))

    graph_app.update_state(
        config,
        {
            "final_decision": "approved",
            "override_reviewer_id": "jane.doe@example.com",
            "override_reason": "Verified registration manually with the state registry.",
        },
    )
    graph_app.invoke(None, config)

    final_state = graph_app.get_state(config).values
    assert len(final_state["review_audit"]) == 1
    entry = final_state["review_audit"][0]
    assert entry["reviewer_id"] == "jane.doe@example.com"
    assert entry["decision"] == "approved"
    assert "state registry" in entry["reason"]
    assert entry["timestamp"]

    # The dossier carries its own copy too, so it's a self-contained record.
    dossier_entries = final_state["dossier"]["review_audit"]
    assert len(dossier_entries) == 1
    assert dossier_entries[0]["reviewer_id"] == "jane.doe@example.com"


def test_override_without_reviewer_id_falls_back_to_unspecified(stub_llm, graph_app):
    """
    If a caller resumes the graph without setting override_reviewer_id
    (e.g. a bug in a calling script), the audit trail should say so
    explicitly rather than silently recording an empty/missing reviewer.
    """
    config = {"configurable": {"thread_id": "e2e_review_audit_missing"}}
    initial_state = _empty_state("Borderline Ventures LLC", "PENDING-4455", "https://www.borderlineventures.io")

    list(graph_app.stream(initial_state, config))
    graph_app.update_state(config, {"final_decision": "approved"})  # no reviewer id set
    graph_app.invoke(None, config)

    entry = graph_app.get_state(config).values["review_audit"][0]
    assert entry["reviewer_id"] == "unspecified_reviewer"
