import sqlite3

import streamlit as st
from dotenv import load_dotenv
from langgraph.checkpoint.sqlite import SqliteSaver

load_dotenv()

st.set_page_config(page_title="Merchant Onboarding Auditor", page_icon="🛡️", layout="centered")

st.title("🛡️ Merchant Onboarding Risk Auditor")
st.markdown(
    "Ingests merchant application data, extracts structured fields with confidence "
    "scores, checks them against policy rules, retrieves supporting evidence, and "
    "produces an auditable risk dossier - escalating to a human reviewer whenever "
    "confidence is low or a policy check fails."
)

from src.workflow import workflow, DB_PATH, _empty_state
from src.schemas import RiskDossier


@st.cache_resource
def get_graph_app():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    checkpointer = SqliteSaver(conn)
    return workflow.compile(checkpointer=checkpointer, interrupt_before=["human_review"])


app = get_graph_app()

with st.sidebar:
    st.header("📝 New Merchant Application")
    merchant_name = st.text_input("Merchant Name", value="Borderline Ventures LLC")
    reg_number = st.text_input("Registration Number", value="PENDING-4455")
    website = st.text_input("Website URL", value="https://www.borderlineventures.io")
    thread_id = st.text_input("Session Thread ID", value="streamlit_thread_001")
    submit_btn = st.button("Run Onboarding Pipeline", type="primary")

if submit_btn:
    with st.spinner("🤖 Running ingestion → extraction → evidence retrieval → policy checks → dossier..."):
        thread_config = {"configurable": {"thread_id": thread_id}}
        initial_state = _empty_state(merchant_name, reg_number, website)
        list(app.stream(initial_state, thread_config))
        st.session_state["thread_id"] = thread_id
        st.success("Pipeline executed. Review the dossier below.")

current_thread_id = st.session_state.get("thread_id", thread_id)
current_config = {"configurable": {"thread_id": current_thread_id}}
state_snapshot = app.get_state(current_config)

if state_snapshot and state_snapshot.values.get("dossier"):
    values = state_snapshot.values
    dossier = RiskDossier(**values["dossier"])
    # The graph is still paused awaiting a human decision only while
    # `human_review` hasn't run yet. `dossier.requires_human_review` is
    # frozen at the value it had when the dossier was first built, so it
    # must NOT be used to decide whether to keep showing the override
    # panel - use the live pending-node info from the checkpoint instead.
    awaiting_review = bool(state_snapshot.next)
    live_decision = values.get("final_decision") or dossier.decision

    st.divider()
    st.subheader("📊 Risk Dossier")

    col1, col2, col3 = st.columns(3)
    col1.metric("KYC Status", values.get("verification_status", "N/A").upper())
    col2.metric("Fraud Risk Score", f"{dossier.fraud_risk_score:.2f}")
    badge = {"approved": "✅ APPROVED", "rejected": "❌ REJECTED", "manual_review": "⚠️ MANUAL REVIEW"}
    col3.metric("Final Decision", badge.get(live_decision, live_decision.upper()))

    if awaiting_review:
        st.warning("⚠️ Escalated for human review.")
        with st.expander("🎯 Why was this escalated? (demo question)", expanded=True):
            st.text(dossier.escalation_report())
    elif dossier.requires_human_review:
        st.info(f"✅ This escalated case was resolved by compliance override: **{live_decision.upper()}**.")

    with st.expander("🧾 Extracted Fields (with confidence & source)"):
        for name, field in [
            ("Legal name", dossier.extracted_data.legal_name),
            ("Registration number", dossier.extracted_data.registration_number),
            ("Website URL", dossier.extracted_data.website_url),
        ]:
            st.write(f"**{name}**: {field.value}  \n"
                     f"confidence={field.confidence:.2f} · source=`{field.source.document}`")

    with st.expander("✅ Policy Checks"):
        for c in dossier.policy_checks:
            icon = "✅" if c.passed else "❌"
            st.write(f"{icon} **{c.check_name}** [{c.severity}] — {c.details}")

    if dossier.discrepancies:
        with st.expander("⚠️ Discrepancies"):
            for d in dossier.discrepancies:
                st.write(f"- **{d.field}** [{d.severity}]: expected `{d.expected}`, found `{d.found}` — {d.explanation}")

    with st.expander("🔗 Evidence"):
        for e in dossier.evidence:
            st.write(f"- **{e.source}**: {e.summary}  \n  _raw: {e.raw_result}_")

    with st.expander("📜 Full Audit Trail", expanded=False):
        for step in values.get("audit_trail", []):
            st.write(f"- {step}")

    if dossier.review_audit:
        with st.expander("🕵️ Review Audit Log (who decided, when, why)", expanded=True):
            for entry in dossier.review_audit:
                reason_suffix = f" — _{entry.reason}_" if entry.reason else ""
                st.write(f"- **{entry.timestamp}** · reviewer=`{entry.reviewer_id}` · decision=**{entry.decision}**{reason_suffix}")

    st.caption(f"Policy v{dossier.policy_version} · model `{dossier.model_name}` · evaluated {dossier.evaluated_at}")

    if awaiting_review:
        st.divider()
        st.subheader("👤 Compliance Officer Override")
        st.caption("A reviewer identity is required for every override so the decision is accountable.")
        reviewer_id = st.text_input("Your name / reviewer ID (required)", key="reviewer_id_input")
        override_reason = st.text_area("Reason for this decision (optional but recommended)", key="override_reason_input")
        reviewer_provided = bool(reviewer_id.strip())
        if not reviewer_provided:
            st.caption("⚠️ Enter your reviewer ID above to enable the override buttons.")

        col_app, col_rej = st.columns(2)
        if col_app.button("✅ Approve Application", use_container_width=True, disabled=not reviewer_provided):
            app.update_state(
                current_config,
                {
                    "final_decision": "approved",
                    "override_reviewer_id": reviewer_id.strip(),
                    "override_reason": override_reason.strip(),
                },
            )
            app.invoke(None, current_config)
            st.success(f"Application overridden to APPROVED by {reviewer_id.strip()}.")
            st.rerun()
        if col_rej.button("❌ Confirm Rejection", use_container_width=True, disabled=not reviewer_provided):
            app.update_state(
                current_config,
                {
                    "final_decision": "rejected",
                    "override_reviewer_id": reviewer_id.strip(),
                    "override_reason": override_reason.strip(),
                },
            )
            app.invoke(None, current_config)
            st.error(f"Rejection confirmed by {reviewer_id.strip()}.")
            st.rerun()
else:
    st.info("👈 Fill out the merchant details in the sidebar and click **Run Onboarding Pipeline** to start.")
