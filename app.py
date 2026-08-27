import streamlit as st
import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3

load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Merchant Onboarding AI Agent",
    page_icon="🛡️",
    layout="centered"
)

st.title("🛡️ Autonomous Merchant Onboarding & Risk Workflow")
st.markdown("Powered by **Google Gemini 3.5 Flash-Lite**, **LangGraph**, and **Human-in-the-Loop (HITL)** oversight.")

# Import workflow components from src.workflow
# Make sure your workflow.py exports `workflow` or `app`
from src.workflow import workflow, DB_PATH

# Initialize Checkpointer and App for Streamlit session
@st.cache_resource
def get_graph_app():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    checkpointer = SqliteSaver(conn)
    return workflow.compile(checkpointer=checkpointer, interrupt_after=["orchestrator"])

app = get_graph_app()

# Sidebar for Input Form
with st.sidebar:
    st.header("📝 New Merchant Application")
    merchant_name = st.text_input("Merchant Name", value="Borderline Ventures LLC")
    reg_number = st.text_input("Registration Number", value="PENDING-4455")
    website = st.text_input("Website URL", value="https://www.borderlineventures.io")
    
    thread_id = st.text_input("Session Thread ID", value="streamlit_thread_001")
    
    submit_btn = st.button("Run Onboarding Pipeline", type="primary")

# Main Interface Area
if submit_btn:
    with st.spinner("🤖 Running multi-agent evaluation..."):
        thread_config = {"configurable": {"thread_id": thread_id}}
        
        initial_state = {
            "merchant_name": merchant_name,
            "registration_number": reg_number,
            "website_url": website,
            "extracted_data": {},
            "verification_status": "",
            "fraud_risk_score": 0.0,
            "final_decision": "",
            "audit_trail": []
        }
        
        # Stream the execution
        events = list(app.stream(initial_state, thread_config))
        st.session_state['thread_id'] = thread_id
        st.success("Pipeline executed successfully up to evaluation stage!")

# Check current state of the thread to see if it's paused at HITL
current_thread_id = st.session_state.get('thread_id', thread_id)
current_config = {"configurable": {"thread_id": current_thread_id}}
state_snapshot = app.get_state(current_config)

if state_snapshot and state_snapshot.values.get("final_decision"):
    values = state_snapshot.values
    decision = values.get("final_decision")
    
    st.divider()
    st.subheader("📊 Onboarding Evaluation Results")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("KYC Status", values.get("verification_status", "N/A").upper())
    col2.metric("Fraud Risk Score", f"{values.get('fraud_risk_score', 0.0):.2f}")
    
    # Status styling badge
    if decision == "approved":
        col3.metric("Final Decision", "✅ APPROVED")
    elif decision == "rejected":
        col3.metric("Final Decision", "❌ REJECTED")
    else:
        col3.metric("Final Decision", "⚠️ MANUAL REVIEW")

    with st.expander("🔍 View Complete Agent Audit Trail", expanded=True):
        for step in values.get("audit_trail", []):
            st.write(f"- {step}")

    # HITL Compliance Intervention Panel if status is manual_review or if user wants to override
    if decision == "manual_review" or decision == "rejected":
        st.warning("⚠️ This application requires compliance oversight or can be manually overridden.")
        
        col_app, col_rej = st.columns(2)
        if col_app.button("✅ Approve Application (Compliance Override)", use_container_width=True):
            app.update_state(current_config, {"final_decision": "approved"})
            app.invoke(None, current_config)
            st.success("Application successfully overridden to APPROVED!")
            st.rerun()
            
        if col_rej.button("❌ Confirm Rejection", use_container_width=True):
            app.update_state(current_config, {"final_decision": "rejected"})
            app.invoke(None, current_config)
            st.error("Application final rejection confirmed.")
            st.rerun()
else:
    st.info("👈 Fill out the merchant details in the sidebar and click **Run Onboarding Pipeline** to start.")