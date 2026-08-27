import os
from dotenv import load_dotenv

# Load environment variables (including LangSmith tracing if enabled)
load_dotenv()

import operator
from typing import Annotated, List, TypedDict
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, StateGraph
from langgraph.checkpoint.sqlite import SqliteSaver

# 1. Define State Schema
class OnboardingState(TypedDict):
    merchant_name: str
    registration_number: str
    website_url: str
    extracted_data: dict
    verification_status: str
    fraud_risk_score: float
    final_decision: str
    audit_trail: Annotated[List[str], operator.add]

# Initialize Gemini 3.5 Flash-Lite
llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite")

# Helper function to extract text safely
def extract_text(response) -> str:
    content = response.content
    if isinstance(content, list):
        return "".join([block.get("text", "") for block in content if isinstance(block, dict)])
    return str(content)

# ==========================================
# EXTERNAL TOOL DEFINITION
# ==========================================
@tool
def query_business_registry(registration_number: str) -> str:
    """
    Queries the official government business registry database 
    to verify if a company registration number is active and legitimate.
    """
    reg_id = registration_number.upper().strip()
    
    if "FAKE" in reg_id or reg_id.startswith("999"):
        return "Registry Result: STATUS_INVALID - Company record not found or flagged as dissolved/fraudulent."
    elif "PENDING" in reg_id:
        return "Registry Result: STATUS_PENDING - Incorporation filing is under manual state review."
    else:
        return "Registry Result: STATUS_ACTIVE - Company is officially registered, active, and in good standing."


@tool
def check_aml_sanctions_database(merchant_name: str) -> str:
    """
    Screens the merchant entity and associated beneficial owners against 
    global watchlists, PEP (Politically Exposed Persons) databases, and OFAC sanctions.
    """
    name_clean = merchant_name.lower().strip()
    
    # Simulate a match on a restricted/sanctioned name
    if "scam" in name_clean or "shadow" in name_clean or "sanctioned" in name_clean:
        return "AML Screening Result: CRITICAL ALERT - Entity matches global sanctions watchlist or restricted list."
    elif "pending" in name_clean or "startup" in name_clean:
        return "AML Screening Result: WATCH - Minor risk flags found; requires enhanced due diligence (EDD)."
    else:
        return "AML Screening Result: CLEAR - No matches found on international watchlists or PEP databases."


# 2. Agent Nodes
def data_extraction_agent(state: OnboardingState):
    prompt = f"""
    Extract and structure the following merchant details:
    Name: {state['merchant_name']}
    Registration ID: {state['registration_number']}
    Website: {state['website_url']}
    """
    response = llm.invoke([HumanMessage(content=prompt)])
    summary_text = extract_text(response).strip()
    
    return {
        "extracted_data": {
            "legal_name": state['merchant_name'], 
            "id": state['registration_number'],
            "summary": summary_text
        },
        "audit_trail": ["Data Extraction Agent: Dynamically parsed merchant details."]
    }

def verification_agent(state: OnboardingState):
    reg_id = state['registration_number']
    merchant_name = state['merchant_name']
    
    # Run both external tools
    registry_output = query_business_registry.invoke({"registration_number": reg_id})
    aml_output = check_aml_sanctions_database.invoke({"merchant_name": merchant_name})
    
    # Determine cumulative status based on tool results
    if "CRITICAL ALERT" in aml_output or "STATUS_INVALID" in registry_output:
        status = "failed"
    elif "STATUS_PENDING" in registry_output or "WATCH" in aml_output:
        status = "needs_review"
    else:
        status = "passed"

    return {
        "verification_status": status,
        "audit_trail": [
            f"Verification Agent: Queried business registry & AML sanctions database.",
            f"Registry Tool Output: {registry_output}",
            f"AML Tool Output: {aml_output}",
            f"Composite verification status resolved as '{status}'."
        ]
    }

def fraud_detection_agent(state: OnboardingState):
    prompt = f"""
    Analyze this merchant's digital footprint for fraud risk:
    Merchant Name: {state['merchant_name']}
    Registration Number: {state['registration_number']}
    Website URL: {state['website_url']}
    
    Rules for scoring:
    - If this is a standard corporate entity with a valid format UK registration and a professional website domain, output a very low risk score (e.g., 0.1).
    - Only output high risk scores (> 0.7) if there are clear scam, fraudulent, or suspicious indicators in the name or URL.
    
    Respond with ONLY a single decimal number between 0.0 and 1.0. No other text.
    """
    response = llm.invoke([HumanMessage(content=prompt)])
    cleaned_response = extract_text(response).strip()
    
    try:
        risk_score = float(cleaned_response)
    except ValueError:
        risk_score = 0.9 if "scam" in state['website_url'] else 0.1
        
    return {
        "fraud_risk_score": risk_score,
        "audit_trail": [f"Fraud Detection Agent: Risk score evaluated at {risk_score}."]
    }

def orchestrator_agent(state: OnboardingState):
    kyc = state['verification_status']
    risk = state['fraud_risk_score']
    
    if kyc == "passed" and risk < 0.4:
        decision = "approved"
    elif kyc == "failed" or risk > 0.7:
        decision = "rejected"
    else:
        decision = "manual_review"
        
    return {
        "final_decision": decision,
        "audit_trail": [f"Orchestrator Agent: Final decision rendered -> {decision.upper()}."]
    }

# 3. Build Graph Workflow
workflow = StateGraph(OnboardingState)
workflow.add_node("extractor", data_extraction_agent)
workflow.add_node("verifier", verification_agent)
workflow.add_node("fraud_checker", fraud_detection_agent)
workflow.add_node("orchestrator", orchestrator_agent)

workflow.set_entry_point("extractor")
workflow.add_edge("extractor", "verifier")
workflow.add_edge("verifier", "fraud_checker")
workflow.add_edge("fraud_checker", "orchestrator")
workflow.add_edge("orchestrator", END)

# ==========================================
# GLOBAL APP COMPILATION (Required for LangGraph Studio & UI)
# ==========================================
# Compile globally with HITL breakpoint after orchestrator
app = workflow.compile(interrupt_after=["orchestrator"])


# ==========================================
# MULTI-SCENARIO TEST SUITE WITH SQLITE
# ==========================================
DB_PATH = "checkpoints.sqlite"

if __name__ == "__main__":
    with SqliteSaver.from_conn_string(DB_PATH) as checkpointer:
        # For local script execution, we can bind the SQLite checkpointer
        app_local = workflow.compile(checkpointer=checkpointer, interrupt_after=["orchestrator"])

        test_scenarios = [
            {
                "id": "scenario_thread_1",
                "name": "Scenario 1: Legitimate Enterprise (Expected: APPROVED)",
                "state": {
                    "merchant_name": "Global Tech Solutions Ltd",
                    "registration_number": "UK-9843210",
                    "website_url": "https://www.globaltechsolutions.co.uk",
                    "extracted_data": {},
                    "verification_status": "",
                    "fraud_risk_score": 0.0,
                    "final_decision": "",
                    "audit_trail": []
                },
                "requires_hitl": False
            },
            {
                "id": "scenario_thread_2",
                "name": "Scenario 2: Fraudulent / Fake Registry (Expected: REJECTED)",
                "state": {
                    "merchant_name": "Quick Cash Scam Corp",
                    "registration_number": "FAKE-999999",
                    "website_url": "https://get-rich-quick-free-money.biz",
                    "extracted_data": {},
                    "verification_status": "",
                    "fraud_risk_score": 0.0,
                    "final_decision": "",
                    "audit_trail": []
                },
                "requires_hitl": False
            },
            {
                "id": "scenario_thread_3",
                "name": "Scenario 3: Pending Startup with HITL Override (Expected: APPROVED via Manual Review)",
                "state": {
                    "merchant_name": "Borderline Ventures LLC",
                    "registration_number": "PENDING-4455",
                    "website_url": "https://www.borderlineventures.io",
                    "extracted_data": {},
                    "verification_status": "",
                    "fraud_risk_score": 0.0,
                    "final_decision": "",
                    "audit_trail": []
                },
                "requires_hitl": True,
                "human_override": "approved"
            }
        ]

        print("Running comprehensive multi-scenario test suite with SQLite persistence...\n" + "="*65)

        for scenario in test_scenarios:
            print(f"\n▶ Executing: {scenario['name']}")
            thread_config = {"configurable": {"thread_id": scenario["id"]}}

            events = list(app_local.stream(scenario["state"], thread_config))
            current_state = app_local.get_state(thread_config)
            
            if scenario["requires_hitl"]:
                print(f"⏸️ Graph hit HITL breakpoint after Orchestrator.")
                print(f"   Initial Recommended Decision: {current_state.values.get('final_decision')}")
                
                override_val = scenario["human_override"]
                print(f"👤 Compliance Officer overrides decision to: {override_val.upper()}")
                
                app_local.update_state(thread_config, {"final_decision": override_val})
                final_output = app_local.invoke(None, thread_config)
            else:
                final_output = current_state.values

            print(f"👉 Final Resolution: {final_output.get('final_decision', 'UNKNOWN').upper()}")
            print(f"   KYC Status: {final_output.get('verification_status')} | Risk Score: {final_output.get('fraud_risk_score')}")
            print("-" * 65)
            
        print(f"\n✨ All scenarios completed successfully! State checkpoints safely saved to {DB_PATH}.")