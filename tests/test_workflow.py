import pytest
from src.workflow import (
    query_business_registry,
    check_aml_sanctions_database,
    data_extraction_agent,
    verification_agent,
    fraud_detection_agent,
    orchestrator_agent,
    OnboardingState
)

# 1. Test External Tool Functions
def test_query_business_registry_active():
    result = query_business_registry.invoke({"registration_number": "UK-123456"})
    assert "STATUS_ACTIVE" in result

def test_query_business_registry_invalid():
    result = query_business_registry.invoke({"registration_number": "FAKE-999"})
    assert "STATUS_INVALID" in result

def test_check_aml_sanctions_database_clear():
    result = check_aml_sanctions_database.invoke({"merchant_name": "Safe Commerce Ltd"})
    assert "CLEAR" in result

def test_check_aml_sanctions_database_alert():
    result = check_aml_sanctions_database.invoke({"merchant_name": "Shadow Scam Corp"})
    assert "CRITICAL ALERT" in result


# 2. Test Orchestration Decision Logic Directly
@pytest.mark.parametrize("kyc_status, risk_score, expected_decision", [
    ("passed", 0.1, "approved"),
    ("failed", 0.2, "rejected"),
    ("passed", 0.85, "rejected"),
    ("needs_review", 0.2, "manual_review"),
    ("passed", 0.5, "manual_review"),
])
def test_orchestrator_decision_matrix(kyc_status, risk_score, expected_decision):
    state: OnboardingState = {
        "merchant_name": "Test Merchant",
        "registration_number": "UK-000",
        "website_url": "https://test.com",
        "extracted_data": {},
        "verification_status": kyc_status,
        "fraud_risk_score": risk_score,
        "final_decision": "",
        "audit_trail": []
    }
    
    result = orchestrator_agent(state)
    assert result["final_decision"] == expected_decision