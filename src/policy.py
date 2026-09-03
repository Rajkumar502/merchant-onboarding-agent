"""
Policy rules and discrepancy detection for merchant onboarding.

Each policy check is a small, independently testable function that takes
the extracted data plus evidence gathered from external systems and
returns a PolicyCheckResult. Keeping these as discrete functions (rather
than one big LLM prompt) is what makes the workflow auditable: every
check has a name, a pass/fail, a severity, and cites the evidence it
relied on.
"""
from __future__ import annotations

from typing import List
from urllib.parse import urlparse

from src.schemas import (
    Discrepancy,
    EvidenceItem,
    ExtractedMerchantData,
    PolicyCheckResult,
)

# Confidence below this threshold on any extracted field forces human review,
# regardless of how the other checks resolve.
EXTRACTION_CONFIDENCE_THRESHOLD = 0.75

# Fraud risk score above this threshold is an automatic escalation trigger.
FRAUD_RISK_ESCALATION_THRESHOLD = 0.7

# If deterministic evidence (registry/AML) indicates high risk but the
# LLM-generated fraud_risk_score claims otherwise, the disagreement itself
# is a signal worth surfacing - not silently trusting whichever number came
# last. This threshold defines "the LLM claimed low risk" for that check.
LLM_LOW_RISK_CLAIM_THRESHOLD = 0.5

# Bumped whenever a rule's logic or a threshold above changes, so every
# RiskDossier can record exactly which policy ruleset produced its
# decision (see src/schemas.py RiskDossier.policy_version).
POLICY_VERSION = "1.1.0"


def check_registry_status(registry_evidence: EvidenceItem) -> PolicyCheckResult:
    raw = registry_evidence.raw_result
    if "STATUS_INVALID" in raw:
        return PolicyCheckResult(
            check_name="business_registry_status",
            description="Registration number must resolve to an active, legitimate business registry record.",
            passed=False,
            severity="critical",
            details="Business registry record is invalid, dissolved, or flagged as fraudulent.",
            evidence=[registry_evidence],
        )
    if "STATUS_PENDING" in raw:
        return PolicyCheckResult(
            check_name="business_registry_status",
            description="Registration number must resolve to an active, legitimate business registry record.",
            passed=False,
            severity="medium",
            details="Registration is under manual state review; not yet confirmed active.",
            evidence=[registry_evidence],
        )
    return PolicyCheckResult(
        check_name="business_registry_status",
        description="Registration number must resolve to an active, legitimate business registry record.",
        passed=True,
        severity="low",
        details="Registry confirms the business is active and in good standing.",
        evidence=[registry_evidence],
    )


def check_aml_sanctions(aml_evidence: EvidenceItem) -> PolicyCheckResult:
    raw = aml_evidence.raw_result
    if "CRITICAL ALERT" in raw:
        return PolicyCheckResult(
            check_name="aml_sanctions_screening",
            description="Merchant and beneficial owners must not match sanctions/PEP watchlists.",
            passed=False,
            severity="critical",
            details="Entity matches a global sanctions watchlist or restricted list.",
            evidence=[aml_evidence],
        )
    if "WATCH" in raw:
        return PolicyCheckResult(
            check_name="aml_sanctions_screening",
            description="Merchant and beneficial owners must not match sanctions/PEP watchlists.",
            passed=False,
            severity="medium",
            details="Minor risk flags found; enhanced due diligence required.",
            evidence=[aml_evidence],
        )
    return PolicyCheckResult(
        check_name="aml_sanctions_screening",
        description="Merchant and beneficial owners must not match sanctions/PEP watchlists.",
        passed=True,
        severity="low",
        details="No matches found on international watchlists or PEP databases.",
        evidence=[aml_evidence],
    )


def check_extraction_confidence(extracted: ExtractedMerchantData) -> PolicyCheckResult:
    min_conf = extracted.min_confidence()
    passed = min_conf >= EXTRACTION_CONFIDENCE_THRESHOLD
    return PolicyCheckResult(
        check_name="extraction_confidence",
        description=f"All extracted fields must meet a confidence threshold of {EXTRACTION_CONFIDENCE_THRESHOLD}.",
        passed=passed,
        severity="low" if passed else "high",
        details=(
            f"Minimum field confidence was {min_conf:.2f}."
            + ("" if passed else " Below threshold; requires manual verification against source documents.")
        ),
        evidence=[],
    )


def check_fraud_risk_score(fraud_risk_score: float) -> PolicyCheckResult:
    passed = fraud_risk_score <= FRAUD_RISK_ESCALATION_THRESHOLD
    return PolicyCheckResult(
        check_name="fraud_risk_score",
        description=f"Fraud risk score must not exceed {FRAUD_RISK_ESCALATION_THRESHOLD}.",
        passed=passed,
        severity="low" if passed else "critical",
        details=f"Digital footprint fraud risk score evaluated at {fraud_risk_score:.2f}.",
        evidence=[],
    )


def check_fraud_score_consistency(
    fraud_risk_score: float,
    registry_evidence: EvidenceItem,
    aml_evidence: EvidenceItem,
) -> PolicyCheckResult:
    """
    Guardrail against a bad/hallucinated LLM fraud score: cross-checks the
    LLM-generated `fraud_risk_score` against the two deterministic, tool-
    sourced signals we already trust (registry status, AML screening). If
    the deterministic evidence says "critical" but the LLM claims low risk,
    that disagreement is itself flagged - we do not let a single low LLM
    number silently override real evidence of fraud. This never overrides
    `check_fraud_risk_score` in the other direction (LLM says high risk,
    evidence is clean); that case is left to escalate through the ordinary
    high-fraud-score path, which already forces review.
    """
    deterministic_critical = (
        "STATUS_INVALID" in registry_evidence.raw_result or "CRITICAL ALERT" in aml_evidence.raw_result
    )
    llm_claims_low_risk = fraud_risk_score < LLM_LOW_RISK_CLAIM_THRESHOLD

    if deterministic_critical and llm_claims_low_risk:
        return PolicyCheckResult(
            check_name="fraud_score_consistency",
            description=(
                "The LLM-generated fraud risk score must not contradict deterministic "
                "evidence (registry status, AML screening) from external tools."
            ),
            passed=False,
            severity="high",
            details=(
                f"Deterministic evidence indicates critical risk (registry/AML), but the "
                f"fraud-scoring model returned a low score ({fraud_risk_score:.2f}). Treating "
                "this as a possible model error rather than trusting the score at face value."
            ),
            evidence=[registry_evidence, aml_evidence],
        )

    return PolicyCheckResult(
        check_name="fraud_score_consistency",
        description=(
            "The LLM-generated fraud risk score must not contradict deterministic "
            "evidence (registry status, AML screening) from external tools."
        ),
        passed=True,
        severity="low",
        details="Fraud risk score is consistent with (or more cautious than) deterministic evidence.",
        evidence=[],
    )


def detect_discrepancies(extracted: ExtractedMerchantData) -> List[Discrepancy]:
    """
    Cross-checks extracted fields against each other for internal
    consistency. This is deliberately simple/rule-based so it's testable
    and explainable; a production system would layer in more checks
    (e.g. address/jurisdiction matching, name-similarity fuzzy matching).
    """
    discrepancies: List[Discrepancy] = []

    reg_number = extracted.registration_number.value.upper()
    website = extracted.website_url.value.lower()

    try:
        domain = urlparse(website).netloc or website
    except Exception:
        domain = website

    # Example rule: a UK registration number paired with a non-UK-style domain
    # is not necessarily wrong, but it's worth flagging for a human to confirm.
    if reg_number.startswith("UK-") and not (domain.endswith(".uk") or domain.endswith(".co.uk")):
        discrepancies.append(
            Discrepancy(
                field="website_url",
                expected="Domain consistent with UK registration (e.g. .co.uk / .uk)",
                found=domain,
                severity="low",
                explanation=(
                    "Registration number indicates a UK entity, but the website domain "
                    "does not use a UK TLD. Not necessarily fraudulent, but worth confirming."
                ),
            )
        )

    # Placeholder/suspicious registration numbers.
    if "FAKE" in reg_number or "TEST" in reg_number:
        discrepancies.append(
            Discrepancy(
                field="registration_number",
                expected="A well-formed, real registration identifier",
                found=extracted.registration_number.value,
                severity="critical",
                explanation="Registration number contains placeholder/test-like text, suggesting fabricated data.",
            )
        )

    return discrepancies
