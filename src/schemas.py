"""
Pydantic schemas for the merchant onboarding auditor.

These models give every downstream consumer (UI, eval harness, human
reviewers) a stable, typed contract for what the graph produces at each
stage: extracted fields (with confidence + source citation), evidence
gathered from external checks, policy check results, discrepancies, and
the final auditable risk dossier.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import List, Literal, Optional

from pydantic import BaseModel, Field

Severity = Literal["low", "medium", "high", "critical"]
Decision = Literal["approved", "rejected", "manual_review"]


class SourceCitation(BaseModel):
    """Where a piece of information came from, for auditability."""
    document: str = Field(..., description="Name/type of the source document or system, e.g. 'application_form', 'business_registry'")
    field: str = Field(..., description="The field or excerpt this citation supports")
    snippet: str = Field(..., description="Short verbatim-ish excerpt or system response backing the value")


class ExtractedField(BaseModel):
    """A single structured field pulled from merchant documents."""
    value: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    source: SourceCitation


class ExtractedMerchantData(BaseModel):
    legal_name: ExtractedField
    registration_number: ExtractedField
    website_url: ExtractedField
    business_summary: Optional[ExtractedField] = None

    def min_confidence(self) -> float:
        """
        Confidence gate covers the compliance-critical identity fields only.
        `business_summary` is an LLM-generated paraphrase used for reviewer
        context, not a fact pulled from a document, so it deliberately does
        not participate in the extraction-confidence threshold.
        """
        fields = [self.legal_name, self.registration_number, self.website_url]
        return min(f.confidence for f in fields)


class EvidenceItem(BaseModel):
    """A piece of supporting evidence gathered during verification."""
    source: str = Field(..., description="System or database queried, e.g. 'business_registry', 'aml_sanctions_db'")
    summary: str = Field(..., description="Human-readable summary of what was found")
    raw_result: str = Field(..., description="Raw/verbatim tool output, for audit purposes")
    retrieved_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class PolicyCheckResult(BaseModel):
    """Result of evaluating one policy rule against the merchant application."""
    check_name: str
    description: str
    passed: bool
    severity: Severity
    details: str
    evidence: List[EvidenceItem] = Field(default_factory=list)


class Discrepancy(BaseModel):
    """A mismatch found between extracted data, evidence, and/or policy expectations."""
    field: str
    expected: str
    found: str
    severity: Severity
    explanation: str


class ReviewAuditEntry(BaseModel):
    """
    Records who made a human-in-the-loop decision, when, and why. This is
    the accountability trail for compliance overrides: a decision alone
    ("approved") is not auditable without knowing who approved it and when.
    """
    reviewer_id: str = Field(..., description="Identity of the compliance officer/reviewer who acted")
    decision: Decision
    reason: Optional[str] = Field(default=None, description="Optional free-text justification for the decision")
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class RiskDossier(BaseModel):
    """The final auditable output of the workflow."""
    merchant_name: str
    extracted_data: ExtractedMerchantData
    policy_checks: List[PolicyCheckResult]
    discrepancies: List[Discrepancy]
    evidence: List[EvidenceItem]
    fraud_risk_score: float = Field(..., ge=0.0, le=1.0)
    min_extraction_confidence: float = Field(..., ge=0.0, le=1.0)
    escalation_reasons: List[str] = Field(default_factory=list)
    requires_human_review: bool
    decision: Decision
    audit_trail: List[str]
    review_audit: List[ReviewAuditEntry] = Field(
        default_factory=list, description="Who resolved this case (if escalated), when, and why."
    )
    # Governance / provenance stamping - see src/policy.py POLICY_VERSION and
    # src/workflow.py MODEL_NAME. Every dossier records which policy ruleset
    # and which model version produced it, so a later threshold or model
    # change never leaves a historical decision unexplainable.
    policy_version: str = Field(default="unspecified")
    model_name: str = Field(default="unspecified")
    evaluated_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def escalation_report(self) -> str:
        """Human-readable answer to: which checks caused escalation, and what evidence supports each."""
        if not self.escalation_reasons:
            return "No escalation triggers were raised; the application was processed automatically."

        lines = [f"This application was escalated for {len(self.escalation_reasons)} reason(s):"]
        for i, reason in enumerate(self.escalation_reasons, 1):
            lines.append(f"\n{i}. {reason}")

        lines.append("\nSupporting evidence:")
        for check in self.policy_checks:
            if not check.passed or check.severity in ("high", "critical"):
                lines.append(f"\n- Check '{check.check_name}' ({check.severity}): {check.details}")
                for ev in check.evidence:
                    lines.append(f"    [{ev.source}] {ev.summary}")
        for d in self.discrepancies:
            lines.append(
                f"\n- Discrepancy in '{d.field}' ({d.severity}): expected '{d.expected}', "
                f"found '{d.found}'. {d.explanation}"
            )
        if self.min_extraction_confidence < 0.75:
            lines.append(
                f"\n- Low extraction confidence ({self.min_extraction_confidence:.2f}) on one or "
                "more fields required manual verification."
            )
        if self.review_audit:
            lines.append("\nReview history:")
            for entry in self.review_audit:
                reason_suffix = f" — {entry.reason}" if entry.reason else ""
                lines.append(f"\n- {entry.timestamp}: {entry.reviewer_id} set decision to '{entry.decision}'{reason_suffix}")
        return "\n".join(lines)
