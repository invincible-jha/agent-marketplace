"""Capability verification — multi-rule trust and schema checks.

Verifies that an :class:`~agent_marketplace.schema.capability.AgentCapability`
meets the marketplace's trust, quality, and completeness requirements before
it is listed or activated.

Classes
-------
CapabilityVerifier
    Orchestrates a configurable rule pipeline.
VerificationResult
    Immutable outcome of a verification run.
VerificationRule
    ABC for individual rules.
CompletenessRule
    Checks required fields are populated.
QualityMetricsRule
    Checks quality metrics are present and within thresholds.
TrustLevelRule
    Checks the trust_level meets a minimum threshold.
"""
from __future__ import annotations

from agent_marketplace.verification.verifier import (
    CapabilityVerifier,
    CompletenessRule,
    QualityMetricsRule,
    TrustLevelRule,
    VerificationResult,
    VerificationRule,
)

__all__ = [
    "CapabilityVerifier",
    "CompletenessRule",
    "QualityMetricsRule",
    "TrustLevelRule",
    "VerificationResult",
    "VerificationRule",
]
