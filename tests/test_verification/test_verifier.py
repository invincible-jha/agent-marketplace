"""Tests for agent_marketplace.verification.verifier."""
from __future__ import annotations

import pytest

from agent_marketplace.schema.capability import (
    AgentCapability,
    CapabilityCategory,
    QualityMetrics,
)
from agent_marketplace.schema.provider import ProviderInfo
from agent_marketplace.verification.verifier import (
    CapabilityVerifier,
    CompletenessRule,
    QualityMetricsRule,
    SupportedFrameworksRule,
    TrustLevelRule,
    VerificationResult,
    VerificationRule,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _cap(
    description: str = "A good description",
    input_types: list[str] | None = None,
    output_type: str = "application/json",
    trust_level: float = 0.8,
    tags: list[str] | None = None,
    supported_frameworks: list[str] | None = None,
) -> AgentCapability:
    # Use sentinel so callers can pass an explicit empty list without
    # it being overridden by the `or default` fallback.
    resolved_input_types: list[str] = ["text/plain"] if input_types is None else input_types
    resolved_tags: list[str] = ["test"] if tags is None else tags
    return AgentCapability(
        name="test-cap",
        version="1.0",
        description=description,
        category=CapabilityCategory.ANALYSIS,
        input_types=resolved_input_types,
        output_type=output_type,
        trust_level=trust_level,
        tags=resolved_tags,
        supported_frameworks=supported_frameworks or ["langchain"],
        provider=ProviderInfo(name="Test Provider"),
    )


# ===========================================================================
# VerificationResult
# ===========================================================================


class TestVerificationResult:
    def _make_result(self, passed: bool = True) -> VerificationResult:
        from datetime import datetime, timezone
        return VerificationResult(
            capability_id="cap-001",
            passed=passed,
            failures=() if passed else ("rule X failed",),
            warnings=(),
            rules_run=("completeness", "trust_level"),
            verified_at=datetime.now(timezone.utc),
        )

    def test_frozen(self) -> None:
        result = self._make_result()
        with pytest.raises(Exception):
            result.passed = False  # type: ignore[misc]

    def test_failure_count(self) -> None:
        result = self._make_result(passed=False)
        assert result.failure_count == 1

    def test_summary_passed(self) -> None:
        result = self._make_result(passed=True)
        assert "PASSED" in result.summary()

    def test_summary_failed(self) -> None:
        result = self._make_result(passed=False)
        assert "FAILED" in result.summary()

    def test_to_dict_keys(self) -> None:
        result = self._make_result()
        d = result.to_dict()
        for key in ("capability_id", "passed", "failures", "warnings",
                    "rules_run", "verified_at", "failure_count"):
            assert key in d


# ===========================================================================
# CompletenessRule
# ===========================================================================


class TestCompletenessRule:
    def test_is_verification_rule(self) -> None:
        assert isinstance(CompletenessRule(), VerificationRule)

    def test_name(self) -> None:
        assert CompletenessRule().name == "completeness"

    def test_passes_complete_capability(self) -> None:
        rule = CompletenessRule()
        passed, msg = rule.check(_cap())
        assert passed

    def test_fails_empty_description(self) -> None:
        rule = CompletenessRule()
        cap = _cap(description="   ")
        passed, msg = rule.check(cap)
        assert not passed
        assert "description" in msg

    def test_fails_empty_input_types(self) -> None:
        rule = CompletenessRule()
        cap = _cap(input_types=[])
        passed, msg = rule.check(cap)
        assert not passed
        assert "input_types" in msg

    def test_fails_empty_output_type(self) -> None:
        rule = CompletenessRule()
        cap = _cap(output_type="")
        passed, msg = rule.check(cap)
        assert not passed
        assert "output_type" in msg

    def test_require_tags_fails_when_no_tags(self) -> None:
        rule = CompletenessRule(require_tags=True)
        cap = _cap(tags=[])
        passed, msg = rule.check(cap)
        assert not passed
        assert "tags" in msg

    def test_require_tags_passes_with_tags(self) -> None:
        rule = CompletenessRule(require_tags=True)
        cap = _cap(tags=["useful"])
        passed, msg = rule.check(cap)
        assert passed

    def test_not_warning_only(self) -> None:
        assert not CompletenessRule().is_warning_only


# ===========================================================================
# TrustLevelRule
# ===========================================================================


class TestTrustLevelRule:
    def test_name(self) -> None:
        assert TrustLevelRule().name == "trust_level"

    def test_passes_above_threshold(self) -> None:
        rule = TrustLevelRule(min_trust=0.5)
        cap = _cap(trust_level=0.8)
        passed, msg = rule.check(cap)
        assert passed

    def test_fails_below_threshold(self) -> None:
        rule = TrustLevelRule(min_trust=0.7)
        cap = _cap(trust_level=0.3)
        passed, msg = rule.check(cap)
        assert not passed
        assert "0.300" in msg or "0.3" in msg

    def test_passes_at_exact_threshold(self) -> None:
        rule = TrustLevelRule(min_trust=0.5)
        cap = _cap(trust_level=0.5)
        passed, _ = rule.check(cap)
        assert passed

    def test_invalid_min_trust_raises(self) -> None:
        with pytest.raises(ValueError, match="min_trust"):
            TrustLevelRule(min_trust=1.5)

    def test_warning_only_mode(self) -> None:
        rule = TrustLevelRule(min_trust=0.9, warning_only=True)
        assert rule.is_warning_only is True


# ===========================================================================
# QualityMetricsRule
# ===========================================================================


class TestQualityMetricsRule:
    def _cap_with_metrics(self, verified: bool = True) -> AgentCapability:
        cap = _cap()
        return cap.model_copy(
            update={
                "quality_metrics": QualityMetrics(
                    metrics={"accuracy": 0.92, "f1": 0.88},
                    benchmark_source="test-bench",
                    benchmark_date="2025-01-01",
                    verified=verified,
                )
            }
        )

    def test_name(self) -> None:
        assert QualityMetricsRule().name == "quality_metrics"

    def test_passes_no_requirements(self) -> None:
        rule = QualityMetricsRule()
        passed, msg = rule.check(_cap())
        assert passed

    def test_fails_when_verified_required_but_false(self) -> None:
        rule = QualityMetricsRule(require_verified=True)
        cap = self._cap_with_metrics(verified=False)
        passed, msg = rule.check(cap)
        assert not passed
        assert "verified" in msg

    def test_passes_when_verified_and_required(self) -> None:
        rule = QualityMetricsRule(require_verified=True)
        cap = self._cap_with_metrics(verified=True)
        passed, msg = rule.check(cap)
        assert passed

    def test_fails_metric_below_minimum(self) -> None:
        rule = QualityMetricsRule(min_metric_value=0.95)
        cap = self._cap_with_metrics()
        passed, msg = rule.check(cap)
        assert not passed
        assert "accuracy" in msg or "f1" in msg

    def test_require_specific_metric_missing(self) -> None:
        rule = QualityMetricsRule(require_metrics=["precision"])
        cap = self._cap_with_metrics()
        passed, msg = rule.check(cap)
        assert not passed
        assert "precision" in msg

    def test_require_specific_metric_present(self) -> None:
        rule = QualityMetricsRule(require_metrics=["accuracy"])
        cap = self._cap_with_metrics()
        passed, msg = rule.check(cap)
        assert passed

    def test_warning_only_mode(self) -> None:
        rule = QualityMetricsRule(require_verified=True, warning_only=True)
        assert rule.is_warning_only is True


# ===========================================================================
# SupportedFrameworksRule
# ===========================================================================


class TestSupportedFrameworksRule:
    def test_name(self) -> None:
        rule = SupportedFrameworksRule(["langchain"])
        assert rule.name == "supported_frameworks"

    def test_passes_when_framework_supported(self) -> None:
        rule = SupportedFrameworksRule(["langchain"])
        cap = _cap(supported_frameworks=["langchain", "crewai"])
        passed, msg = rule.check(cap)
        assert passed

    def test_fails_when_no_required_framework(self) -> None:
        rule = SupportedFrameworksRule(["crewai"])
        cap = _cap(supported_frameworks=["langchain"])
        passed, msg = rule.check(cap)
        assert not passed
        assert "crewai" in msg

    def test_empty_required_raises(self) -> None:
        with pytest.raises(ValueError, match="required_frameworks"):
            SupportedFrameworksRule([])


# ===========================================================================
# CapabilityVerifier
# ===========================================================================


class TestCapabilityVerifier:
    def test_default_rules(self) -> None:
        verifier = CapabilityVerifier()
        assert verifier.rule_count >= 1

    def test_custom_rules(self) -> None:
        verifier = CapabilityVerifier([CompletenessRule(), TrustLevelRule(min_trust=0.5)])
        assert verifier.rule_count == 2

    def test_verify_passes_good_capability(self) -> None:
        verifier = CapabilityVerifier([CompletenessRule(), TrustLevelRule(min_trust=0.5)])
        result = verifier.verify(_cap(trust_level=0.8))
        assert result.passed

    def test_verify_fails_incomplete(self) -> None:
        verifier = CapabilityVerifier([CompletenessRule()])
        cap = _cap(description="")
        result = verifier.verify(cap)
        assert not result.passed
        assert result.failure_count == 1

    def test_all_rules_run_by_default(self) -> None:
        verifier = CapabilityVerifier([
            CompletenessRule(),
            TrustLevelRule(min_trust=0.5),
            QualityMetricsRule(require_verified=True),
        ])
        cap = _cap(description="", trust_level=0.2)
        result = verifier.verify(cap)
        assert len(result.rules_run) == 3

    def test_stop_on_first_failure(self) -> None:
        verifier = CapabilityVerifier(
            [CompletenessRule(), TrustLevelRule(min_trust=0.9)],
            stop_on_first_failure=True,
        )
        cap = _cap(description="", trust_level=0.2)
        result = verifier.verify(cap)
        assert result.failure_count == 1
        assert len(result.rules_run) == 1

    def test_warning_only_rule_does_not_block(self) -> None:
        verifier = CapabilityVerifier([
            CompletenessRule(),
            TrustLevelRule(min_trust=0.9, warning_only=True),  # won't block
        ])
        cap = _cap(trust_level=0.1)
        result = verifier.verify(cap)
        assert result.passed  # completeness passes, trust is warning
        assert result.warning_count == 1

    def test_verify_batch(self) -> None:
        verifier = CapabilityVerifier([CompletenessRule()])
        caps = [_cap(), _cap(description="")]
        results = verifier.verify_batch(caps)
        assert len(results) == 2
        assert results[0].passed
        assert not results[1].passed

    def test_add_rule(self) -> None:
        verifier = CapabilityVerifier([])
        verifier.add_rule(CompletenessRule())
        assert verifier.rule_count == 1

    def test_rule_names(self) -> None:
        verifier = CapabilityVerifier([CompletenessRule(), TrustLevelRule()])
        names = verifier.rule_names
        assert "completeness" in names
        assert "trust_level" in names

    def test_result_has_capability_id(self) -> None:
        verifier = CapabilityVerifier([CompletenessRule()])
        cap = _cap()
        result = verifier.verify(cap)
        assert result.capability_id == cap.capability_id

    def test_result_verified_at_is_utc(self) -> None:
        from datetime import timezone
        verifier = CapabilityVerifier([CompletenessRule()])
        result = verifier.verify(_cap())
        assert result.verified_at.tzinfo is not None
