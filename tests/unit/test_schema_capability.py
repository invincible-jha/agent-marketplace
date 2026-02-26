"""Unit tests for agent_marketplace.schema.capability."""
from __future__ import annotations

import math

import pytest
from pydantic import ValidationError

from agent_marketplace.schema.capability import (
    AgentCapability,
    CapabilityCategory,
    LatencyProfile,
    PricingModel,
    QualityMetrics,
)
from agent_marketplace.schema.provider import ProviderInfo
from tests.unit.conftest import make_capability


# ---------------------------------------------------------------------------
# CapabilityCategory enum
# ---------------------------------------------------------------------------


class TestCapabilityCategory:
    def test_all_expected_values_exist(self) -> None:
        expected = {
            "analysis", "generation", "transformation", "extraction",
            "interaction", "automation", "evaluation", "research",
            "reasoning", "specialized",
        }
        actual = {member.value for member in CapabilityCategory}
        assert actual == expected

    def test_construction_from_string_value(self) -> None:
        assert CapabilityCategory("analysis") is CapabilityCategory.ANALYSIS

    def test_invalid_value_raises_value_error(self) -> None:
        with pytest.raises(ValueError):
            CapabilityCategory("nonexistent-category")

    def test_category_is_a_str_subclass(self) -> None:
        assert isinstance(CapabilityCategory.GENERATION, str)


# ---------------------------------------------------------------------------
# PricingModel enum
# ---------------------------------------------------------------------------


class TestPricingModel:
    def test_all_expected_values_exist(self) -> None:
        expected = {"per_call", "per_token", "per_minute", "free", "custom"}
        actual = {member.value for member in PricingModel}
        assert actual == expected

    def test_free_is_the_default(self) -> None:
        cap = make_capability()
        assert cap.pricing_model is PricingModel.FREE


# ---------------------------------------------------------------------------
# QualityMetrics
# ---------------------------------------------------------------------------


class TestQualityMetrics:
    def test_empty_metrics_is_valid(self) -> None:
        qm = QualityMetrics()
        assert qm.metrics == {}
        assert qm.verified is False

    def test_finite_metrics_accepted(self) -> None:
        qm = QualityMetrics(metrics={"accuracy": 0.94, "f1": 0.88})
        assert qm.metrics["accuracy"] == pytest.approx(0.94)

    def test_infinite_metric_value_raises_validation_error(self) -> None:
        with pytest.raises(ValidationError, match="non-finite"):
            QualityMetrics(metrics={"score": float("inf")})

    def test_nan_metric_value_raises_validation_error(self) -> None:
        with pytest.raises(ValidationError, match="non-finite"):
            QualityMetrics(metrics={"score": float("nan")})

    def test_valid_iso_date_passes(self) -> None:
        qm = QualityMetrics(benchmark_date="2025-06-15")
        assert qm.benchmark_date == "2025-06-15"

    def test_invalid_date_format_raises_validation_error(self) -> None:
        with pytest.raises(ValidationError, match="ISO 8601"):
            QualityMetrics(benchmark_date="15/06/2025")

    def test_empty_date_is_allowed(self) -> None:
        qm = QualityMetrics(benchmark_date="")
        assert qm.benchmark_date == ""


# ---------------------------------------------------------------------------
# LatencyProfile
# ---------------------------------------------------------------------------


class TestLatencyProfile:
    def test_default_values_are_zero(self) -> None:
        lp = LatencyProfile()
        assert lp.p50_ms == 0.0
        assert lp.p95_ms == 0.0
        assert lp.p99_ms == 0.0

    def test_valid_ordered_percentiles(self) -> None:
        lp = LatencyProfile(p50_ms=100.0, p95_ms=200.0, p99_ms=350.0)
        assert lp.p99_ms == 350.0

    def test_negative_latency_raises_validation_error(self) -> None:
        with pytest.raises(ValidationError, match="non-negative"):
            LatencyProfile(p50_ms=-1.0)

    def test_p50_exceeding_p95_raises_validation_error(self) -> None:
        with pytest.raises(ValidationError, match="p50_ms must not exceed p95_ms"):
            LatencyProfile(p50_ms=300.0, p95_ms=100.0, p99_ms=400.0)

    def test_p95_exceeding_p99_raises_validation_error(self) -> None:
        with pytest.raises(ValidationError, match="p95_ms must not exceed p99_ms"):
            LatencyProfile(p50_ms=100.0, p95_ms=400.0, p99_ms=200.0)

    def test_zero_values_bypass_ordering_check(self) -> None:
        # Zero means "not set" — ordering is not enforced against zero
        lp = LatencyProfile(p50_ms=500.0, p95_ms=0.0, p99_ms=0.0)
        assert lp.p50_ms == 500.0


# ---------------------------------------------------------------------------
# AgentCapability — construction
# ---------------------------------------------------------------------------


class TestAgentCapabilityConstruction:
    def test_minimal_required_fields(self) -> None:
        cap = AgentCapability(
            name="tool",
            version="1.0",
            description="Does stuff.",
            category=CapabilityCategory.AUTOMATION,
            provider=ProviderInfo(name="Dev"),
        )
        assert cap.name == "tool"
        assert cap.version == "1.0"

    def test_capability_id_is_auto_generated(self) -> None:
        cap = make_capability()
        assert len(cap.capability_id) == 16
        assert cap.capability_id.isalnum() or all(
            c in "0123456789abcdef" for c in cap.capability_id
        )

    def test_capability_id_is_deterministic(self) -> None:
        cap_a = make_capability(name="tool-x", version="2.0.0", provider_name="ProvA")
        cap_b = make_capability(name="tool-x", version="2.0.0", provider_name="ProvA")
        assert cap_a.capability_id == cap_b.capability_id

    def test_capability_id_differs_for_different_versions(self) -> None:
        cap_v1 = make_capability(name="tool", version="1.0.0")
        cap_v2 = make_capability(name="tool", version="2.0.0")
        assert cap_v1.capability_id != cap_v2.capability_id

    def test_user_supplied_capability_id_is_preserved(self) -> None:
        cap = AgentCapability(
            capability_id="custom-id-abc",
            name="tool",
            version="1.0.0",
            description="desc",
            category=CapabilityCategory.AUTOMATION,
            provider=ProviderInfo(name="Dev"),
        )
        assert cap.capability_id == "custom-id-abc"

    def test_str_representation(self) -> None:
        cap = make_capability(name="my-tool", version="1.2.3", provider_name="Acme")
        assert str(cap) == "my-tool v1.2.3 (Acme)"

    def test_repr_contains_key_fields(self) -> None:
        cap = make_capability()
        representation = repr(cap)
        assert "capability_id=" in representation
        assert "name=" in representation
        assert "category=" in representation


# ---------------------------------------------------------------------------
# AgentCapability — name validator
# ---------------------------------------------------------------------------


class TestAgentCapabilityNameValidator:
    def test_whitespace_trimmed_from_name(self) -> None:
        cap = AgentCapability(
            name="  trimmed  ",
            version="1.0.0",
            description="desc",
            category=CapabilityCategory.ANALYSIS,
            provider=ProviderInfo(name="Dev"),
        )
        assert cap.name == "trimmed"

    def test_empty_name_raises_validation_error(self) -> None:
        with pytest.raises(ValidationError, match="name must not be empty"):
            AgentCapability(
                name="",
                version="1.0.0",
                description="desc",
                category=CapabilityCategory.ANALYSIS,
                provider=ProviderInfo(name="Dev"),
            )


# ---------------------------------------------------------------------------
# AgentCapability — version validator
# ---------------------------------------------------------------------------


class TestAgentCapabilityVersionValidator:
    @pytest.mark.parametrize("version", ["1", "1.0", "1.0.0", "1.0.0.0", "2.1.0-alpha"])
    def test_valid_version_formats(self, version: str) -> None:
        cap = make_capability(version=version)
        assert cap.version == version

    def test_empty_version_raises_validation_error(self) -> None:
        with pytest.raises(ValidationError, match="must not be empty"):
            make_capability(version="")

    def test_non_numeric_version_raises_validation_error(self) -> None:
        with pytest.raises(ValidationError, match="must start with digits"):
            make_capability(version="abc.def")

    def test_too_many_parts_raises_validation_error(self) -> None:
        with pytest.raises(ValidationError, match="1.*4 dot-separated parts"):
            make_capability(version="1.0.0.0.0")


# ---------------------------------------------------------------------------
# AgentCapability — cost validator
# ---------------------------------------------------------------------------


class TestAgentCapabilityCostValidator:
    def test_zero_cost_is_valid(self) -> None:
        cap = make_capability(cost=0.0)
        assert cap.cost == 0.0

    def test_positive_cost_is_valid(self) -> None:
        cap = make_capability(cost=0.01)
        assert cap.cost == pytest.approx(0.01)

    def test_negative_cost_raises_validation_error(self) -> None:
        with pytest.raises(ValidationError, match="non-negative"):
            make_capability(cost=-0.001)


# ---------------------------------------------------------------------------
# AgentCapability — trust_level field
# ---------------------------------------------------------------------------


class TestAgentCapabilityTrustLevel:
    def test_trust_level_defaults_to_zero(self) -> None:
        cap = make_capability()
        assert cap.trust_level == 0.0

    def test_trust_level_at_boundary_1_0(self) -> None:
        cap = make_capability(trust_level=1.0)
        assert cap.trust_level == 1.0

    def test_trust_level_above_1_raises_validation_error(self) -> None:
        with pytest.raises(ValidationError):
            make_capability(trust_level=1.1)

    def test_trust_level_below_0_raises_validation_error(self) -> None:
        with pytest.raises(ValidationError):
            make_capability(trust_level=-0.1)


# ---------------------------------------------------------------------------
# AgentCapability — validate() business rules
# ---------------------------------------------------------------------------


class TestAgentCapabilityValidateMethod:
    def test_fully_valid_capability_returns_empty_errors(self) -> None:
        cap = make_capability()
        errors = cap.validate()
        assert errors == []

    def test_empty_description_returns_error(self) -> None:
        cap = AgentCapability(
            name="tool",
            version="1.0.0",
            description="   ",
            category=CapabilityCategory.AUTOMATION,
            input_types=["application/json"],
            output_type="application/json",
            provider=ProviderInfo(name="Dev"),
        )
        errors = cap.validate()
        assert any("description" in e for e in errors)

    def test_paid_model_with_zero_cost_returns_error(self) -> None:
        cap = make_capability(pricing_model=PricingModel.PER_CALL, cost=0.0)
        errors = cap.validate()
        assert any("cost" in e or "pricing_model" in e for e in errors)

    def test_empty_input_types_returns_error(self) -> None:
        cap = AgentCapability(
            name="tool",
            version="1.0.0",
            description="A valid description here.",
            category=CapabilityCategory.AUTOMATION,
            input_types=[],
            output_type="application/json",
            provider=ProviderInfo(name="Dev"),
        )
        errors = cap.validate()
        assert any("input_types" in e for e in errors)

    def test_empty_output_type_returns_error(self) -> None:
        cap = AgentCapability(
            name="tool",
            version="1.0.0",
            description="A valid description here.",
            category=CapabilityCategory.AUTOMATION,
            input_types=["application/json"],
            output_type="",
            provider=ProviderInfo(name="Dev"),
        )
        errors = cap.validate()
        assert any("output_type" in e for e in errors)


# ---------------------------------------------------------------------------
# AgentCapability — serialization
# ---------------------------------------------------------------------------


class TestAgentCapabilitySerialization:
    def test_to_dict_returns_json_serializable_dict(self) -> None:
        cap = make_capability()
        data = cap.to_dict()
        assert isinstance(data, dict)
        assert data["name"] == cap.name

    def test_to_json_returns_valid_json_string(self) -> None:
        import json
        cap = make_capability()
        json_text = cap.to_json()
        parsed = json.loads(json_text)
        assert parsed["name"] == cap.name

    def test_from_json_roundtrip(self) -> None:
        cap = make_capability()
        json_text = cap.to_json()
        cap2 = AgentCapability.from_json(json_text)
        assert cap2.capability_id == cap.capability_id
        assert cap2.name == cap.name

    def test_to_yaml_returns_valid_yaml_string(self) -> None:
        import yaml
        cap = make_capability()
        yaml_text = cap.to_yaml()
        parsed = yaml.safe_load(yaml_text)
        assert parsed["name"] == cap.name

    def test_from_yaml_roundtrip(self) -> None:
        cap = make_capability()
        yaml_text = cap.to_yaml()
        cap2 = AgentCapability.from_yaml(yaml_text)
        assert cap2.capability_id == cap.capability_id

    def test_from_yaml_malformed_raises_error(self) -> None:
        with pytest.raises(Exception):
            AgentCapability.from_yaml("name: [unclosed")
