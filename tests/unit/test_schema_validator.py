"""Unit tests for agent_marketplace.schema.validator."""
from __future__ import annotations

import pytest

from agent_marketplace.schema.capability import (
    AgentCapability,
    CapabilityCategory,
    PricingModel,
    QualityMetrics,
)
from agent_marketplace.schema.provider import ProviderInfo
from agent_marketplace.schema.validator import (
    SchemaValidator,
    ValidationError,
    ValidationResult,
)
from tests.unit.conftest import make_capability


# ---------------------------------------------------------------------------
# ValidationResult
# ---------------------------------------------------------------------------


class TestValidationResult:
    def test_valid_result_is_truthy(self) -> None:
        result = ValidationResult(valid=True)
        assert bool(result) is True

    def test_invalid_result_is_falsy(self) -> None:
        result = ValidationResult(valid=False, errors=["something wrong"])
        assert bool(result) is False

    def test_default_errors_and_warnings_are_empty(self) -> None:
        result = ValidationResult(valid=True)
        assert result.errors == []
        assert result.warnings == []


# ---------------------------------------------------------------------------
# ValidationError
# ---------------------------------------------------------------------------


class TestValidationError:
    def test_exception_message_joins_errors(self) -> None:
        exc = ValidationError(["err one", "err two"])
        assert "err one" in str(exc)
        assert "err two" in str(exc)

    def test_errors_attribute_is_preserved(self) -> None:
        errors = ["problem A", "problem B"]
        exc = ValidationError(errors)
        assert exc.errors == errors


# ---------------------------------------------------------------------------
# SchemaValidator.validate()
# ---------------------------------------------------------------------------


class TestSchemaValidatorValidate:
    def test_fully_valid_capability_passes(self) -> None:
        cap = make_capability(
            tags=["extraction", "pdf"],
            trust_level=0.8,
        )
        cap = cap.model_copy(
            update={
                "quality_metrics": QualityMetrics(
                    metrics={"accuracy": 0.9},
                    benchmark_source="internal",
                )
            }
        )
        validator = SchemaValidator()
        result = validator.validate(cap)
        assert result.valid is True

    def test_capability_without_tags_generates_warning(self) -> None:
        # make_capability(tags=[]) evaluates `[] or default_tags` as the default
        # list because [] is falsy.  Build the capability directly so that the
        # tags field is genuinely empty and the validator warning is triggered.
        cap = AgentCapability(
            name="no-tags-tool",
            version="1.0.0",
            description="Extracts structured data from PDF documents.",
            category=CapabilityCategory.EXTRACTION,
            tags=[],
            input_types=["application/pdf"],
            output_type="application/json",
            provider=ProviderInfo(name="Acme Corp", verified=True),
            trust_level=0.5,
        )
        validator = SchemaValidator()
        result = validator.validate(cap)
        assert any("tags" in w.lower() or "discoverability" in w.lower() for w in result.warnings)

    def test_tag_with_spaces_generates_warning(self) -> None:
        cap = make_capability(tags=["my tag"])
        validator = SchemaValidator()
        result = validator.validate(cap)
        assert any("spaces" in w.lower() or "hyphens" in w.lower() for w in result.warnings)

    def test_short_description_generates_warning(self) -> None:
        cap = AgentCapability(
            name="tool",
            version="1.0.0",
            description="Short.",
            category=CapabilityCategory.AUTOMATION,
            input_types=["application/json"],
            output_type="application/json",
            provider=ProviderInfo(name="Dev", verified=True),
            trust_level=0.5,
        )
        validator = SchemaValidator()
        result = validator.validate(cap)
        assert any("word" in w.lower() for w in result.warnings)

    def test_unverified_provider_with_zero_trust_generates_warning(self) -> None:
        cap = AgentCapability(
            name="tool",
            version="1.0.0",
            description="Does something interesting enough for testing.",
            category=CapabilityCategory.AUTOMATION,
            input_types=["application/json"],
            output_type="application/json",
            provider=ProviderInfo(name="Dev", verified=False),
            trust_level=0.0,
            tags=["automation"],
        )
        validator = SchemaValidator()
        result = validator.validate(cap)
        assert any("trust" in w.lower() for w in result.warnings)

    def test_missing_quality_metrics_generates_warning(self) -> None:
        cap = make_capability()
        validator = SchemaValidator()
        result = validator.validate(cap)
        assert any("quality" in w.lower() or "benchmark" in w.lower() for w in result.warnings)

    def test_quality_metrics_without_source_generates_warning(self) -> None:
        cap = make_capability()
        cap = cap.model_copy(
            update={
                "quality_metrics": QualityMetrics(
                    metrics={"accuracy": 0.9},
                    benchmark_source="",
                )
            }
        )
        validator = SchemaValidator()
        result = validator.validate(cap)
        assert any("benchmark_source" in w for w in result.warnings)

    def test_capability_with_business_rule_violation_is_invalid(self) -> None:
        cap = make_capability(pricing_model=PricingModel.PER_CALL, cost=0.0)
        validator = SchemaValidator()
        result = validator.validate(cap)
        assert result.valid is False
        assert len(result.errors) > 0


# ---------------------------------------------------------------------------
# SchemaValidator.validate_or_raise()
# ---------------------------------------------------------------------------


class TestSchemaValidatorValidateOrRaise:
    def test_valid_capability_does_not_raise(self) -> None:
        cap = make_capability()
        validator = SchemaValidator()
        validator.validate_or_raise(cap)  # should not raise

    def test_invalid_capability_raises_validation_error(self) -> None:
        cap = make_capability(pricing_model=PricingModel.PER_CALL, cost=0.0)
        validator = SchemaValidator()
        with pytest.raises(ValidationError):
            validator.validate_or_raise(cap)


# ---------------------------------------------------------------------------
# SchemaValidator.from_dict()
# ---------------------------------------------------------------------------


class TestSchemaValidatorFromDict:
    def test_valid_dict_returns_passing_result(self) -> None:
        data: dict[str, object] = {
            "name": "my-tool",
            "version": "1.0.0",
            "description": "A sufficiently long description for testing purposes.",
            "category": "analysis",
            "tags": ["analysis", "data"],
            "input_types": ["application/json"],
            "output_type": "application/json",
            "provider": {"name": "AnalyticsCo", "verified": True},
            "trust_level": 0.8,
        }
        result = SchemaValidator.from_dict(data)
        assert result.valid is True

    def test_dict_missing_required_field_returns_parse_errors(self) -> None:
        data: dict[str, object] = {
            # 'name' is intentionally missing
            "version": "1.0.0",
            "description": "desc",
            "category": "analysis",
            "provider": {"name": "Dev"},
        }
        result = SchemaValidator.from_dict(data)
        assert result.valid is False
        assert len(result.errors) > 0

    def test_dict_with_invalid_category_returns_parse_error(self) -> None:
        data: dict[str, object] = {
            "name": "tool",
            "version": "1.0.0",
            "description": "desc",
            "category": "not-a-real-category",
            "provider": {"name": "Dev"},
        }
        result = SchemaValidator.from_dict(data)
        assert result.valid is False
