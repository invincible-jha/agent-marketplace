"""Schema validation for agent-marketplace.

Provides both JSON-Schema structural validation and custom business-rule
validation for AgentCapability objects.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from pydantic import ValidationError as PydanticValidationError

from agent_marketplace.schema.capability import AgentCapability


# ---------------------------------------------------------------------------
# Public value objects
# ---------------------------------------------------------------------------


class ValidationError(Exception):
    """Raised when a capability fails schema or business-rule validation."""

    def __init__(self, errors: list[str]) -> None:
        self.errors = errors
        super().__init__("; ".join(errors))


@dataclass
class ValidationResult:
    """The outcome of a validation run.

    Attributes
    ----------
    valid:
        True if no errors were found.
    errors:
        List of human-readable error messages.
    warnings:
        List of non-fatal advisory messages.
    """

    valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def __bool__(self) -> bool:
        return self.valid


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------


class SchemaValidator:
    """Validates AgentCapability objects against structural and business rules.

    Usage
    -----
    ::

        validator = SchemaValidator()
        result = validator.validate(capability)
        if not result:
            for error in result.errors:
                print(error)
    """

    # Minimum description word count to produce a warning (not an error).
    _MIN_DESCRIPTION_WORDS: int = 5

    def validate(self, capability: AgentCapability) -> ValidationResult:
        """Run all validation rules against *capability*.

        Parameters
        ----------
        capability:
            The capability to validate.

        Returns
        -------
        ValidationResult
            Contains all errors and warnings discovered.
        """
        errors: list[str] = []
        warnings: list[str] = []

        # --- Business rules from the capability itself ---
        business_errors = capability.validate()
        errors.extend(business_errors)

        # --- Additional structural checks ---
        self._check_tags(capability, warnings)
        self._check_description_quality(capability, warnings)
        self._check_trust_level(capability, warnings)
        self._check_quality_metrics(capability, warnings)

        return ValidationResult(valid=len(errors) == 0, errors=errors, warnings=warnings)

    def validate_or_raise(self, capability: AgentCapability) -> None:
        """Validate and raise ``ValidationError`` if any errors are found.

        Parameters
        ----------
        capability:
            The capability to validate.

        Raises
        ------
        ValidationError
            If the capability has one or more validation errors.
        """
        result = self.validate(capability)
        if not result.valid:
            raise ValidationError(result.errors)

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> ValidationResult:
        """Validate a raw dictionary by first parsing it into AgentCapability.

        Parameters
        ----------
        data:
            Raw mapping (e.g. parsed from JSON or YAML).

        Returns
        -------
        ValidationResult
            Includes Pydantic parse errors if the structure is invalid.
        """
        try:
            capability = AgentCapability.model_validate(data)
        except PydanticValidationError as exc:
            errors = [f"{err['loc']}: {err['msg']}" for err in exc.errors()]
            return ValidationResult(valid=False, errors=errors)

        validator = cls()
        return validator.validate(capability)

    # ------------------------------------------------------------------
    # Private rule helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _check_tags(capability: AgentCapability, warnings: list[str]) -> None:
        if not capability.tags:
            warnings.append(
                "No tags provided; adding tags improves discoverability in search."
            )
        for tag in capability.tags:
            if " " in tag:
                warnings.append(
                    f"Tag {tag!r} contains spaces; consider using hyphens for consistency."
                )

    @staticmethod
    def _check_description_quality(
        capability: AgentCapability, warnings: list[str]
    ) -> None:
        word_count = len(capability.description.split())
        if word_count < SchemaValidator._MIN_DESCRIPTION_WORDS:
            warnings.append(
                f"Description has only {word_count} word(s); "
                f"at least {SchemaValidator._MIN_DESCRIPTION_WORDS} words are recommended."
            )

    @staticmethod
    def _check_trust_level(capability: AgentCapability, warnings: list[str]) -> None:
        if capability.trust_level == 0.0 and not capability.provider.verified:
            warnings.append(
                "trust_level is 0.0 and provider is not verified; "
                "run TrustScorer.compute() to initialize trust."
            )

    @staticmethod
    def _check_quality_metrics(
        capability: AgentCapability, warnings: list[str]
    ) -> None:
        if not capability.quality_metrics.metrics:
            warnings.append(
                "No quality metrics provided; benchmarks improve trust scoring."
            )
        elif not capability.quality_metrics.benchmark_source:
            warnings.append(
                "Quality metrics present but benchmark_source is empty; "
                "document where the numbers come from."
            )
