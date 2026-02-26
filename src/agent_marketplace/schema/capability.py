"""Core capability schema for agent-marketplace.

Defines the AgentCapability model along with all supporting enumerations
and nested value objects used to describe what an agent capability does,
how it is priced, and how it performs.
"""
from __future__ import annotations

import hashlib
import json
from datetime import date
from enum import Enum
from typing import ClassVar

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

from agent_marketplace.schema.provider import ProviderInfo


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class CapabilityCategory(str, Enum):
    """High-level functional category for an agent capability."""

    ANALYSIS = "analysis"
    GENERATION = "generation"
    TRANSFORMATION = "transformation"
    EXTRACTION = "extraction"
    INTERACTION = "interaction"
    AUTOMATION = "automation"
    EVALUATION = "evaluation"
    RESEARCH = "research"
    REASONING = "reasoning"
    SPECIALIZED = "specialized"


class PricingModel(str, Enum):
    """How the capability charges for its use."""

    PER_CALL = "per_call"
    PER_TOKEN = "per_token"
    PER_MINUTE = "per_minute"
    FREE = "free"
    CUSTOM = "custom"


# ---------------------------------------------------------------------------
# Value objects
# ---------------------------------------------------------------------------


class QualityMetrics(BaseModel):
    """Quantitative quality measurements for a capability.

    Attributes
    ----------
    metrics:
        Mapping of metric name to numeric score (e.g. {"accuracy": 0.94}).
    benchmark_source:
        Name or URL of the benchmark that produced these numbers.
    benchmark_date:
        ISO 8601 date string when benchmarks were last run.
    verified:
        Whether these metrics have been independently verified by the
        marketplace trust layer.
    """

    metrics: dict[str, float] = Field(default_factory=dict)
    benchmark_source: str = ""
    benchmark_date: str = ""
    verified: bool = False

    @field_validator("metrics")
    @classmethod
    def metrics_values_must_be_finite(cls, value: dict[str, float]) -> dict[str, float]:
        import math

        for key, score in value.items():
            if not math.isfinite(score):
                raise ValueError(f"Metric {key!r} has non-finite value {score!r}.")
        return value

    @field_validator("benchmark_date")
    @classmethod
    def validate_date_format(cls, value: str) -> str:
        if value:
            try:
                date.fromisoformat(value)
            except ValueError as exc:
                raise ValueError(
                    f"benchmark_date {value!r} must be an ISO 8601 date string (YYYY-MM-DD)."
                ) from exc
        return value


class LatencyProfile(BaseModel):
    """Latency percentile measurements in milliseconds.

    Attributes
    ----------
    p50_ms:
        Median (50th percentile) latency in milliseconds.
    p95_ms:
        95th percentile latency in milliseconds.
    p99_ms:
        99th percentile latency in milliseconds.
    """

    p50_ms: float = 0.0
    p95_ms: float = 0.0
    p99_ms: float = 0.0

    @field_validator("p50_ms", "p95_ms", "p99_ms")
    @classmethod
    def latency_must_be_non_negative(cls, value: float) -> float:
        if value < 0:
            raise ValueError(f"Latency value must be non-negative, got {value!r}.")
        return value

    @model_validator(mode="after")
    def percentiles_must_be_ordered(self) -> "LatencyProfile":
        if self.p50_ms > self.p95_ms and self.p95_ms != 0.0:
            raise ValueError("p50_ms must not exceed p95_ms.")
        if self.p95_ms > self.p99_ms and self.p99_ms != 0.0:
            raise ValueError("p95_ms must not exceed p99_ms.")
        return self


# ---------------------------------------------------------------------------
# Core capability model
# ---------------------------------------------------------------------------


class AgentCapability(BaseModel):
    """A fully described agent capability registered in the marketplace.

    The ``capability_id`` is auto-derived from the combination of
    ``name``, ``version``, and ``provider.name``.  It is a stable,
    deterministic identifier that uniquely addresses this capability.

    Attributes
    ----------
    capability_id:
        Auto-generated stable identifier (SHA-256 hex prefix of name+version+provider).
    name:
        Short, human-readable capability name (e.g. "pdf-extractor").
    version:
        Semantic version string (e.g. "1.2.0").
    description:
        Prose description of what the capability does.
    category:
        Primary functional category from ``CapabilityCategory``.
    tags:
        Free-form tags for keyword search.
    input_types:
        MIME types or schema names accepted by the capability.
    output_type:
        MIME type or schema name produced by the capability.
    quality_metrics:
        Benchmark and quality scores.
    pricing_model:
        How usage is billed.
    cost:
        Unit cost corresponding to the pricing model (USD).
    latency:
        Latency percentile profile.
    supported_languages:
        Natural language codes (ISO 639-1) the capability handles.
    supported_frameworks:
        Agent framework names this capability has adapters for.
    trust_level:
        Current computed trust score (0.0–1.0); updated by TrustScorer.
    provider:
        Identity information for the capability publisher.
    """

    # Sentinel so the validator can detect user-supplied vs auto-generated ids.
    _AUTO_ID_SENTINEL: ClassVar[str] = ""

    capability_id: str = ""
    name: str
    version: str
    description: str
    category: CapabilityCategory
    tags: list[str] = Field(default_factory=list)
    input_types: list[str] = Field(default_factory=list)
    output_type: str = ""
    quality_metrics: QualityMetrics = Field(default_factory=QualityMetrics)
    pricing_model: PricingModel = PricingModel.FREE
    cost: float = 0.0
    latency: LatencyProfile = Field(default_factory=LatencyProfile)
    supported_languages: list[str] = Field(default_factory=list)
    supported_frameworks: list[str] = Field(default_factory=list)
    trust_level: float = Field(default=0.0, ge=0.0, le=1.0)
    provider: ProviderInfo

    # ------------------------------------------------------------------
    # Validators
    # ------------------------------------------------------------------

    @field_validator("name")
    @classmethod
    def name_must_not_be_empty(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("Capability name must not be empty.")
        return stripped

    @field_validator("version")
    @classmethod
    def version_must_follow_semver_loosely(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("Version must not be empty.")
        parts = stripped.split(".")
        if not (1 <= len(parts) <= 4):
            raise ValueError(
                f"Version {stripped!r} must have 1–4 dot-separated parts."
            )
        for part in parts:
            # Allow pre-release suffixes like "1.0.0-alpha"
            numeric = part.split("-")[0]
            if not numeric.isdigit():
                raise ValueError(
                    f"Version part {part!r} must start with digits."
                )
        return stripped

    @field_validator("cost")
    @classmethod
    def cost_must_be_non_negative(cls, value: float) -> float:
        if value < 0:
            raise ValueError(f"Cost must be non-negative, got {value!r}.")
        return value

    @model_validator(mode="after")
    def auto_generate_capability_id(self) -> "AgentCapability":
        """Derive the capability_id from name + version + provider name."""
        if not self.capability_id:
            raw = f"{self.name}::{self.version}::{self.provider.name}"
            digest = hashlib.sha256(raw.encode()).hexdigest()[:16]
            # Use object.__setattr__ to bypass Pydantic's immutability guard
            # when model_config allows mutation.  We set via __dict__ for
            # models in default (mutable) mode.
            object.__setattr__(self, "capability_id", digest)
        return self

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self) -> list[str]:
        """Run additional business-rule validation beyond Pydantic field checks.

        Returns
        -------
        list[str]
            A list of human-readable error messages.  An empty list means
            the capability is valid.
        """
        errors: list[str] = []

        if not self.description.strip():
            errors.append("description must not be empty.")

        if self.pricing_model != PricingModel.FREE and self.cost == 0.0:
            errors.append(
                f"pricing_model is {self.pricing_model.value!r} but cost is 0.0; "
                "set a positive cost or use PricingModel.FREE."
            )

        if not self.input_types:
            errors.append("input_types must not be empty; specify at least one accepted type.")

        if not self.output_type.strip():
            errors.append("output_type must not be empty.")

        return errors

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable dictionary representation."""
        return self.model_dump(mode="json")

    def to_json(self, indent: int = 2) -> str:
        """Serialize to a JSON string.

        Parameters
        ----------
        indent:
            Number of spaces to indent the JSON output.
        """
        return self.model_dump_json(indent=indent)

    def to_yaml(self) -> str:
        """Serialize to a YAML string."""
        return yaml.dump(self.to_dict(), sort_keys=False, allow_unicode=True)

    @classmethod
    def from_yaml(cls, yaml_text: str) -> "AgentCapability":
        """Deserialize from a YAML string.

        Parameters
        ----------
        yaml_text:
            Raw YAML content representing a single AgentCapability.

        Returns
        -------
        AgentCapability
            A validated capability instance.

        Raises
        ------
        pydantic.ValidationError
            If the YAML content does not satisfy the schema.
        yaml.YAMLError
            If the YAML is malformed.
        """
        data = yaml.safe_load(yaml_text)
        return cls.model_validate(data)

    @classmethod
    def from_json(cls, json_text: str) -> "AgentCapability":
        """Deserialize from a JSON string.

        Parameters
        ----------
        json_text:
            Raw JSON content representing a single AgentCapability.
        """
        return cls.model_validate_json(json_text)

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __str__(self) -> str:
        return f"{self.name} v{self.version} ({self.provider.name})"

    def __repr__(self) -> str:
        return (
            f"AgentCapability(capability_id={self.capability_id!r}, "
            f"name={self.name!r}, version={self.version!r}, "
            f"category={self.category.value!r})"
        )
