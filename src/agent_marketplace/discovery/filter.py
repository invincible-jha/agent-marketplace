"""Constraint-based capability filter for agent-marketplace.

Filters an iterable of AgentCapability objects against threshold
constraints for quality, cost, latency, and trust.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from agent_marketplace.schema.capability import AgentCapability, CapabilityCategory


@dataclass
class FilterConstraints:
    """Threshold-based constraints for capability filtering.

    All constraints are optional; unset constraints do not filter.

    Attributes
    ----------
    min_trust:
        Minimum acceptable trust_level (0.0–1.0).
    max_cost:
        Maximum acceptable cost (USD per unit).
    max_p95_latency_ms:
        Maximum acceptable 95th-percentile latency in milliseconds.
    min_quality_score:
        Minimum acceptable value for a named quality metric.
    required_quality_metric:
        Name of the quality metric that ``min_quality_score`` applies to.
    category:
        Restrict results to a single CapabilityCategory.
    required_tags:
        Tags that must all be present (AND semantics).
    supported_language:
        Language code that must appear in supported_languages.
    supported_framework:
        Framework name that must appear in supported_frameworks.
    pricing_models:
        Allowed pricing models (OR semantics).  Empty = allow all.
    """

    min_trust: float = 0.0
    max_cost: float = float("inf")
    max_p95_latency_ms: float = float("inf")
    min_quality_score: float = 0.0
    required_quality_metric: str = ""
    category: Optional[CapabilityCategory] = None
    required_tags: list[str] = field(default_factory=list)
    supported_language: str = ""
    supported_framework: str = ""
    pricing_models: list[str] = field(default_factory=list)


class ConstraintFilter:
    """Filters a list of capabilities against a set of hard constraints.

    Usage
    -----
    ::

        constraints = FilterConstraints(min_trust=0.7, max_cost=0.01)
        filter_ = ConstraintFilter(constraints)
        passing = filter_.apply(all_capabilities)
    """

    def __init__(self, constraints: FilterConstraints) -> None:
        self._constraints = constraints

    @property
    def constraints(self) -> FilterConstraints:
        """Return the active constraint set."""
        return self._constraints

    def apply(self, capabilities: list[AgentCapability]) -> list[AgentCapability]:
        """Return the subset of *capabilities* that pass all constraints.

        Parameters
        ----------
        capabilities:
            The full candidate list to filter.

        Returns
        -------
        list[AgentCapability]
            Capabilities that satisfy every active constraint.
        """
        return [cap for cap in capabilities if self._passes(cap)]

    def passes(self, capability: AgentCapability) -> bool:
        """Return True if *capability* satisfies all constraints."""
        return self._passes(capability)

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _passes(self, capability: AgentCapability) -> bool:
        constraints = self._constraints

        if capability.trust_level < constraints.min_trust:
            return False

        if capability.cost > constraints.max_cost:
            return False

        if capability.latency.p95_ms > constraints.max_p95_latency_ms:
            return False

        if constraints.required_quality_metric:
            metric_value = capability.quality_metrics.metrics.get(
                constraints.required_quality_metric, 0.0
            )
            if metric_value < constraints.min_quality_score:
                return False

        if constraints.category is not None and capability.category != constraints.category:
            return False

        if constraints.required_tags:
            capability_tags = {t.lower() for t in capability.tags}
            for tag in constraints.required_tags:
                if tag.lower() not in capability_tags:
                    return False

        if constraints.supported_language:
            lang = constraints.supported_language.lower()
            if lang not in [lang_.lower() for lang_ in capability.supported_languages]:
                return False

        if constraints.supported_framework:
            fw = constraints.supported_framework.lower()
            if fw not in [fw_.lower() for fw_ in capability.supported_frameworks]:
                return False

        if constraints.pricing_models:
            allowed = {pm.lower() for pm in constraints.pricing_models}
            if capability.pricing_model.value.lower() not in allowed:
                return False

        return True
