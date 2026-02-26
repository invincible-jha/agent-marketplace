"""Capability matching engine for agent-marketplace.

Matches a ``CapabilityRequest`` against a set of candidate
``AgentCapability`` objects and returns ranked ``MatchResult`` records.

The matching logic applies hard eligibility filters first (trust floor,
cost ceiling, required certifications) and then scores each surviving
candidate with a composite fitness formula that rewards:

- capability name / tag overlap with the request's required_capabilities
- low latency relative to the caller's preference
- high trust level
- low cost relative to the budget
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from agent_marketplace.matching.request import CapabilityRequest
from agent_marketplace.schema.capability import AgentCapability


# ---------------------------------------------------------------------------
# Result model
# ---------------------------------------------------------------------------


@dataclass
class MatchResult:
    """A capability paired with its match score for a specific request.

    Attributes
    ----------
    capability:
        The matched ``AgentCapability``.
    match_score:
        Composite match quality score in [0.0, 1.0].  Higher is better.
    capability_overlap:
        Fraction of required capabilities matched [0.0, 1.0].
    latency_score:
        Latency fitness component [0.0, 1.0].
    trust_score:
        Trust component (direct from capability.trust_level).
    cost_score:
        Cost fitness component [0.0, 1.0].
    """

    capability: AgentCapability
    match_score: float
    capability_overlap: float
    latency_score: float
    trust_score: float
    cost_score: float


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class MatchingEngine:
    """Matches capability requests to provider candidates.

    Parameters
    ----------
    capability_weight:
        Weight for capability overlap signal (default 0.40).
    latency_weight:
        Weight for latency fitness signal (default 0.20).
    trust_weight:
        Weight for trust level signal (default 0.25).
    cost_weight:
        Weight for cost fitness signal (default 0.15).

    Raises
    ------
    ValueError
        If the four weights do not sum to 1.0.
    """

    def __init__(
        self,
        capability_weight: float = 0.40,
        latency_weight: float = 0.20,
        trust_weight: float = 0.25,
        cost_weight: float = 0.15,
    ) -> None:
        total = capability_weight + latency_weight + trust_weight + cost_weight
        if not math.isclose(total, 1.0, rel_tol=1e-6):
            raise ValueError(
                f"Weights must sum to 1.0, got {total:.6f}."
            )
        self._capability_weight = capability_weight
        self._latency_weight = latency_weight
        self._trust_weight = trust_weight
        self._cost_weight = cost_weight

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def match(
        self,
        request: CapabilityRequest,
        candidates: list[AgentCapability],
    ) -> list[MatchResult]:
        """Match *request* against *candidates* and return ranked results.

        Hard eligibility filters are applied first:

        - ``capability.trust_level >= request.min_trust``
        - ``capability.cost <= request.max_cost``
        - All ``required_certifications`` are satisfied (checked against tags
          and ``supported_frameworks`` as proxy when a dedicated field is absent).

        Surviving candidates are scored and returned in descending match
        order.

        Parameters
        ----------
        request:
            The caller's structured capability request.
        candidates:
            The full pool of registered capabilities to match against.

        Returns
        -------
        list[MatchResult]
            Eligible candidates ranked best-first by composite match score.
            Returns an empty list when no candidate passes eligibility.
        """
        eligible = self._filter_eligible(request, candidates)
        if not eligible:
            return []

        max_cost = max(
            (cap.cost for cap in eligible if cap.cost > 0),
            default=1.0,
        )

        results: list[MatchResult] = []
        for capability in eligible:
            overlap = self._capability_overlap(request, capability)
            latency = self._latency_score(request, capability)
            trust = capability.trust_level
            cost = self._cost_score(capability, max_cost)

            composite = (
                self._capability_weight * overlap
                + self._latency_weight * latency
                + self._trust_weight * trust
                + self._cost_weight * cost
            )

            results.append(
                MatchResult(
                    capability=capability,
                    match_score=round(composite, 6),
                    capability_overlap=round(overlap, 6),
                    latency_score=round(latency, 6),
                    trust_score=round(trust, 6),
                    cost_score=round(cost, 6),
                )
            )

        results.sort(key=lambda result: result.match_score, reverse=True)
        return results

    # ------------------------------------------------------------------
    # Eligibility filtering
    # ------------------------------------------------------------------

    @staticmethod
    def _filter_eligible(
        request: CapabilityRequest,
        candidates: list[AgentCapability],
    ) -> list[AgentCapability]:
        eligible: list[AgentCapability] = []
        for capability in candidates:
            if capability.trust_level < request.min_trust:
                continue
            if capability.cost > request.max_cost:
                continue
            if not MatchingEngine._certifications_satisfied(request, capability):
                continue
            eligible.append(capability)
        return eligible

    @staticmethod
    def _certifications_satisfied(
        request: CapabilityRequest,
        capability: AgentCapability,
    ) -> bool:
        if not request.required_certifications:
            return True
        # Check against tags and supported_frameworks as proxies for certifications
        provider_labels = {label.lower() for label in capability.tags}
        provider_labels.update(fw.lower() for fw in capability.supported_frameworks)
        for cert in request.required_certifications:
            if cert.lower() not in provider_labels:
                return False
        return True

    # ------------------------------------------------------------------
    # Scoring helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _capability_overlap(
        request: CapabilityRequest,
        capability: AgentCapability,
    ) -> float:
        """Fraction of required_capabilities matched by name, tags, or category."""
        if not request.required_capabilities:
            return 1.0

        searchable_tokens = set()
        searchable_tokens.add(capability.name.lower())
        searchable_tokens.add(capability.category.value.lower())
        searchable_tokens.update(tag.lower() for tag in capability.tags)

        matched = sum(
            1
            for required in request.required_capabilities
            if required.lower() in searchable_tokens
            or any(required.lower() in token for token in searchable_tokens)
        )
        return matched / len(request.required_capabilities)

    @staticmethod
    def _latency_score(
        request: CapabilityRequest,
        capability: AgentCapability,
    ) -> float:
        """Score latency fitness: 1.0 when at or below preference, decays above."""
        preferred = request.preferred_latency_ms
        actual = capability.latency.p50_ms

        if preferred <= 0 or actual <= 0:
            # No preference or no data — treat as neutral
            return 0.5

        if actual <= preferred:
            return 1.0

        # Exponential decay: score halves for each doubling of overage
        ratio = actual / preferred
        return max(0.0, math.exp(-math.log(2) * (ratio - 1)))

    @staticmethod
    def _cost_score(capability: AgentCapability, max_cost: float) -> float:
        """Invert-normalised cost: 1.0 for free, decreasing towards 0.0."""
        if capability.cost == 0.0:
            return 1.0
        if max_cost == 0.0:
            return 1.0
        return 1.0 - (capability.cost / max_cost)
