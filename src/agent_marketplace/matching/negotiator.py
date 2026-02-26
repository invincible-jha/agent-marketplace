"""Price negotiation for agent-marketplace matching.

Selects the best-value provider offer that fits within a budget.
``PriceNegotiator`` applies a value-for-money ranking so callers can
choose the optimal trade-off between cost and quality rather than simply
picking the cheapest option.
"""
from __future__ import annotations

import math
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class PriceOffer:
    """A cost offer from a specific provider for a capability.

    Attributes
    ----------
    provider_id:
        Unique identifier of the offering provider.
    capability_id:
        Identifier of the capability being offered.
    cost_per_call:
        Quoted cost in USD per invocation.
    quality_score:
        Self-reported or observed quality score in [0.0, 1.0].
    trust_score:
        Current trust level of the provider in [0.0, 1.0].
    latency_p50_ms:
        Median latency in milliseconds.  Use 0.0 if unknown.
    """

    provider_id: str
    capability_id: str
    cost_per_call: float
    quality_score: float
    trust_score: float
    latency_p50_ms: float = 0.0

    def __post_init__(self) -> None:
        if self.cost_per_call < 0:
            raise ValueError(
                f"cost_per_call must be non-negative, got {self.cost_per_call!r}."
            )
        if not (0.0 <= self.quality_score <= 1.0):
            raise ValueError(
                f"quality_score must be in [0.0, 1.0], got {self.quality_score!r}."
            )
        if not (0.0 <= self.trust_score <= 1.0):
            raise ValueError(
                f"trust_score must be in [0.0, 1.0], got {self.trust_score!r}."
            )
        if self.latency_p50_ms < 0:
            raise ValueError(
                f"latency_p50_ms must be non-negative, got {self.latency_p50_ms!r}."
            )


@dataclass
class NegotiationResult:
    """The outcome of a price negotiation.

    Attributes
    ----------
    selected_offer:
        The winning offer, or ``None`` if no offer fits the budget.
    rejected_offers:
        Offers that were excluded (over budget or zero candidates).
    value_score:
        Composite value-for-money score for the winning offer [0.0, 1.0].
    """

    selected_offer: PriceOffer | None
    rejected_offers: list[PriceOffer]
    value_score: float


# ---------------------------------------------------------------------------
# Negotiator
# ---------------------------------------------------------------------------


class PriceNegotiator:
    """Selects the best-value offer within a budget.

    The value-for-money score is computed as::

        value = (quality_weight * quality_score
                 + trust_weight   * trust_score
                 - cost_weight    * normalised_cost)

    where ``normalised_cost = cost_per_call / max_budget`` (capped at 1.0).

    Parameters
    ----------
    quality_weight:
        Weight given to provider quality (default 0.45).
    trust_weight:
        Weight given to provider trust (default 0.35).
    cost_weight:
        Weight given to cost penalty (default 0.20).

    Raises
    ------
    ValueError
        If the three weights do not sum to 1.0 (within 1e-6 tolerance).
    """

    def __init__(
        self,
        quality_weight: float = 0.45,
        trust_weight: float = 0.35,
        cost_weight: float = 0.20,
    ) -> None:
        total = quality_weight + trust_weight + cost_weight
        if not math.isclose(total, 1.0, rel_tol=1e-6):
            raise ValueError(
                f"Weights must sum to 1.0, got {total:.6f}."
            )
        self._quality_weight = quality_weight
        self._trust_weight = trust_weight
        self._cost_weight = cost_weight

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def negotiate(
        self,
        offers: list[PriceOffer],
        max_budget: float,
    ) -> NegotiationResult:
        """Select the best offer within *max_budget*.

        Parameters
        ----------
        offers:
            Candidate price offers to evaluate.
        max_budget:
            Maximum cost per call the caller is willing to pay (USD).
            Must be non-negative.

        Returns
        -------
        NegotiationResult
            Contains the winning offer and metadata.  ``selected_offer``
            is ``None`` when no offer fits the budget.
        """
        if max_budget < 0:
            raise ValueError(
                f"max_budget must be non-negative, got {max_budget!r}."
            )

        within_budget = [o for o in offers if o.cost_per_call <= max_budget]
        rejected = [o for o in offers if o.cost_per_call > max_budget]

        if not within_budget:
            return NegotiationResult(
                selected_offer=None,
                rejected_offers=rejected,
                value_score=0.0,
            )

        # Use max_budget as normalisation denominator; fall back to 1.0 for free offers
        denominator = max_budget if max_budget > 0 else 1.0

        best_offer: PriceOffer | None = None
        best_value: float = -1.0

        for offer in within_budget:
            normalised_cost = min(offer.cost_per_call / denominator, 1.0)
            value = (
                self._quality_weight * offer.quality_score
                + self._trust_weight * offer.trust_score
                - self._cost_weight * normalised_cost
            )
            if value > best_value:
                best_value = value
                best_offer = offer

        return NegotiationResult(
            selected_offer=best_offer,
            rejected_offers=rejected,
            value_score=round(max(best_value, 0.0), 6),
        )

    def rank_offers(
        self, offers: list[PriceOffer], max_budget: float
    ) -> list[tuple[PriceOffer, float]]:
        """Rank all within-budget offers by value-for-money score.

        Parameters
        ----------
        offers:
            Candidate offers.
        max_budget:
            Budget ceiling.

        Returns
        -------
        list[tuple[PriceOffer, float]]
            ``(offer, value_score)`` pairs sorted descending by score.
            Over-budget offers are excluded.
        """
        within_budget = [o for o in offers if o.cost_per_call <= max_budget]
        denominator = max_budget if max_budget > 0 else 1.0

        scored: list[tuple[PriceOffer, float]] = []
        for offer in within_budget:
            normalised_cost = min(offer.cost_per_call / denominator, 1.0)
            value = (
                self._quality_weight * offer.quality_score
                + self._trust_weight * offer.trust_score
                - self._cost_weight * normalised_cost
            )
            scored.append((offer, round(value, 6)))

        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored
