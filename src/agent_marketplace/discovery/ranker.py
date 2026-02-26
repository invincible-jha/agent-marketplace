"""Fitness-based capability ranker for agent-marketplace.

Ranks a list of capabilities using a composite fitness score that
balances relevance, quality, cost-efficiency, and trust.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from agent_marketplace.schema.capability import AgentCapability


@dataclass
class RankedCapability:
    """A capability paired with its composite fitness score.

    Attributes
    ----------
    capability:
        The original AgentCapability object.
    fitness_score:
        Composite score in [0.0, 1.0] — higher is better.
    relevance_score:
        Keyword/semantic relevance component (0.0–1.0).
    quality_score:
        Quality metrics component (0.0–1.0).
    cost_efficiency_score:
        Inverse normalized cost component (0.0–1.0).
    trust_score:
        Trust level component (direct from capability.trust_level).
    """

    capability: AgentCapability
    fitness_score: float
    relevance_score: float
    quality_score: float
    cost_efficiency_score: float
    trust_score: float


class FitnessRanker:
    """Ranks capabilities by composite fitness.

    The fitness formula is::

        fitness = (
            relevance_weight   * relevance_score
            + quality_weight   * quality_score
            + cost_weight      * cost_efficiency_score
            + trust_weight     * trust_score
        )

    All weights must sum to 1.0.

    Parameters
    ----------
    relevance_weight:
        Weight given to keyword/semantic relevance (default 0.4).
    quality_weight:
        Weight given to quality metrics (default 0.2).
    cost_weight:
        Weight given to cost efficiency (default 0.2).
    trust_weight:
        Weight given to trust level (default 0.2).
    """

    def __init__(
        self,
        relevance_weight: float = 0.4,
        quality_weight: float = 0.2,
        cost_weight: float = 0.2,
        trust_weight: float = 0.2,
    ) -> None:
        total = relevance_weight + quality_weight + cost_weight + trust_weight
        if not math.isclose(total, 1.0, rel_tol=1e-6):
            raise ValueError(
                f"Ranker weights must sum to 1.0, got {total:.6f}. "
                "Adjust weights so they total exactly 1.0."
            )
        self._relevance_weight = relevance_weight
        self._quality_weight = quality_weight
        self._cost_weight = cost_weight
        self._trust_weight = trust_weight

    def rank(
        self,
        capabilities: list[AgentCapability],
        relevance_scores: dict[str, float] | None = None,
    ) -> list[RankedCapability]:
        """Rank *capabilities* and return them in descending fitness order.

        Parameters
        ----------
        capabilities:
            Candidate capabilities to rank.
        relevance_scores:
            Optional mapping of ``capability_id`` to relevance score (0.0–1.0).
            When omitted, all capabilities receive a relevance score of 1.0,
            which means ranking is driven purely by quality/cost/trust.

        Returns
        -------
        list[RankedCapability]
            Capabilities in descending fitness order.
        """
        if not capabilities:
            return []

        relevance_map = relevance_scores or {}
        max_cost = max((cap.cost for cap in capabilities if cap.cost > 0), default=1.0)

        ranked: list[RankedCapability] = []
        for cap in capabilities:
            relevance = relevance_map.get(cap.capability_id, 1.0)
            quality = self._compute_quality_score(cap)
            cost_eff = self._compute_cost_efficiency(cap, max_cost)
            trust = cap.trust_level

            fitness = (
                self._relevance_weight * relevance
                + self._quality_weight * quality
                + self._cost_weight * cost_eff
                + self._trust_weight * trust
            )

            ranked.append(
                RankedCapability(
                    capability=cap,
                    fitness_score=round(fitness, 6),
                    relevance_score=round(relevance, 6),
                    quality_score=round(quality, 6),
                    cost_efficiency_score=round(cost_eff, 6),
                    trust_score=round(trust, 6),
                )
            )

        ranked.sort(key=lambda r: r.fitness_score, reverse=True)
        return ranked

    # ------------------------------------------------------------------
    # Private score computations
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_quality_score(capability: AgentCapability) -> float:
        """Aggregate quality metrics into a 0.0–1.0 score."""
        metrics = capability.quality_metrics.metrics
        if not metrics:
            return 0.0
        values = list(metrics.values())
        # Clamp each metric to [0, 1] before averaging
        clamped = [max(0.0, min(1.0, v)) for v in values]
        return sum(clamped) / len(clamped)

    @staticmethod
    def _compute_cost_efficiency(
        capability: AgentCapability, max_cost: float
    ) -> float:
        """Return 1.0 for free, decreasing towards 0.0 as cost approaches max_cost."""
        if capability.cost == 0.0:
            return 1.0
        if max_cost == 0.0:
            return 1.0
        # Invert: lower cost = higher efficiency
        return 1.0 - (capability.cost / max_cost)
