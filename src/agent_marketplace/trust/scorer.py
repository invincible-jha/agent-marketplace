"""Composite trust scorer for agent-marketplace providers.

Computes a single weighted trust score for a provider by combining four
independent signals:

- ``registration_age``  — how long the provider has been registered.
- ``usage_count``       — total number of times the provider's capabilities have been used.
- ``success_rate``      — fraction of usages that completed without errors.
- ``review_score``      — normalised average user review rating.

Each signal is normalised to [0.0, 1.0] before weighting.  The default
weights sum to 1.0 and can be overridden at construction time.
"""
from __future__ import annotations

import math
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Provider data container
# ---------------------------------------------------------------------------


@dataclass
class ProviderTrustData:
    """Input data required to score a provider's trustworthiness.

    Attributes
    ----------
    provider_id:
        Unique identifier of the provider being scored.
    registration_age_days:
        Number of days since the provider was first registered.
        Must be non-negative.
    usage_count:
        Total number of recorded usages across all capabilities.
        Must be non-negative.
    success_rate:
        Fraction of usages that succeeded (0.0–1.0).
    review_score:
        Average user review rating on a 1–5 scale.  Use 0.0 when
        no reviews exist (treated as neutral/unknown).
    """

    provider_id: str
    registration_age_days: float
    usage_count: int
    success_rate: float
    review_score: float

    def __post_init__(self) -> None:
        if self.registration_age_days < 0:
            raise ValueError(
                f"registration_age_days must be non-negative, "
                f"got {self.registration_age_days!r}."
            )
        if self.usage_count < 0:
            raise ValueError(
                f"usage_count must be non-negative, got {self.usage_count!r}."
            )
        if not (0.0 <= self.success_rate <= 1.0):
            raise ValueError(
                f"success_rate must be in [0.0, 1.0], got {self.success_rate!r}."
            )
        if not (0.0 <= self.review_score <= 5.0):
            raise ValueError(
                f"review_score must be in [0.0, 5.0], got {self.review_score!r}."
            )


# ---------------------------------------------------------------------------
# Scorer
# ---------------------------------------------------------------------------


class TrustScorer:
    """Computes a composite trust score for a capability provider.

    The composite score is::

        score = (
            age_weight     * age_signal
            + usage_weight * usage_signal
            + rate_weight  * success_rate
            + review_weight * review_signal
        )

    where each signal is independently normalised to [0.0, 1.0].

    Parameters
    ----------
    age_weight:
        Weight for registration age signal (default 0.20).
    usage_weight:
        Weight for usage count signal (default 0.30).
    success_rate_weight:
        Weight for call success-rate signal (default 0.30).
    review_weight:
        Weight for user review signal (default 0.20).
    age_saturation_days:
        Number of days after which the age signal saturates at 1.0
        (default 365 days — one year).
    usage_saturation:
        Number of usages after which the usage signal saturates at 1.0
        (default 1000).

    Raises
    ------
    ValueError
        If the four weights do not sum to 1.0 (within 1e-6 tolerance).
    """

    def __init__(
        self,
        age_weight: float = 0.20,
        usage_weight: float = 0.30,
        success_rate_weight: float = 0.30,
        review_weight: float = 0.20,
        age_saturation_days: float = 365.0,
        usage_saturation: int = 1000,
    ) -> None:
        total = age_weight + usage_weight + success_rate_weight + review_weight
        if not math.isclose(total, 1.0, rel_tol=1e-6):
            raise ValueError(
                f"Weights must sum to 1.0, got {total:.6f}. "
                "Adjust weights so they total exactly 1.0."
            )
        if age_saturation_days <= 0:
            raise ValueError(
                f"age_saturation_days must be positive, got {age_saturation_days!r}."
            )
        if usage_saturation < 1:
            raise ValueError(
                f"usage_saturation must be at least 1, got {usage_saturation!r}."
            )

        self._age_weight = age_weight
        self._usage_weight = usage_weight
        self._success_rate_weight = success_rate_weight
        self._review_weight = review_weight
        self._age_saturation_days = age_saturation_days
        self._usage_saturation = usage_saturation

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def score(self, provider: ProviderTrustData) -> float:
        """Compute and return a composite trust score for *provider*.

        Parameters
        ----------
        provider:
            A ``ProviderTrustData`` instance populated with current metrics.

        Returns
        -------
        float
            Composite trust score in [0.0, 1.0].  Higher is more trusted.
        """
        age_signal = self._normalise_age(provider.registration_age_days)
        usage_signal = self._normalise_usage(provider.usage_count)
        review_signal = self._normalise_review(provider.review_score)

        composite = (
            self._age_weight * age_signal
            + self._usage_weight * usage_signal
            + self._success_rate_weight * provider.success_rate
            + self._review_weight * review_signal
        )

        return round(min(max(composite, 0.0), 1.0), 6)

    # ------------------------------------------------------------------
    # Normalisation helpers
    # ------------------------------------------------------------------

    def _normalise_age(self, age_days: float) -> float:
        """Map registration age (days) to [0.0, 1.0] using logarithmic scaling.

        Uses ``log(1 + age) / log(1 + saturation)`` so that young providers
        grow quickly in the score and the signal asymptotically approaches 1.0.
        """
        if age_days <= 0.0:
            return 0.0
        return math.log1p(age_days) / math.log1p(self._age_saturation_days)

    def _normalise_usage(self, usage_count: int) -> float:
        """Map usage count to [0.0, 1.0] using logarithmic scaling."""
        if usage_count <= 0:
            return 0.0
        return math.log1p(usage_count) / math.log1p(self._usage_saturation)

    @staticmethod
    def _normalise_review(review_score: float) -> float:
        """Map a 0–5 review score to [0.0, 1.0].

        A score of 0.0 (no reviews) is treated as 0.5 (neutral) so that
        new providers without reviews are not unfairly penalised.
        """
        if review_score == 0.0:
            return 0.5  # Neutral — no data
        # Scale 1–5 linearly to 0.0–1.0
        return (review_score - 1.0) / 4.0
