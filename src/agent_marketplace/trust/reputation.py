"""Sliding-window reputation tracker for agent-marketplace.

Tracks per-provider outcome histories and computes a rolling success-rate
reputation score over a configurable window of recent outcomes.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone


# ---------------------------------------------------------------------------
# Internal record
# ---------------------------------------------------------------------------


@dataclass
class _OutcomeRecord:
    """A single recorded outcome for a provider."""

    success: bool
    recorded_at: datetime = field(
        default_factory=lambda: datetime.now(tz=timezone.utc)
    )


# ---------------------------------------------------------------------------
# Tracker
# ---------------------------------------------------------------------------


class ReputationTracker:
    """Per-provider sliding-window reputation tracker.

    Maintains a bounded deque of recent outcome records for each provider.
    The reputation score is the fraction of successful outcomes within the
    window: ``successes / window_size`` (or total outcomes if fewer than
    ``window_size`` records have been recorded).

    Parameters
    ----------
    window_size:
        Maximum number of recent outcomes to retain per provider.
        Must be a positive integer.

    Usage
    -----
    ::

        tracker = ReputationTracker(window_size=100)
        tracker.record_outcome("acme-provider", success=True)
        tracker.record_outcome("acme-provider", success=False)
        score = tracker.get_reputation("acme-provider")  # 0.5
    """

    def __init__(self, window_size: int = 100) -> None:
        if window_size < 1:
            raise ValueError(
                f"window_size must be at least 1, got {window_size!r}."
            )
        self._window_size = window_size
        self._outcomes: dict[str, deque[_OutcomeRecord]] = {}

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record_outcome(self, provider_id: str, success: bool) -> None:
        """Record a single call outcome for a provider.

        Parameters
        ----------
        provider_id:
            Unique identifier of the provider.
        success:
            True if the call succeeded; False if it failed.
        """
        if provider_id not in self._outcomes:
            self._outcomes[provider_id] = deque(maxlen=self._window_size)
        self._outcomes[provider_id].append(_OutcomeRecord(success=success))

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def get_reputation(self, provider_id: str) -> float:
        """Return the sliding-window success rate for a provider.

        Parameters
        ----------
        provider_id:
            Unique identifier of the provider.

        Returns
        -------
        float
            Success rate in [0.0, 1.0].  Returns 0.0 if no outcomes have
            been recorded for this provider.
        """
        records = self._outcomes.get(provider_id)
        if not records:
            return 0.0
        successes = sum(1 for record in records if record.success)
        return successes / len(records)

    # ------------------------------------------------------------------
    # Inspection
    # ------------------------------------------------------------------

    def total_recorded(self, provider_id: str) -> int:
        """Return the number of outcomes recorded within the window for a provider."""
        records = self._outcomes.get(provider_id)
        return len(records) if records else 0

    def known_providers(self) -> list[str]:
        """Return a sorted list of provider IDs with at least one recorded outcome."""
        return sorted(self._outcomes.keys())

    def reset(self, provider_id: str) -> None:
        """Clear all outcome records for a provider.

        Parameters
        ----------
        provider_id:
            The provider whose history to clear.
        """
        self._outcomes.pop(provider_id, None)

    @property
    def window_size(self) -> int:
        """Maximum number of outcomes retained per provider."""
        return self._window_size
