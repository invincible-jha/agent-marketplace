"""Usage tracking for agent-marketplace analytics.

``UsageTracker`` records every capability invocation and exposes methods
for computing popularity (all-time use count) and trend (recent activity
relative to baseline).  All data is kept in-memory; a persistent backend
can wrap this class and override ``record_usage`` / ``_get_records``.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class UsageRecord:
    """A single recorded capability usage event.

    Attributes
    ----------
    capability_id:
        Identifier of the capability that was invoked.
    provider_id:
        Identifier of the provider that served the request.
    success:
        Whether the invocation completed successfully.
    latency_ms:
        Observed latency in milliseconds (0 if not measured).
    cost_usd:
        Actual billed cost for this invocation in USD.
    recorded_at:
        UTC timestamp of the usage event.
    caller_id:
        Optional identifier of the calling agent or user.
    """

    capability_id: str
    provider_id: str
    success: bool = True
    latency_ms: float = 0.0
    cost_usd: float = 0.0
    recorded_at: datetime = field(
        default_factory=lambda: datetime.now(tz=timezone.utc)
    )
    caller_id: str = ""


# ---------------------------------------------------------------------------
# Tracker
# ---------------------------------------------------------------------------


class UsageTracker:
    """Records and analyses capability usage events.

    Usage
    -----
    ::

        tracker = UsageTracker()
        tracker.record_usage(capability_id="abc123", provider_id="acme")
        popular = tracker.get_popular(top_n=5)
        trending = tracker.get_trending(window_hours=24, top_n=5)

    Parameters
    ----------
    trending_window_hours:
        Default look-back window (hours) for trend calculations.
    """

    def __init__(self, trending_window_hours: int = 24) -> None:
        if trending_window_hours < 1:
            raise ValueError(
                f"trending_window_hours must be at least 1, "
                f"got {trending_window_hours!r}."
            )
        self._trending_window_hours = trending_window_hours
        self._records: list[UsageRecord] = []
        # Fast-path counters
        self._total_by_capability: dict[str, int] = defaultdict(int)

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record_usage(
        self,
        capability_id: str,
        provider_id: str = "",
        success: bool = True,
        latency_ms: float = 0.0,
        cost_usd: float = 0.0,
        caller_id: str = "",
        recorded_at: datetime | None = None,
    ) -> UsageRecord:
        """Record a single capability invocation.

        Parameters
        ----------
        capability_id:
            Identifier of the invoked capability.
        provider_id:
            Identifier of the serving provider.
        success:
            Whether the invocation succeeded.
        latency_ms:
            Observed latency in milliseconds.
        cost_usd:
            Billed cost in USD.
        caller_id:
            Optional caller identifier for segmented analytics.
        recorded_at:
            Optional explicit timestamp (defaults to now UTC).

        Returns
        -------
        UsageRecord
            The persisted usage record.
        """
        timestamp = recorded_at or datetime.now(tz=timezone.utc)
        record = UsageRecord(
            capability_id=capability_id,
            provider_id=provider_id,
            success=success,
            latency_ms=latency_ms,
            cost_usd=cost_usd,
            recorded_at=timestamp,
            caller_id=caller_id,
        )
        self._records.append(record)
        self._total_by_capability[capability_id] += 1
        return record

    # ------------------------------------------------------------------
    # Popularity (all-time)
    # ------------------------------------------------------------------

    def get_popular(self, top_n: int = 10) -> list[tuple[str, int]]:
        """Return the most-used capabilities by total invocation count.

        Parameters
        ----------
        top_n:
            Maximum number of entries to return.

        Returns
        -------
        list[tuple[str, int]]
            ``(capability_id, total_uses)`` pairs sorted descending.
        """
        ranked = sorted(
            self._total_by_capability.items(),
            key=lambda pair: pair[1],
            reverse=True,
        )
        return ranked[:top_n]

    # ------------------------------------------------------------------
    # Trending (recent window)
    # ------------------------------------------------------------------

    def get_trending(
        self,
        top_n: int = 10,
        window_hours: int | None = None,
    ) -> list[tuple[str, int]]:
        """Return capabilities with the most activity in a recent time window.

        Parameters
        ----------
        top_n:
            Maximum number of entries to return.
        window_hours:
            Look-back window in hours.  Defaults to ``trending_window_hours``
            passed at construction.

        Returns
        -------
        list[tuple[str, int]]
            ``(capability_id, recent_uses)`` pairs sorted descending.
        """
        hours = window_hours if window_hours is not None else self._trending_window_hours
        cutoff = datetime.now(tz=timezone.utc) - timedelta(hours=hours)

        recent_counts: dict[str, int] = defaultdict(int)
        for record in self._records:
            if record.recorded_at >= cutoff:
                recent_counts[record.capability_id] += 1

        ranked = sorted(recent_counts.items(), key=lambda pair: pair[1], reverse=True)
        return ranked[:top_n]

    # ------------------------------------------------------------------
    # Aggregation helpers
    # ------------------------------------------------------------------

    def total_invocations(self) -> int:
        """Return the total number of recorded invocations."""
        return len(self._records)

    def success_rate(self, capability_id: str | None = None) -> float:
        """Return the success rate for a capability or across all capabilities.

        Parameters
        ----------
        capability_id:
            When provided, computes the rate for that capability only.
            When ``None``, computes the global success rate.

        Returns
        -------
        float
            Success rate in [0.0, 1.0].  Returns 0.0 when no records match.
        """
        records = (
            [r for r in self._records if r.capability_id == capability_id]
            if capability_id is not None
            else self._records
        )
        if not records:
            return 0.0
        return sum(1 for r in records if r.success) / len(records)

    def average_latency_ms(self, capability_id: str | None = None) -> float:
        """Return mean latency in milliseconds for a capability or globally."""
        records = (
            [r for r in self._records if r.capability_id == capability_id]
            if capability_id is not None
            else self._records
        )
        measured = [r.latency_ms for r in records if r.latency_ms > 0]
        if not measured:
            return 0.0
        return sum(measured) / len(measured)

    def total_cost_usd(self, capability_id: str | None = None) -> float:
        """Return total billed cost in USD for a capability or globally."""
        records = (
            [r for r in self._records if r.capability_id == capability_id]
            if capability_id is not None
            else self._records
        )
        return round(sum(r.cost_usd for r in records), 6)

    def list_records(
        self,
        capability_id: str | None = None,
        limit: int = 100,
    ) -> list[UsageRecord]:
        """Return raw usage records, optionally filtered by capability.

        Parameters
        ----------
        capability_id:
            Optional filter.
        limit:
            Maximum number of records returned (most recent first).
        """
        records = (
            [r for r in self._records if r.capability_id == capability_id]
            if capability_id is not None
            else list(self._records)
        )
        # Most recent first
        records.sort(key=lambda r: r.recorded_at, reverse=True)
        return records[:limit]
