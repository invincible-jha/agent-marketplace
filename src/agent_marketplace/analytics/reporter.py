"""Marketplace analytics reporter for agent-marketplace.

``MarketplaceReporter`` aggregates data from the registry store and usage
tracker to produce human-readable summary reports.
"""
from __future__ import annotations

from datetime import datetime, timezone

from agent_marketplace import __version__
from agent_marketplace.analytics.usage import UsageTracker
from agent_marketplace.registry.store import RegistryStore


class MarketplaceReporter:
    """Generates summary analytics reports for the marketplace.

    Combines registry metadata and usage telemetry into structured
    report dicts that can be serialised to JSON, printed to the terminal,
    or forwarded to a monitoring system.

    Parameters
    ----------
    store:
        The registry backend to query for capability metadata.
    usage_tracker:
        The usage tracker containing invocation history.

    Usage
    -----
    ::

        reporter = MarketplaceReporter(store=store, usage_tracker=tracker)
        report = reporter.summary_report()
    """

    def __init__(
        self,
        store: RegistryStore,
        usage_tracker: UsageTracker,
    ) -> None:
        self._store = store
        self._usage_tracker = usage_tracker

    # ------------------------------------------------------------------
    # Reports
    # ------------------------------------------------------------------

    def summary_report(self) -> dict[str, object]:
        """Generate a comprehensive marketplace summary report.

        Returns
        -------
        dict[str, object]
            A JSON-serializable mapping with the following top-level keys:

            - ``generated_at``       — ISO-8601 UTC timestamp
            - ``version``            — package version
            - ``registry``           — registry statistics
            - ``usage``              — usage and performance statistics
            - ``popular``            — top-10 capabilities by all-time usage
            - ``trending``           — top-10 capabilities by recent usage
            - ``categories``         — breakdown of capabilities by category
            - ``providers``          — breakdown of capabilities by provider
        """
        return {
            "generated_at": datetime.now(tz=timezone.utc).isoformat(),
            "version": __version__,
            "registry": self._registry_section(),
            "usage": self._usage_section(),
            "popular": self._popular_section(top_n=10),
            "trending": self._trending_section(top_n=10),
            "categories": self._categories_section(),
            "providers": self._providers_section(),
        }

    def capability_report(self, capability_id: str) -> dict[str, object]:
        """Generate a detailed report for a single capability.

        Parameters
        ----------
        capability_id:
            The capability to report on.

        Returns
        -------
        dict[str, object]
            Report containing capability metadata and usage statistics, or
            an error dict when the capability is not found.
        """
        try:
            capability = self._store.get(capability_id)
        except KeyError:
            return {
                "error": f"Capability {capability_id!r} not found.",
                "capability_id": capability_id,
            }

        return {
            "generated_at": datetime.now(tz=timezone.utc).isoformat(),
            "capability_id": capability_id,
            "name": capability.name,
            "version": capability.version,
            "provider": capability.provider.name,
            "category": capability.category.value,
            "trust_level": capability.trust_level,
            "usage": {
                "total_invocations": self._usage_tracker.total_invocations()
                if True  # scoped below
                else 0,
                "success_rate": self._usage_tracker.success_rate(capability_id),
                "average_latency_ms": self._usage_tracker.average_latency_ms(
                    capability_id
                ),
                "total_cost_usd": self._usage_tracker.total_cost_usd(capability_id),
            },
        }

    def provider_report(self) -> dict[str, object]:
        """Generate a per-provider breakdown of capabilities and usage.

        Returns
        -------
        dict[str, object]
            Mapping of ``provider_name -> {capability_count, capabilities}``.
        """
        capabilities = self._store.list_all()
        providers: dict[str, dict[str, object]] = {}

        for cap in capabilities:
            provider_name = cap.provider.name
            if provider_name not in providers:
                providers[provider_name] = {
                    "capability_count": 0,
                    "capabilities": [],
                }
            entry = providers[provider_name]
            capability_count = entry["capability_count"]
            if isinstance(capability_count, int):
                entry["capability_count"] = capability_count + 1
            capabilities_list = entry["capabilities"]
            if isinstance(capabilities_list, list):
                capabilities_list.append(
                    {
                        "capability_id": cap.capability_id,
                        "name": cap.name,
                        "version": cap.version,
                        "category": cap.category.value,
                        "trust_level": cap.trust_level,
                    }
                )

        return {
            "generated_at": datetime.now(tz=timezone.utc).isoformat(),
            "providers": providers,
            "total_providers": len(providers),
        }

    # ------------------------------------------------------------------
    # Private section builders
    # ------------------------------------------------------------------

    def _registry_section(self) -> dict[str, object]:
        capabilities = self._store.list_all()
        trust_levels = [cap.trust_level for cap in capabilities]
        avg_trust = (
            round(sum(trust_levels) / len(trust_levels), 4) if trust_levels else 0.0
        )
        return {
            "total_capabilities": len(capabilities),
            "average_trust_level": avg_trust,
        }

    def _usage_section(self) -> dict[str, object]:
        total = self._usage_tracker.total_invocations()
        return {
            "total_invocations": total,
            "global_success_rate": round(self._usage_tracker.success_rate(), 4),
            "global_average_latency_ms": round(
                self._usage_tracker.average_latency_ms(), 4
            ),
            "total_cost_usd": self._usage_tracker.total_cost_usd(),
        }

    def _popular_section(self, top_n: int) -> list[dict[str, object]]:
        return [
            {"capability_id": cap_id, "total_uses": count}
            for cap_id, count in self._usage_tracker.get_popular(top_n=top_n)
        ]

    def _trending_section(self, top_n: int) -> list[dict[str, object]]:
        return [
            {"capability_id": cap_id, "recent_uses": count}
            for cap_id, count in self._usage_tracker.get_trending(top_n=top_n)
        ]

    def _categories_section(self) -> dict[str, int]:
        capabilities = self._store.list_all()
        counts: dict[str, int] = {}
        for cap in capabilities:
            cat = cap.category.value
            counts[cat] = counts.get(cat, 0) + 1
        return counts

    def _providers_section(self) -> dict[str, int]:
        capabilities = self._store.list_all()
        counts: dict[str, int] = {}
        for cap in capabilities:
            provider = cap.provider.name
            counts[provider] = counts.get(provider, 0) + 1
        return counts
