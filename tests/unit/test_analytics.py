"""Unit tests for agent-marketplace analytics modules.

Covers UsageTracker (usage.py) and MarketplaceReporter (reporter.py).
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import pytest

from agent_marketplace.analytics.reporter import MarketplaceReporter
from agent_marketplace.analytics.usage import UsageRecord, UsageTracker


# ---------------------------------------------------------------------------
# UsageRecord
# ---------------------------------------------------------------------------


class TestUsageRecord:
    def test_defaults(self) -> None:
        record = UsageRecord(capability_id="cap-1", provider_id="prov-1")
        assert record.success is True
        assert record.latency_ms == 0.0
        assert record.cost_usd == 0.0
        assert record.caller_id == ""
        assert isinstance(record.recorded_at, datetime)

    def test_custom_values(self) -> None:
        ts = datetime(2025, 1, 1, tzinfo=timezone.utc)
        record = UsageRecord(
            capability_id="cap-2",
            provider_id="prov-2",
            success=False,
            latency_ms=42.5,
            cost_usd=0.001,
            recorded_at=ts,
            caller_id="user-99",
        )
        assert record.capability_id == "cap-2"
        assert record.success is False
        assert record.latency_ms == pytest.approx(42.5)
        assert record.cost_usd == pytest.approx(0.001)
        assert record.caller_id == "user-99"


# ---------------------------------------------------------------------------
# UsageTracker — construction
# ---------------------------------------------------------------------------


class TestUsageTrackerInit:
    def test_default_window(self) -> None:
        tracker = UsageTracker()
        assert tracker._trending_window_hours == 24

    def test_custom_window(self) -> None:
        tracker = UsageTracker(trending_window_hours=48)
        assert tracker._trending_window_hours == 48

    def test_invalid_window_raises(self) -> None:
        with pytest.raises(ValueError, match="trending_window_hours"):
            UsageTracker(trending_window_hours=0)

    def test_negative_window_raises(self) -> None:
        with pytest.raises(ValueError):
            UsageTracker(trending_window_hours=-1)


# ---------------------------------------------------------------------------
# UsageTracker — record_usage
# ---------------------------------------------------------------------------


class TestUsageTrackerRecordUsage:
    def setup_method(self) -> None:
        self.tracker = UsageTracker()

    def test_returns_usage_record(self) -> None:
        record = self.tracker.record_usage("cap-1", provider_id="prov-1")
        assert isinstance(record, UsageRecord)

    def test_record_stored(self) -> None:
        self.tracker.record_usage("cap-1")
        assert self.tracker.total_invocations() == 1

    def test_multiple_records_accumulate(self) -> None:
        for _ in range(5):
            self.tracker.record_usage("cap-1")
        assert self.tracker.total_invocations() == 5

    def test_custom_timestamp_accepted(self) -> None:
        ts = datetime(2024, 6, 1, tzinfo=timezone.utc)
        record = self.tracker.record_usage("cap-1", recorded_at=ts)
        assert record.recorded_at == ts

    def test_success_flag_stored(self) -> None:
        record = self.tracker.record_usage("cap-1", success=False)
        assert record.success is False

    def test_latency_stored(self) -> None:
        record = self.tracker.record_usage("cap-1", latency_ms=100.0)
        assert record.latency_ms == pytest.approx(100.0)

    def test_cost_stored(self) -> None:
        record = self.tracker.record_usage("cap-1", cost_usd=0.01)
        assert record.cost_usd == pytest.approx(0.01)

    def test_caller_id_stored(self) -> None:
        record = self.tracker.record_usage("cap-1", caller_id="user-42")
        assert record.caller_id == "user-42"


# ---------------------------------------------------------------------------
# UsageTracker — get_popular
# ---------------------------------------------------------------------------


class TestUsageTrackerGetPopular:
    def setup_method(self) -> None:
        self.tracker = UsageTracker()

    def test_empty_tracker_returns_empty(self) -> None:
        assert self.tracker.get_popular() == []

    def test_single_capability(self) -> None:
        self.tracker.record_usage("cap-1")
        popular = self.tracker.get_popular()
        assert popular == [("cap-1", 1)]

    def test_ordered_descending(self) -> None:
        for _ in range(3):
            self.tracker.record_usage("cap-a")
        self.tracker.record_usage("cap-b")
        popular = self.tracker.get_popular()
        assert popular[0][0] == "cap-a"

    def test_top_n_limits_results(self) -> None:
        for i in range(10):
            for _ in range(i + 1):
                self.tracker.record_usage(f"cap-{i}")
        popular = self.tracker.get_popular(top_n=3)
        assert len(popular) == 3


# ---------------------------------------------------------------------------
# UsageTracker — get_trending
# ---------------------------------------------------------------------------


class TestUsageTrackerGetTrending:
    def test_recent_usage_included(self) -> None:
        tracker = UsageTracker()
        tracker.record_usage("cap-1")
        trending = tracker.get_trending(window_hours=1)
        assert any(cap_id == "cap-1" for cap_id, _ in trending)

    def test_old_usage_excluded(self) -> None:
        tracker = UsageTracker()
        old_ts = datetime.now(tz=timezone.utc) - timedelta(hours=48)
        tracker.record_usage("cap-old", recorded_at=old_ts)
        trending = tracker.get_trending(window_hours=1)
        assert not any(cap_id == "cap-old" for cap_id, _ in trending)

    def test_default_window_used_when_not_specified(self) -> None:
        tracker = UsageTracker(trending_window_hours=2)
        tracker.record_usage("cap-1")
        trending = tracker.get_trending()
        assert any(cap_id == "cap-1" for cap_id, _ in trending)

    def test_top_n_limits_results(self) -> None:
        tracker = UsageTracker()
        for i in range(5):
            tracker.record_usage(f"cap-{i}")
        trending = tracker.get_trending(top_n=2)
        assert len(trending) == 2


# ---------------------------------------------------------------------------
# UsageTracker — success_rate
# ---------------------------------------------------------------------------


class TestUsageTrackerSuccessRate:
    def test_empty_returns_zero(self) -> None:
        tracker = UsageTracker()
        assert tracker.success_rate() == 0.0

    def test_all_success(self) -> None:
        tracker = UsageTracker()
        for _ in range(4):
            tracker.record_usage("cap-1", success=True)
        assert tracker.success_rate() == pytest.approx(1.0)

    def test_all_failure(self) -> None:
        tracker = UsageTracker()
        for _ in range(3):
            tracker.record_usage("cap-1", success=False)
        assert tracker.success_rate() == pytest.approx(0.0)

    def test_mixed_success_rate(self) -> None:
        tracker = UsageTracker()
        tracker.record_usage("cap-1", success=True)
        tracker.record_usage("cap-1", success=False)
        assert tracker.success_rate() == pytest.approx(0.5)

    def test_per_capability_filter(self) -> None:
        tracker = UsageTracker()
        tracker.record_usage("cap-a", success=True)
        tracker.record_usage("cap-b", success=False)
        assert tracker.success_rate("cap-a") == pytest.approx(1.0)
        assert tracker.success_rate("cap-b") == pytest.approx(0.0)

    def test_unknown_capability_returns_zero(self) -> None:
        tracker = UsageTracker()
        tracker.record_usage("cap-1")
        assert tracker.success_rate("no-such-cap") == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# UsageTracker — average_latency_ms
# ---------------------------------------------------------------------------


class TestUsageTrackerAverageLatency:
    def test_empty_returns_zero(self) -> None:
        tracker = UsageTracker()
        assert tracker.average_latency_ms() == 0.0

    def test_all_zero_latency_excluded(self) -> None:
        tracker = UsageTracker()
        tracker.record_usage("cap-1", latency_ms=0.0)
        assert tracker.average_latency_ms() == 0.0

    def test_average_computed(self) -> None:
        tracker = UsageTracker()
        tracker.record_usage("cap-1", latency_ms=100.0)
        tracker.record_usage("cap-1", latency_ms=200.0)
        assert tracker.average_latency_ms() == pytest.approx(150.0)

    def test_per_capability(self) -> None:
        tracker = UsageTracker()
        tracker.record_usage("cap-a", latency_ms=50.0)
        tracker.record_usage("cap-b", latency_ms=200.0)
        assert tracker.average_latency_ms("cap-a") == pytest.approx(50.0)


# ---------------------------------------------------------------------------
# UsageTracker — total_cost_usd
# ---------------------------------------------------------------------------


class TestUsageTrackerTotalCost:
    def test_empty_returns_zero(self) -> None:
        tracker = UsageTracker()
        assert tracker.total_cost_usd() == 0.0

    def test_sums_costs(self) -> None:
        tracker = UsageTracker()
        tracker.record_usage("cap-1", cost_usd=0.001)
        tracker.record_usage("cap-1", cost_usd=0.002)
        assert tracker.total_cost_usd() == pytest.approx(0.003)

    def test_per_capability_filter(self) -> None:
        tracker = UsageTracker()
        tracker.record_usage("cap-a", cost_usd=1.0)
        tracker.record_usage("cap-b", cost_usd=2.0)
        assert tracker.total_cost_usd("cap-a") == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# UsageTracker — list_records
# ---------------------------------------------------------------------------


class TestUsageTrackerListRecords:
    def test_returns_all_when_no_filter(self) -> None:
        tracker = UsageTracker()
        tracker.record_usage("cap-1")
        tracker.record_usage("cap-2")
        records = tracker.list_records()
        assert len(records) == 2

    def test_filter_by_capability(self) -> None:
        tracker = UsageTracker()
        tracker.record_usage("cap-1")
        tracker.record_usage("cap-2")
        records = tracker.list_records(capability_id="cap-1")
        assert all(r.capability_id == "cap-1" for r in records)

    def test_limit_applied(self) -> None:
        tracker = UsageTracker()
        for _ in range(10):
            tracker.record_usage("cap-1")
        records = tracker.list_records(limit=3)
        assert len(records) == 3

    def test_sorted_most_recent_first(self) -> None:
        tracker = UsageTracker()
        old = datetime(2024, 1, 1, tzinfo=timezone.utc)
        new = datetime(2025, 1, 1, tzinfo=timezone.utc)
        tracker.record_usage("cap-1", recorded_at=old)
        tracker.record_usage("cap-1", recorded_at=new)
        records = tracker.list_records()
        assert records[0].recorded_at == new


# ---------------------------------------------------------------------------
# MarketplaceReporter
# ---------------------------------------------------------------------------


def _make_mock_store(capabilities=None):
    """Build a mock RegistryStore that returns the given capabilities."""
    from agent_marketplace.schema.capability import AgentCapability, CapabilityCategory, PricingModel
    from agent_marketplace.schema.provider import ProviderInfo

    if capabilities is None:
        cap = AgentCapability(
            name="Test Cap",
            version="1.0.0",
            description="A test capability.",
            category=CapabilityCategory.ANALYSIS,
            tags=[],
            input_types=["application/json"],
            output_type="application/json",
            pricing_model=PricingModel.FREE,
            provider=ProviderInfo(name="TestProvider"),
            trust_level=0.8,
        )
        capabilities = [cap]

    store = MagicMock()
    store.list_all.return_value = capabilities
    store.count.return_value = len(capabilities)

    def _get(cap_id: str):
        for c in capabilities:
            if c.capability_id == cap_id:
                return c
        raise KeyError(cap_id)

    store.get.side_effect = _get
    return store


def _make_tracker_with_data() -> UsageTracker:
    tracker = UsageTracker()
    tracker.record_usage("cap-1", success=True, latency_ms=50.0, cost_usd=0.001)
    tracker.record_usage("cap-1", success=False, latency_ms=80.0, cost_usd=0.002)
    tracker.record_usage("cap-2", success=True, latency_ms=10.0, cost_usd=0.0)
    return tracker


class TestMarketplaceReporterSummaryReport:
    def test_returns_dict_with_required_keys(self) -> None:
        store = _make_mock_store()
        tracker = UsageTracker()
        reporter = MarketplaceReporter(store, tracker)
        report = reporter.summary_report()
        for key in ("generated_at", "version", "registry", "usage", "popular", "trending", "categories", "providers"):
            assert key in report

    def test_registry_section_counts(self) -> None:
        store = _make_mock_store()
        tracker = UsageTracker()
        reporter = MarketplaceReporter(store, tracker)
        report = reporter.summary_report()
        assert report["registry"]["total_capabilities"] == 1

    def test_usage_section_totals(self) -> None:
        store = _make_mock_store()
        tracker = _make_tracker_with_data()
        reporter = MarketplaceReporter(store, tracker)
        report = reporter.summary_report()
        assert report["usage"]["total_invocations"] == 3

    def test_popular_section_is_list(self) -> None:
        store = _make_mock_store()
        tracker = _make_tracker_with_data()
        reporter = MarketplaceReporter(store, tracker)
        report = reporter.summary_report()
        assert isinstance(report["popular"], list)

    def test_categories_section_is_dict(self) -> None:
        store = _make_mock_store()
        tracker = UsageTracker()
        reporter = MarketplaceReporter(store, tracker)
        report = reporter.summary_report()
        assert isinstance(report["categories"], dict)

    def test_empty_store_average_trust_is_zero(self) -> None:
        store = _make_mock_store(capabilities=[])
        tracker = UsageTracker()
        reporter = MarketplaceReporter(store, tracker)
        report = reporter.summary_report()
        assert report["registry"]["average_trust_level"] == 0.0


class TestMarketplaceReporterCapabilityReport:
    def test_known_capability(self) -> None:
        store = _make_mock_store()
        tracker = _make_tracker_with_data()
        reporter = MarketplaceReporter(store, tracker)
        caps = store.list_all()
        cap_id = caps[0].capability_id
        report = reporter.capability_report(cap_id)
        assert "name" in report
        assert "usage" in report

    def test_unknown_capability_returns_error(self) -> None:
        store = _make_mock_store()
        tracker = UsageTracker()
        reporter = MarketplaceReporter(store, tracker)
        report = reporter.capability_report("does-not-exist")
        assert "error" in report

    def test_capability_report_has_provider(self) -> None:
        store = _make_mock_store()
        tracker = UsageTracker()
        reporter = MarketplaceReporter(store, tracker)
        caps = store.list_all()
        report = reporter.capability_report(caps[0].capability_id)
        assert report["provider"] == "TestProvider"


class TestMarketplaceReporterProviderReport:
    def test_returns_dict_with_providers_key(self) -> None:
        store = _make_mock_store()
        tracker = UsageTracker()
        reporter = MarketplaceReporter(store, tracker)
        report = reporter.provider_report()
        assert "providers" in report
        assert "total_providers" in report

    def test_provider_count(self) -> None:
        store = _make_mock_store()
        tracker = UsageTracker()
        reporter = MarketplaceReporter(store, tracker)
        report = reporter.provider_report()
        assert report["total_providers"] == 1

    def test_empty_store_returns_zero_providers(self) -> None:
        store = _make_mock_store(capabilities=[])
        tracker = UsageTracker()
        reporter = MarketplaceReporter(store, tracker)
        report = reporter.provider_report()
        assert report["total_providers"] == 0
