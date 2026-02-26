"""Unit tests for agent-marketplace discovery modules.

Covers ConstraintFilter (filter.py), FitnessRanker (ranker.py),
SearchEngine (search.py), and DiscoveryClient (client.py).
"""
from __future__ import annotations

import pytest

from agent_marketplace.discovery.filter import ConstraintFilter, FilterConstraints
from agent_marketplace.discovery.ranker import FitnessRanker, RankedCapability
from agent_marketplace.discovery.search import SearchEngine
from agent_marketplace.discovery.client import DiscoveryClient, DiscoveryResult
from agent_marketplace.registry.memory_store import MemoryStore
from agent_marketplace.schema.capability import (
    AgentCapability,
    CapabilityCategory,
    LatencyProfile,
    PricingModel,
    QualityMetrics,
)
from agent_marketplace.schema.provider import ProviderInfo


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_cap(
    name: str = "test-cap",
    category: CapabilityCategory = CapabilityCategory.ANALYSIS,
    tags: list[str] | None = None,
    trust_level: float = 0.8,
    cost: float = 0.0,
    pricing_model: PricingModel = PricingModel.FREE,
    supported_languages: list[str] | None = None,
    supported_frameworks: list[str] | None = None,
    latency_p50: float = 0.0,
    latency_p95: float = 0.0,
    quality_metrics: dict[str, float] | None = None,
) -> AgentCapability:
    kwargs: dict = dict(
        name=name,
        version="1.0.0",
        description=f"A {name} capability for testing.",
        category=category,
        tags=tags or [],
        input_types=["application/json"],
        output_type="application/json",
        pricing_model=pricing_model,
        cost=cost,
        trust_level=trust_level,
        supported_languages=supported_languages or [],
        supported_frameworks=supported_frameworks or [],
        provider=ProviderInfo(name="TestProvider"),
    )
    cap = AgentCapability(**kwargs)
    if latency_p50 > 0 or latency_p95 > 0:
        cap = cap.model_copy(update={
            "latency": LatencyProfile(p50_ms=latency_p50, p95_ms=latency_p95)
        })
    if quality_metrics:
        cap = cap.model_copy(update={
            "quality_metrics": QualityMetrics(metrics=quality_metrics)
        })
    return cap


# ---------------------------------------------------------------------------
# FilterConstraints
# ---------------------------------------------------------------------------


class TestFilterConstraints:
    def test_defaults_pass_everything(self) -> None:
        fc = FilterConstraints()
        cap = _make_cap()
        cf = ConstraintFilter(fc)
        assert cf.passes(cap) is True

    def test_constraints_property(self) -> None:
        fc = FilterConstraints(min_trust=0.5)
        cf = ConstraintFilter(fc)
        assert cf.constraints is fc


# ---------------------------------------------------------------------------
# ConstraintFilter — trust
# ---------------------------------------------------------------------------


class TestConstraintFilterTrust:
    def test_trust_above_minimum_passes(self) -> None:
        cap = _make_cap(trust_level=0.8)
        cf = ConstraintFilter(FilterConstraints(min_trust=0.5))
        assert cf.passes(cap) is True

    def test_trust_at_minimum_passes(self) -> None:
        cap = _make_cap(trust_level=0.5)
        cf = ConstraintFilter(FilterConstraints(min_trust=0.5))
        assert cf.passes(cap) is True

    def test_trust_below_minimum_fails(self) -> None:
        cap = _make_cap(trust_level=0.3)
        cf = ConstraintFilter(FilterConstraints(min_trust=0.5))
        assert cf.passes(cap) is False


# ---------------------------------------------------------------------------
# ConstraintFilter — cost
# ---------------------------------------------------------------------------


class TestConstraintFilterCost:
    def test_cost_below_max_passes(self) -> None:
        cap = _make_cap(cost=0.001)
        cf = ConstraintFilter(FilterConstraints(max_cost=0.01))
        assert cf.passes(cap) is True

    def test_cost_above_max_fails(self) -> None:
        cap = _make_cap(cost=1.0)
        cf = ConstraintFilter(FilterConstraints(max_cost=0.01))
        assert cf.passes(cap) is False


# ---------------------------------------------------------------------------
# ConstraintFilter — latency
# ---------------------------------------------------------------------------


class TestConstraintFilterLatency:
    def test_low_latency_passes(self) -> None:
        cap = _make_cap(latency_p95=50.0)
        cf = ConstraintFilter(FilterConstraints(max_p95_latency_ms=100.0))
        assert cf.passes(cap) is True

    def test_high_latency_fails(self) -> None:
        cap = _make_cap(latency_p95=500.0)
        cf = ConstraintFilter(FilterConstraints(max_p95_latency_ms=100.0))
        assert cf.passes(cap) is False


# ---------------------------------------------------------------------------
# ConstraintFilter — category
# ---------------------------------------------------------------------------


class TestConstraintFilterCategory:
    def test_matching_category_passes(self) -> None:
        cap = _make_cap(category=CapabilityCategory.ANALYSIS)
        cf = ConstraintFilter(FilterConstraints(category=CapabilityCategory.ANALYSIS))
        assert cf.passes(cap) is True

    def test_wrong_category_fails(self) -> None:
        cap = _make_cap(category=CapabilityCategory.GENERATION)
        cf = ConstraintFilter(FilterConstraints(category=CapabilityCategory.ANALYSIS))
        assert cf.passes(cap) is False


# ---------------------------------------------------------------------------
# ConstraintFilter — tags
# ---------------------------------------------------------------------------


class TestConstraintFilterTags:
    def test_required_tag_present_passes(self) -> None:
        cap = _make_cap(tags=["pdf", "extraction"])
        cf = ConstraintFilter(FilterConstraints(required_tags=["pdf"]))
        assert cf.passes(cap) is True

    def test_required_tag_absent_fails(self) -> None:
        cap = _make_cap(tags=["image"])
        cf = ConstraintFilter(FilterConstraints(required_tags=["pdf"]))
        assert cf.passes(cap) is False

    def test_tag_match_case_insensitive(self) -> None:
        cap = _make_cap(tags=["PDF"])
        cf = ConstraintFilter(FilterConstraints(required_tags=["pdf"]))
        assert cf.passes(cap) is True

    def test_all_required_tags_must_match(self) -> None:
        cap = _make_cap(tags=["pdf"])
        cf = ConstraintFilter(FilterConstraints(required_tags=["pdf", "extraction"]))
        assert cf.passes(cap) is False


# ---------------------------------------------------------------------------
# ConstraintFilter — language / framework / pricing
# ---------------------------------------------------------------------------


class TestConstraintFilterMisc:
    def test_supported_language_passes(self) -> None:
        cap = _make_cap(supported_languages=["en", "fr"])
        cf = ConstraintFilter(FilterConstraints(supported_language="en"))
        assert cf.passes(cap) is True

    def test_unsupported_language_fails(self) -> None:
        cap = _make_cap(supported_languages=["en"])
        cf = ConstraintFilter(FilterConstraints(supported_language="de"))
        assert cf.passes(cap) is False

    def test_supported_framework_passes(self) -> None:
        cap = _make_cap(supported_frameworks=["langchain"])
        cf = ConstraintFilter(FilterConstraints(supported_framework="langchain"))
        assert cf.passes(cap) is True

    def test_unsupported_framework_fails(self) -> None:
        cap = _make_cap(supported_frameworks=["langchain"])
        cf = ConstraintFilter(FilterConstraints(supported_framework="llamaindex"))
        assert cf.passes(cap) is False

    def test_pricing_model_filter_passes(self) -> None:
        cap = _make_cap(pricing_model=PricingModel.FREE)
        cf = ConstraintFilter(FilterConstraints(pricing_models=["free"]))
        assert cf.passes(cap) is True

    def test_pricing_model_filter_fails(self) -> None:
        cap = _make_cap(pricing_model=PricingModel.PER_CALL)
        cf = ConstraintFilter(FilterConstraints(pricing_models=["free"]))
        assert cf.passes(cap) is False

    def test_quality_metric_filter_passes(self) -> None:
        cap = _make_cap(quality_metrics={"accuracy": 0.9})
        cf = ConstraintFilter(FilterConstraints(
            required_quality_metric="accuracy",
            min_quality_score=0.8,
        ))
        assert cf.passes(cap) is True

    def test_quality_metric_below_min_fails(self) -> None:
        cap = _make_cap(quality_metrics={"accuracy": 0.5})
        cf = ConstraintFilter(FilterConstraints(
            required_quality_metric="accuracy",
            min_quality_score=0.8,
        ))
        assert cf.passes(cap) is False

    def test_missing_quality_metric_fails(self) -> None:
        cap = _make_cap()
        cf = ConstraintFilter(FilterConstraints(
            required_quality_metric="f1",
            min_quality_score=0.5,
        ))
        assert cf.passes(cap) is False


class TestConstraintFilterApply:
    def test_apply_returns_passing_subset(self) -> None:
        caps = [_make_cap(trust_level=0.9), _make_cap(trust_level=0.1)]
        cf = ConstraintFilter(FilterConstraints(min_trust=0.5))
        result = cf.apply(caps)
        assert len(result) == 1
        assert result[0].trust_level == pytest.approx(0.9)

    def test_apply_empty_list(self) -> None:
        cf = ConstraintFilter(FilterConstraints())
        assert cf.apply([]) == []


# ---------------------------------------------------------------------------
# FitnessRanker
# ---------------------------------------------------------------------------


class TestFitnessRankerInit:
    def test_weights_not_summing_raises(self) -> None:
        with pytest.raises(ValueError, match="sum to 1.0"):
            FitnessRanker(relevance_weight=0.5, quality_weight=0.5, cost_weight=0.5, trust_weight=0.5)

    def test_default_weights_sum_to_one(self) -> None:
        ranker = FitnessRanker()
        total = ranker._relevance_weight + ranker._quality_weight + ranker._cost_weight + ranker._trust_weight
        assert total == pytest.approx(1.0)


class TestFitnessRankerRank:
    def test_empty_list_returns_empty(self) -> None:
        ranker = FitnessRanker()
        assert ranker.rank([]) == []

    def test_returns_ranked_capabilities(self) -> None:
        cap = _make_cap(trust_level=0.8)
        ranker = FitnessRanker()
        ranked = ranker.rank([cap])
        assert len(ranked) == 1
        assert isinstance(ranked[0], RankedCapability)

    def test_higher_trust_ranks_higher(self) -> None:
        cap_high = _make_cap(name="high-trust", trust_level=0.9)
        cap_low = _make_cap(name="low-trust", trust_level=0.1)
        ranker = FitnessRanker(relevance_weight=0.0, quality_weight=0.0, cost_weight=0.0, trust_weight=1.0)
        ranked = ranker.rank([cap_low, cap_high])
        assert ranked[0].capability.name == "high-trust"

    def test_sorted_descending_by_fitness(self) -> None:
        caps = [_make_cap(trust_level=t) for t in [0.5, 0.9, 0.2]]
        ranker = FitnessRanker(relevance_weight=0.0, quality_weight=0.0, cost_weight=0.0, trust_weight=1.0)
        ranked = ranker.rank(caps)
        scores = [r.fitness_score for r in ranked]
        assert scores == sorted(scores, reverse=True)

    def test_relevance_scores_applied(self) -> None:
        cap_a = _make_cap(name="cap-a", trust_level=0.5)
        cap_b = _make_cap(name="cap-b", trust_level=0.5)
        rel_scores = {cap_a.capability_id: 1.0, cap_b.capability_id: 0.0}
        ranker = FitnessRanker(relevance_weight=1.0, quality_weight=0.0, cost_weight=0.0, trust_weight=0.0)
        ranked = ranker.rank([cap_a, cap_b], relevance_scores=rel_scores)
        assert ranked[0].capability.name == "cap-a"

    def test_free_capability_has_full_cost_efficiency(self) -> None:
        cap = _make_cap(cost=0.0)
        ranker = FitnessRanker()
        ranked = ranker.rank([cap])
        assert ranked[0].cost_efficiency_score == pytest.approx(1.0)

    def test_quality_score_from_metrics(self) -> None:
        cap = _make_cap(quality_metrics={"accuracy": 0.8, "f1": 0.6})
        ranker = FitnessRanker()
        ranked = ranker.rank([cap])
        assert ranked[0].quality_score == pytest.approx(0.7)

    def test_no_quality_metrics_score_is_zero(self) -> None:
        cap = _make_cap()
        ranker = FitnessRanker()
        ranked = ranker.rank([cap])
        assert ranked[0].quality_score == pytest.approx(0.0)

    def test_cost_efficiency_decreases_with_cost(self) -> None:
        cap_cheap = _make_cap(name="cheap", cost=0.001)
        cap_expensive = _make_cap(name="expensive", cost=1.0)
        ranker = FitnessRanker(relevance_weight=0.0, quality_weight=0.0, cost_weight=1.0, trust_weight=0.0)
        ranked = ranker.rank([cap_cheap, cap_expensive])
        assert ranked[0].capability.name == "cheap"

    def test_missing_relevance_score_defaults_to_one(self) -> None:
        cap = _make_cap()
        ranker = FitnessRanker()
        ranked = ranker.rank([cap], relevance_scores={})
        assert ranked[0].relevance_score == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# SearchEngine
# ---------------------------------------------------------------------------


def _populated_store() -> MemoryStore:
    store = MemoryStore()
    store.register(_make_cap(name="analysis-tool", category=CapabilityCategory.ANALYSIS, tags=["analysis"]))
    store.register(_make_cap(name="gen-tool", category=CapabilityCategory.GENERATION, tags=["generation"]))
    return store


class TestSearchEngine:
    def test_search_returns_ranked_results(self) -> None:
        store = _populated_store()
        engine = SearchEngine(store)
        results = engine.search(keyword="analysis")
        assert len(results) >= 1
        assert isinstance(results[0], RankedCapability)

    def test_empty_keyword_returns_all(self) -> None:
        store = _populated_store()
        engine = SearchEngine(store)
        results = engine.search(keyword="")
        assert len(results) == 2

    def test_tag_filter(self) -> None:
        store = _populated_store()
        engine = SearchEngine(store)
        results = engine.search(tags=["generation"])
        assert all("generation" in r.capability.tags for r in results)

    def test_constraint_filter_applied(self) -> None:
        store = _populated_store()
        store.register(_make_cap(name="expensive", cost=100.0))
        engine = SearchEngine(store)
        from agent_marketplace.discovery.filter import FilterConstraints
        results = engine.search(constraints=FilterConstraints(max_cost=1.0))
        assert not any(r.capability.cost > 1.0 for r in results)

    def test_limit_applied(self) -> None:
        store = _populated_store()
        engine = SearchEngine(store)
        results = engine.search(limit=1)
        assert len(results) == 1

    def test_offset_applied(self) -> None:
        store = _populated_store()
        engine = SearchEngine(store)
        all_results = engine.search()
        offset_results = engine.search(offset=1)
        assert len(offset_results) == len(all_results) - 1

    def test_zero_limit_returns_all(self) -> None:
        store = _populated_store()
        engine = SearchEngine(store)
        results = engine.search(limit=0)
        assert len(results) == 2

    def test_search_by_capability_type_valid_category(self) -> None:
        store = _populated_store()
        engine = SearchEngine(store)
        results = engine.search_by_capability_type("analysis")
        assert len(results) >= 1

    def test_search_by_capability_type_invalid_returns_empty(self) -> None:
        store = _populated_store()
        engine = SearchEngine(store)
        results = engine.search_by_capability_type("not_a_category")
        assert results == []

    def test_search_by_capability_type_with_constraints(self) -> None:
        store = _populated_store()
        store.register(_make_cap(name="expensive-analysis", category=CapabilityCategory.ANALYSIS, cost=999.0))
        engine = SearchEngine(store)
        from agent_marketplace.discovery.filter import FilterConstraints
        results = engine.search_by_capability_type("analysis", constraints=FilterConstraints(max_cost=1.0))
        assert not any(r.capability.cost > 1.0 for r in results)

    def test_compute_relevance_empty_keyword(self) -> None:
        scores = SearchEngine._compute_relevance_scores([_make_cap()], "")
        assert scores == {}

    def test_compute_relevance_empty_capabilities(self) -> None:
        scores = SearchEngine._compute_relevance_scores([], "test")
        assert scores == {}

    def test_name_match_boosts_score(self) -> None:
        cap = _make_cap(name="analysis-tool")
        scores = SearchEngine._compute_relevance_scores([cap], "analysis")
        assert scores[cap.capability_id] > 0.0

    def test_tag_match_boosts_score(self) -> None:
        cap = _make_cap(name="tool", tags=["analysis"])
        scores = SearchEngine._compute_relevance_scores([cap], "analysis")
        assert scores[cap.capability_id] > 0.0

    def test_no_match_gives_zero(self) -> None:
        cap = _make_cap(name="image-gen", tags=["image"])
        scores = SearchEngine._compute_relevance_scores([cap], "zzznomatch")
        assert scores[cap.capability_id] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# DiscoveryClient
# ---------------------------------------------------------------------------


class TestDiscoveryResult:
    def test_capabilities_property(self) -> None:
        from agent_marketplace.discovery.ranker import RankedCapability
        cap = _make_cap()
        rc = RankedCapability(
            capability=cap,
            fitness_score=0.9,
            relevance_score=1.0,
            quality_score=0.0,
            cost_efficiency_score=1.0,
            trust_score=0.8,
        )
        result = DiscoveryResult(ranked_capabilities=[rc], total_found=1)
        assert result.capabilities == [cap]

    def test_best_returns_top_capability(self) -> None:
        from agent_marketplace.discovery.ranker import RankedCapability
        cap = _make_cap()
        rc = RankedCapability(cap, 0.9, 1.0, 0.0, 1.0, 0.8)
        result = DiscoveryResult(ranked_capabilities=[rc], total_found=1)
        assert result.best() == cap

    def test_best_returns_none_when_empty(self) -> None:
        result = DiscoveryResult(ranked_capabilities=[], total_found=0)
        assert result.best() is None


class TestDiscoveryClient:
    def _make_client(self, caps=None):
        store = MemoryStore()
        if caps:
            for cap in caps:
                store.register(cap)
        return DiscoveryClient(store=store, use_embeddings=False)

    def test_discover_returns_discovery_result(self) -> None:
        client = self._make_client([_make_cap(name="analysis-tool", tags=["analysis"])])
        result = client.discover("analysis")
        assert isinstance(result, DiscoveryResult)

    def test_discover_empty_store(self) -> None:
        client = self._make_client()
        result = client.discover("anything")
        assert result.total_found == 0

    def test_discover_with_constraints(self) -> None:
        cap_high = _make_cap(name="high-trust", trust_level=0.9)
        cap_low = _make_cap(name="low-trust", trust_level=0.1)
        client = self._make_client([cap_high, cap_low])
        from agent_marketplace.discovery.filter import FilterConstraints
        result = client.discover("", constraints=FilterConstraints(min_trust=0.5))
        assert all(c.trust_level >= 0.5 for c in result.capabilities)

    def test_discover_pagination_limit(self) -> None:
        caps = [_make_cap(name=f"cap-{i}") for i in range(5)]
        client = self._make_client(caps)
        result = client.discover("", limit=2)
        assert len(result.ranked_capabilities) == 2

    def test_discover_pagination_offset(self) -> None:
        caps = [_make_cap(name=f"cap-{i}") for i in range(5)]
        client = self._make_client(caps)
        all_result = client.discover("")
        offset_result = client.discover("", offset=2)
        assert len(offset_result.ranked_capabilities) == len(all_result.ranked_capabilities) - 2

    def test_discover_zero_limit_returns_all(self) -> None:
        caps = [_make_cap(name=f"cap-{i}") for i in range(3)]
        client = self._make_client(caps)
        result = client.discover("", limit=0)
        assert len(result.ranked_capabilities) == 3

    def test_discover_query_keyword_stored(self) -> None:
        client = self._make_client([_make_cap()])
        result = client.discover("my query")
        assert result.query_keyword == "my query"

    def test_refresh_index(self) -> None:
        client = self._make_client([_make_cap()])
        client.refresh_index()
        assert client._index_built is True

    def test_get_by_id_known(self) -> None:
        cap = _make_cap()
        client = self._make_client([cap])
        fetched = client.get_by_id(cap.capability_id)
        assert fetched.capability_id == cap.capability_id

    def test_get_by_id_unknown_raises(self) -> None:
        client = self._make_client()
        with pytest.raises(KeyError):
            client.get_by_id("no-such-id")

    def test_keyword_relevance_scores(self) -> None:
        cap = _make_cap(name="analysis-tool", tags=["analysis"])
        scores = DiscoveryClient._keyword_relevance_scores([cap], "analysis")
        assert cap.capability_id in scores
        assert scores[cap.capability_id] > 0.0

    def test_capability_to_text(self) -> None:
        cap = _make_cap(name="pdf-extractor", tags=["pdf"])
        text = DiscoveryClient._capability_to_text(cap)
        assert "pdf-extractor" in text
        assert "pdf" in text
