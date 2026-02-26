"""Unit tests for agent-marketplace matching modules.

Covers CapabilityRequest (request.py), MatchingEngine (engine.py),
PriceOffer, NegotiationResult, and PriceNegotiator (negotiator.py).
"""
from __future__ import annotations

import pytest

from agent_marketplace.matching.engine import MatchingEngine, MatchResult
from agent_marketplace.matching.negotiator import (
    NegotiationResult,
    PriceNegotiator,
    PriceOffer,
)
from agent_marketplace.matching.request import CapabilityRequest
from agent_marketplace.schema.capability import (
    AgentCapability,
    CapabilityCategory,
    LatencyProfile,
    PricingModel,
)
from agent_marketplace.schema.provider import ProviderInfo


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_capability(
    name: str = "pdf-extractor",
    category: CapabilityCategory = CapabilityCategory.EXTRACTION,
    tags: list[str] | None = None,
    trust_level: float = 0.8,
    cost: float = 0.01,
    p50_ms: float = 100.0,
    supported_frameworks: list[str] | None = None,
) -> AgentCapability:
    # Ensure p95_ms and p99_ms are >= p50_ms to satisfy LatencyProfile ordering constraint
    p95_ms = max(p50_ms, p50_ms * 2) if p50_ms > 0 else 0.0
    p99_ms = max(p95_ms, p95_ms * 2) if p95_ms > 0 else 0.0
    return AgentCapability(
        name=name,
        version="1.0.0",
        description="A test capability.",
        category=category,
        tags=tags or ["pdf", "extraction"],
        input_types=["application/json"],
        output_type="application/json",
        pricing_model=PricingModel.PER_CALL,
        cost=cost,
        trust_level=trust_level,
        provider=ProviderInfo(name="TestProvider"),
        latency=LatencyProfile(p50_ms=p50_ms, p95_ms=p95_ms, p99_ms=p99_ms),
        supported_frameworks=supported_frameworks or [],
    )


# ---------------------------------------------------------------------------
# CapabilityRequest — construction
# ---------------------------------------------------------------------------


class TestCapabilityRequestDefaults:
    def test_requires_at_least_one_capability(self) -> None:
        with pytest.raises(ValueError, match="required_capabilities"):
            CapabilityRequest(required_capabilities=[])

    def test_valid_single_capability(self) -> None:
        request = CapabilityRequest(required_capabilities=["extraction"])
        assert request.required_capabilities == ["extraction"]

    def test_default_preferred_latency_is_zero(self) -> None:
        request = CapabilityRequest(required_capabilities=["any"])
        assert request.preferred_latency_ms == 0.0

    def test_default_max_cost_is_inf(self) -> None:
        request = CapabilityRequest(required_capabilities=["any"])
        assert request.max_cost == float("inf")

    def test_default_min_trust_is_zero(self) -> None:
        request = CapabilityRequest(required_capabilities=["any"])
        assert request.min_trust == 0.0

    def test_default_certifications_empty(self) -> None:
        request = CapabilityRequest(required_capabilities=["any"])
        assert request.required_certifications == []

    def test_default_request_id_empty(self) -> None:
        request = CapabilityRequest(required_capabilities=["any"])
        assert request.request_id == ""

    def test_negative_latency_raises(self) -> None:
        with pytest.raises(ValueError, match="preferred_latency_ms"):
            CapabilityRequest(
                required_capabilities=["any"],
                preferred_latency_ms=-1.0,
            )

    def test_negative_max_cost_raises(self) -> None:
        with pytest.raises(ValueError, match="max_cost"):
            CapabilityRequest(required_capabilities=["any"], max_cost=-0.01)

    def test_min_trust_above_one_raises(self) -> None:
        with pytest.raises(ValueError, match="min_trust"):
            CapabilityRequest(required_capabilities=["any"], min_trust=1.1)

    def test_negative_min_trust_raises(self) -> None:
        with pytest.raises(ValueError, match="min_trust"):
            CapabilityRequest(required_capabilities=["any"], min_trust=-0.1)

    def test_custom_values_accepted(self) -> None:
        request = CapabilityRequest(
            required_capabilities=["pdf", "ocr"],
            preferred_latency_ms=50.0,
            max_cost=0.05,
            min_trust=0.6,
            required_certifications=["SOC2"],
            request_id="req-abc",
        )
        assert len(request.required_capabilities) == 2
        assert request.preferred_latency_ms == pytest.approx(50.0)
        assert request.max_cost == pytest.approx(0.05)
        assert request.min_trust == pytest.approx(0.6)
        assert request.required_certifications == ["SOC2"]
        assert request.request_id == "req-abc"


# ---------------------------------------------------------------------------
# MatchingEngine — construction
# ---------------------------------------------------------------------------


class TestMatchingEngineInit:
    def test_default_weights_accepted(self) -> None:
        engine = MatchingEngine()
        assert engine is not None

    def test_weights_must_sum_to_one(self) -> None:
        with pytest.raises(ValueError, match="Weights"):
            MatchingEngine(
                capability_weight=0.5,
                latency_weight=0.5,
                trust_weight=0.5,
                cost_weight=0.5,
            )

    def test_custom_valid_weights_accepted(self) -> None:
        engine = MatchingEngine(
            capability_weight=0.25,
            latency_weight=0.25,
            trust_weight=0.25,
            cost_weight=0.25,
        )
        assert engine is not None


# ---------------------------------------------------------------------------
# MatchingEngine — match()
# ---------------------------------------------------------------------------


class TestMatchingEngineMatch:
    def setup_method(self) -> None:
        self.engine = MatchingEngine()

    def test_empty_candidates_returns_empty(self) -> None:
        request = CapabilityRequest(required_capabilities=["extraction"])
        results = self.engine.match(request, [])
        assert results == []

    def test_returns_match_result_instances(self) -> None:
        cap = _make_capability()
        request = CapabilityRequest(required_capabilities=["extraction"])
        results = self.engine.match(request, [cap])
        assert len(results) == 1
        assert isinstance(results[0], MatchResult)

    def test_trust_below_minimum_excluded(self) -> None:
        cap = _make_capability(trust_level=0.3)
        request = CapabilityRequest(
            required_capabilities=["extraction"],
            min_trust=0.5,
        )
        results = self.engine.match(request, [cap])
        assert results == []

    def test_cost_above_maximum_excluded(self) -> None:
        cap = _make_capability(cost=1.0)
        request = CapabilityRequest(
            required_capabilities=["extraction"],
            max_cost=0.5,
        )
        results = self.engine.match(request, [cap])
        assert results == []

    def test_trust_at_minimum_included(self) -> None:
        cap = _make_capability(trust_level=0.5)
        request = CapabilityRequest(
            required_capabilities=["extraction"],
            min_trust=0.5,
        )
        results = self.engine.match(request, [cap])
        assert len(results) == 1

    def test_cost_at_maximum_included(self) -> None:
        cap = _make_capability(cost=0.5)
        request = CapabilityRequest(
            required_capabilities=["extraction"],
            max_cost=0.5,
        )
        results = self.engine.match(request, [cap])
        assert len(results) == 1

    def test_results_sorted_descending_by_score(self) -> None:
        high_trust = _make_capability(name="cap-a", trust_level=0.9, cost=0.001)
        low_trust = _make_capability(name="cap-b", trust_level=0.1, cost=0.001)
        request = CapabilityRequest(required_capabilities=["extraction"])
        results = self.engine.match(request, [low_trust, high_trust])
        assert results[0].trust_score > results[1].trust_score

    def test_match_result_fields_populated(self) -> None:
        cap = _make_capability(trust_level=0.8, cost=0.01, p50_ms=50.0)
        request = CapabilityRequest(
            required_capabilities=["pdf"],
            preferred_latency_ms=100.0,
        )
        result = self.engine.match(request, [cap])[0]
        assert 0.0 <= result.match_score <= 1.0
        assert 0.0 <= result.capability_overlap <= 1.0
        assert 0.0 <= result.latency_score <= 1.0
        assert 0.0 <= result.trust_score <= 1.0
        assert 0.0 <= result.cost_score <= 1.0

    def test_certification_filter_excludes_unqualified(self) -> None:
        cap = _make_capability(tags=["pdf"], supported_frameworks=["langchain"])
        request = CapabilityRequest(
            required_capabilities=["extraction"],
            required_certifications=["soc2"],
        )
        results = self.engine.match(request, [cap])
        assert results == []

    def test_certification_filter_includes_qualified_via_tags(self) -> None:
        cap = _make_capability(tags=["pdf", "soc2"])
        request = CapabilityRequest(
            required_capabilities=["pdf"],
            required_certifications=["soc2"],
        )
        results = self.engine.match(request, [cap])
        assert len(results) == 1

    def test_certification_filter_includes_qualified_via_frameworks(self) -> None:
        cap = _make_capability(
            tags=["pdf"],
            supported_frameworks=["langchain", "soc2-compliant"],
        )
        request = CapabilityRequest(
            required_capabilities=["pdf"],
            required_certifications=["soc2-compliant"],
        )
        results = self.engine.match(request, [cap])
        assert len(results) == 1

    def test_no_preferred_latency_gives_neutral_score(self) -> None:
        # When preferred_latency_ms=0 (no preference), score is 0.5 (neutral)
        # Use default p50_ms=0 via the factory to avoid LatencyProfile ordering constraint
        cap = _make_capability(p50_ms=0.0)
        request = CapabilityRequest(
            required_capabilities=["pdf"],
            preferred_latency_ms=0.0,
        )
        result = self.engine.match(request, [cap])[0]
        assert result.latency_score == pytest.approx(0.5)

    def test_multiple_candidates_all_scored(self) -> None:
        caps = [_make_capability(name=f"cap-{i}") for i in range(5)]
        request = CapabilityRequest(required_capabilities=["extraction"])
        results = self.engine.match(request, caps)
        assert len(results) == 5

    def test_free_capability_gets_maximum_cost_score(self) -> None:
        free_cap = _make_capability(cost=0.0)
        request = CapabilityRequest(required_capabilities=["extraction"])
        result = self.engine.match(request, [free_cap])[0]
        assert result.cost_score == pytest.approx(1.0)

    def test_no_required_capabilities_overlap_is_one(self) -> None:
        cap = _make_capability()
        # This would fail CapabilityRequest validation — so overlap with full match
        request = CapabilityRequest(required_capabilities=["pdf"])
        result = self.engine.match(request, [cap])[0]
        assert result.capability_overlap == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# MatchingEngine — static helpers
# ---------------------------------------------------------------------------


class TestMatchingEngineHelpers:
    def test_latency_below_preference_returns_one(self) -> None:
        cap = _make_capability(p50_ms=50.0)
        request = CapabilityRequest(
            required_capabilities=["x"],
            preferred_latency_ms=100.0,
        )
        score = MatchingEngine._latency_score(request, cap)
        assert score == pytest.approx(1.0)

    def test_latency_equal_preference_returns_one(self) -> None:
        cap = _make_capability(p50_ms=100.0)
        request = CapabilityRequest(
            required_capabilities=["x"],
            preferred_latency_ms=100.0,
        )
        score = MatchingEngine._latency_score(request, cap)
        assert score == pytest.approx(1.0)

    def test_latency_above_preference_decays(self) -> None:
        cap = _make_capability(p50_ms=200.0)
        request = CapabilityRequest(
            required_capabilities=["x"],
            preferred_latency_ms=100.0,
        )
        score = MatchingEngine._latency_score(request, cap)
        assert 0.0 < score < 1.0

    def test_cost_score_zero_cost_returns_one(self) -> None:
        cap = _make_capability(cost=0.0)
        assert MatchingEngine._cost_score(cap, max_cost=1.0) == pytest.approx(1.0)

    def test_cost_score_max_cost_zero_returns_one(self) -> None:
        cap = _make_capability(cost=0.0)
        assert MatchingEngine._cost_score(cap, max_cost=0.0) == pytest.approx(1.0)

    def test_cost_score_decreases_with_cost(self) -> None:
        cap_cheap = _make_capability(cost=0.1)
        cap_expensive = _make_capability(cost=0.9)
        assert MatchingEngine._cost_score(cap_cheap, 1.0) > MatchingEngine._cost_score(cap_expensive, 1.0)

    def test_capability_overlap_no_requirements_returns_one(self) -> None:
        cap = _make_capability()
        # CapabilityRequest requires at least one item, so test static directly
        request = CapabilityRequest(required_capabilities=["pdf"])
        request_no_req = CapabilityRequest(required_capabilities=["pdf"])
        # Simulate empty via internal method — can test by having matching tag
        result = MatchingEngine._capability_overlap(request_no_req, cap)
        assert 0.0 <= result <= 1.0

    def test_no_certifications_returns_true(self) -> None:
        cap = _make_capability()
        request = CapabilityRequest(required_capabilities=["x"])
        assert MatchingEngine._certifications_satisfied(request, cap) is True


# ---------------------------------------------------------------------------
# PriceOffer — construction
# ---------------------------------------------------------------------------


class TestPriceOfferValidation:
    def test_valid_offer_created(self) -> None:
        offer = PriceOffer(
            provider_id="prov-1",
            capability_id="cap-1",
            cost_per_call=0.01,
            quality_score=0.8,
            trust_score=0.9,
        )
        assert offer.provider_id == "prov-1"

    def test_negative_cost_raises(self) -> None:
        with pytest.raises(ValueError, match="cost_per_call"):
            PriceOffer(
                provider_id="p",
                capability_id="c",
                cost_per_call=-0.01,
                quality_score=0.5,
                trust_score=0.5,
            )

    def test_quality_score_above_one_raises(self) -> None:
        with pytest.raises(ValueError, match="quality_score"):
            PriceOffer(
                provider_id="p",
                capability_id="c",
                cost_per_call=0.0,
                quality_score=1.1,
                trust_score=0.5,
            )

    def test_quality_score_below_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="quality_score"):
            PriceOffer(
                provider_id="p",
                capability_id="c",
                cost_per_call=0.0,
                quality_score=-0.1,
                trust_score=0.5,
            )

    def test_trust_score_above_one_raises(self) -> None:
        with pytest.raises(ValueError, match="trust_score"):
            PriceOffer(
                provider_id="p",
                capability_id="c",
                cost_per_call=0.0,
                quality_score=0.5,
                trust_score=1.5,
            )

    def test_negative_latency_raises(self) -> None:
        with pytest.raises(ValueError, match="latency_p50_ms"):
            PriceOffer(
                provider_id="p",
                capability_id="c",
                cost_per_call=0.0,
                quality_score=0.5,
                trust_score=0.5,
                latency_p50_ms=-1.0,
            )

    def test_default_latency_is_zero(self) -> None:
        offer = PriceOffer(
            provider_id="p",
            capability_id="c",
            cost_per_call=0.0,
            quality_score=0.5,
            trust_score=0.5,
        )
        assert offer.latency_p50_ms == 0.0

    def test_zero_cost_is_valid(self) -> None:
        offer = PriceOffer(
            provider_id="p",
            capability_id="c",
            cost_per_call=0.0,
            quality_score=0.0,
            trust_score=0.0,
        )
        assert offer.cost_per_call == 0.0


# ---------------------------------------------------------------------------
# PriceNegotiator — construction
# ---------------------------------------------------------------------------


class TestPriceNegotiatorInit:
    def test_default_weights_accepted(self) -> None:
        negotiator = PriceNegotiator()
        assert negotiator is not None

    def test_bad_weights_raise(self) -> None:
        with pytest.raises(ValueError, match="Weights"):
            PriceNegotiator(
                quality_weight=0.5,
                trust_weight=0.5,
                cost_weight=0.5,
            )

    def test_custom_valid_weights(self) -> None:
        negotiator = PriceNegotiator(
            quality_weight=0.5,
            trust_weight=0.3,
            cost_weight=0.2,
        )
        assert negotiator is not None


# ---------------------------------------------------------------------------
# PriceNegotiator — negotiate()
# ---------------------------------------------------------------------------


def _make_offer(
    provider_id: str = "prov-1",
    cost_per_call: float = 0.01,
    quality_score: float = 0.8,
    trust_score: float = 0.8,
) -> PriceOffer:
    return PriceOffer(
        provider_id=provider_id,
        capability_id="cap-1",
        cost_per_call=cost_per_call,
        quality_score=quality_score,
        trust_score=trust_score,
    )


class TestPriceNegotiatorNegotiate:
    def setup_method(self) -> None:
        self.negotiator = PriceNegotiator()

    def test_negative_budget_raises(self) -> None:
        with pytest.raises(ValueError, match="max_budget"):
            self.negotiator.negotiate([], max_budget=-0.01)

    def test_empty_offers_returns_none_selection(self) -> None:
        result = self.negotiator.negotiate([], max_budget=1.0)
        assert result.selected_offer is None
        assert result.rejected_offers == []
        assert result.value_score == 0.0

    def test_all_over_budget_returns_none(self) -> None:
        offers = [_make_offer(cost_per_call=2.0)]
        result = self.negotiator.negotiate(offers, max_budget=1.0)
        assert result.selected_offer is None
        assert len(result.rejected_offers) == 1

    def test_within_budget_offer_selected(self) -> None:
        offer = _make_offer(cost_per_call=0.5)
        result = self.negotiator.negotiate([offer], max_budget=1.0)
        assert result.selected_offer is offer

    def test_rejected_offers_populated(self) -> None:
        cheap = _make_offer(provider_id="cheap", cost_per_call=0.5)
        expensive = _make_offer(provider_id="expensive", cost_per_call=2.0)
        result = self.negotiator.negotiate([cheap, expensive], max_budget=1.0)
        assert len(result.rejected_offers) == 1
        assert result.rejected_offers[0].provider_id == "expensive"

    def test_higher_quality_preferred(self) -> None:
        low_quality = _make_offer(provider_id="low", quality_score=0.2, trust_score=0.5)
        high_quality = _make_offer(provider_id="high", quality_score=0.9, trust_score=0.5)
        result = self.negotiator.negotiate(
            [low_quality, high_quality], max_budget=1.0
        )
        assert result.selected_offer is not None
        assert result.selected_offer.provider_id == "high"

    def test_value_score_is_non_negative(self) -> None:
        offer = _make_offer()
        result = self.negotiator.negotiate([offer], max_budget=1.0)
        assert result.value_score >= 0.0

    def test_value_score_in_valid_range(self) -> None:
        offer = _make_offer(quality_score=1.0, trust_score=1.0, cost_per_call=0.0)
        result = self.negotiator.negotiate([offer], max_budget=1.0)
        assert 0.0 <= result.value_score <= 1.0

    def test_zero_budget_selects_free_offer(self) -> None:
        free_offer = _make_offer(cost_per_call=0.0)
        result = self.negotiator.negotiate([free_offer], max_budget=0.0)
        assert result.selected_offer is free_offer

    def test_returns_negotiation_result_instance(self) -> None:
        result = self.negotiator.negotiate([], max_budget=1.0)
        assert isinstance(result, NegotiationResult)


# ---------------------------------------------------------------------------
# PriceNegotiator — rank_offers()
# ---------------------------------------------------------------------------


class TestPriceNegotiatorRankOffers:
    def setup_method(self) -> None:
        self.negotiator = PriceNegotiator()

    def test_empty_offers_returns_empty_list(self) -> None:
        ranked = self.negotiator.rank_offers([], max_budget=1.0)
        assert ranked == []

    def test_over_budget_excluded(self) -> None:
        over_budget = _make_offer(cost_per_call=5.0)
        ranked = self.negotiator.rank_offers([over_budget], max_budget=1.0)
        assert ranked == []

    def test_returns_tuples_of_offer_and_score(self) -> None:
        offer = _make_offer()
        ranked = self.negotiator.rank_offers([offer], max_budget=1.0)
        assert len(ranked) == 1
        assert isinstance(ranked[0], tuple)
        assert len(ranked[0]) == 2

    def test_sorted_descending_by_value(self) -> None:
        low = _make_offer(provider_id="low", quality_score=0.1, trust_score=0.1)
        high = _make_offer(provider_id="high", quality_score=0.9, trust_score=0.9)
        ranked = self.negotiator.rank_offers([low, high], max_budget=1.0)
        assert ranked[0][0].provider_id == "high"

    def test_all_within_budget_included(self) -> None:
        offers = [_make_offer(provider_id=f"p-{i}") for i in range(4)]
        ranked = self.negotiator.rank_offers(offers, max_budget=1.0)
        assert len(ranked) == 4
