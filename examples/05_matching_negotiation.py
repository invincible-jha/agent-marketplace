#!/usr/bin/env python3
"""Example: Capability Matching and Price Negotiation

Demonstrates the MatchingEngine for finding the best capability
for a request, and PriceNegotiator for automated price negotiation.

Usage:
    python examples/05_matching_negotiation.py

Requirements:
    pip install agent-marketplace
"""
from __future__ import annotations

import agent_marketplace
from agent_marketplace import (
    AgentCapability,
    CapabilityCategory,
    CapabilityRequest,
    LatencyProfile,
    MatchingEngine,
    MemoryStore,
    NegotiationResult,
    PriceNegotiator,
    PriceOffer,
    PricingModel,
    QualityMetrics,
)


def setup_store() -> MemoryStore:
    """Populate a store with capabilities for matching."""
    store = MemoryStore()
    for i, (name, price, accuracy, latency) in enumerate([
        ("Budget Summariser", 0.000001, 0.82, LatencyProfile.LOW),
        ("Standard Summariser", 0.000002, 0.90, LatencyProfile.LOW),
        ("Premium Summariser", 0.000005, 0.97, LatencyProfile.LOW),
    ]):
        store.add(AgentCapability(
            capability_id=f"summariser-{i}",
            name=name,
            description="Summarise documents into concise text.",
            category=CapabilityCategory.NLP,
            pricing_model=PricingModel.PER_TOKEN,
            price_per_unit=price,
            provider_id=f"provider-{i}",
            latency_profile=latency,
            quality_metrics=QualityMetrics(accuracy=accuracy, reliability=0.98),
        ))
    return store


def main() -> None:
    print(f"agent-marketplace version: {agent_marketplace.__version__}")

    store = setup_store()
    print(f"Store: {store.count()} capabilities")

    # Step 1: Create a capability request
    request = CapabilityRequest(
        requester_id="analysis-agent",
        description="Summarise quarterly reports into 3-bullet executive summaries.",
        category=CapabilityCategory.NLP,
        max_price_per_unit=0.000003,
        min_accuracy=0.88,
        max_latency=LatencyProfile.LOW,
    )
    print(f"\nRequest from '{request.requester_id}':")
    print(f"  Category: {request.category.value}")
    print(f"  Max price: ${request.max_price_per_unit:.6f}/unit")
    print(f"  Min accuracy: {request.min_accuracy:.2f}")

    # Step 2: Run matching engine
    engine = MatchingEngine(store=store)
    match_results = engine.match(request)
    print(f"\nMatching engine: {len(match_results)} match(es)")
    for match in match_results:
        print(f"  [{match.score:.3f}] {match.capability.name} "
              f"| ${match.capability.price_per_unit:.6f}/unit "
              f"| accuracy={match.capability.quality_metrics.accuracy:.2f}")

    # Step 3: Negotiate price with the best match
    if match_results:
        best_cap = match_results[0].capability
        negotiator = PriceNegotiator()
        initial_offer = PriceOffer(
            capability_id=best_cap.capability_id,
            offered_price=best_cap.price_per_unit * 0.85,  # 15% discount request
            requester_id=request.requester_id,
        )
        result: NegotiationResult = negotiator.negotiate(capability=best_cap, offer=initial_offer)
        print(f"\nPrice negotiation for '{best_cap.name}':")
        print(f"  Offered: ${initial_offer.offered_price:.6f}")
        print(f"  Final: ${result.final_price:.6f}")
        print(f"  Accepted: {result.accepted}")
        print(f"  Discount: {result.discount_pct:.1f}%")


if __name__ == "__main__":
    main()
