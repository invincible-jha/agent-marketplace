#!/usr/bin/env python3
"""Example: Discovery and Search

Demonstrates the SearchEngine and DiscoveryClient for finding
capabilities by keyword, category, and constraint filters.

Usage:
    python examples/03_discovery_search.py

Requirements:
    pip install agent-marketplace
"""
from __future__ import annotations

import agent_marketplace
from agent_marketplace import (
    AgentCapability,
    CapabilityCategory,
    ConstraintFilter,
    DiscoveryClient,
    FilterConstraints,
    FitnessRanker,
    LatencyProfile,
    MemoryStore,
    PricingModel,
    QualityMetrics,
    SearchEngine,
    SearchQuery,
)


def populate_store(store: MemoryStore) -> None:
    """Add sample capabilities to the store."""
    caps = [
        AgentCapability(
            capability_id=f"cap-{i:03d}",
            name=name,
            description=desc,
            category=category,
            pricing_model=PricingModel.PER_TOKEN,
            price_per_unit=price,
            provider_id=f"provider-{i}",
            latency_profile=latency,
            quality_metrics=QualityMetrics(accuracy=accuracy, reliability=0.95),
        )
        for i, (name, desc, category, price, latency, accuracy) in enumerate([
            ("Fast Summariser", "Quick bullet-point summaries", CapabilityCategory.NLP, 0.000001, LatencyProfile.LOW, 0.88),
            ("Deep Analyser", "In-depth analysis with citations", CapabilityCategory.NLP, 0.000005, LatencyProfile.HIGH, 0.96),
            ("Code Generator", "Generate boilerplate code", CapabilityCategory.CODE, 0.000003, LatencyProfile.MEDIUM, 0.82),
            ("Image Describer", "Describe images in natural language", CapabilityCategory.VISION, 0.003, LatencyProfile.MEDIUM, 0.90),
            ("Data Extractor", "Extract structured data from PDFs", CapabilityCategory.DATA, 0.000004, LatencyProfile.LOW, 0.93),
        ])
    ]
    for cap in caps:
        store.add(cap)


def main() -> None:
    print(f"agent-marketplace version: {agent_marketplace.__version__}")

    # Step 1: Build the store
    store = MemoryStore()
    populate_store(store)
    print(f"Store populated with {store.count()} capabilities.")

    # Step 2: Keyword search
    engine = SearchEngine(store=store)
    query = SearchQuery(keywords=["summarise", "text"], max_results=5)
    results = engine.search(query)
    print(f"\nKeyword search 'summarise text': {len(results)} result(s)")
    for cap in results:
        print(f"  [{cap.capability_id}] {cap.name}")

    # Step 3: Discovery with constraints
    constraints = FilterConstraints(
        max_price_per_unit=0.000003,
        max_latency=LatencyProfile.MEDIUM,
        min_accuracy=0.85,
    )
    client = DiscoveryClient(store=store)
    discovery_result = client.discover(constraints=constraints)
    print(f"\nConstrained discovery: {len(discovery_result.matches)} match(es) "
          f"(price<0.000003, latency<=MEDIUM, accuracy>=0.85)")
    for match in discovery_result.matches:
        print(f"  [{match.name}] ${match.price_per_unit:.6f} | accuracy={match.quality_metrics.accuracy:.2f}")

    # Step 4: Ranked results
    ranker = FitnessRanker(weights={"accuracy": 0.6, "price": 0.3, "latency": 0.1})
    ranked = ranker.rank(discovery_result.matches)
    print(f"\nRanked results:")
    for ranked_cap in ranked[:3]:
        print(f"  [{ranked_cap.fitness_score:.3f}] {ranked_cap.capability.name}")


if __name__ == "__main__":
    main()
