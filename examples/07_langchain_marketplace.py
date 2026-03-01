#!/usr/bin/env python3
"""Example: LangChain Marketplace Integration

Demonstrates using the marketplace to discover capabilities and
dynamically load them as LangChain tools.

Usage:
    python examples/07_langchain_marketplace.py

Requirements:
    pip install agent-marketplace langchain
"""
from __future__ import annotations

try:
    from langchain.tools import StructuredTool
    _LANGCHAIN_AVAILABLE = True
except ImportError:
    _LANGCHAIN_AVAILABLE = False

import agent_marketplace
from agent_marketplace import (
    AgentCapability,
    CapabilityCategory,
    CapabilityRequest,
    LatencyProfile,
    MatchingEngine,
    MemoryStore,
    PricingModel,
    QualityMetrics,
    UsageTracker,
    UsageRecord,
)


def capability_to_tool(cap: AgentCapability) -> "object":
    """Convert a marketplace capability to a LangChain tool."""
    if not _LANGCHAIN_AVAILABLE:
        return None

    def run(input_text: str) -> str:
        return f"[{cap.name}] processed: {input_text[:50]}"

    return StructuredTool.from_function(
        func=run,
        name=cap.capability_id.replace("-", "_"),
        description=cap.description,
    )


def main() -> None:
    print(f"agent-marketplace version: {agent_marketplace.__version__}")

    if not _LANGCHAIN_AVAILABLE:
        print("LangChain not installed — demonstrating discovery only.")
        print("Install with: pip install langchain")

    # Step 1: Set up marketplace
    store = MemoryStore()
    for i, (cap_id, name, desc, accuracy) in enumerate([
        ("nlp-summarise-v2", "Text Summariser", "Summarise long documents.", 0.92),
        ("code-review-v1", "Code Reviewer", "Review code for quality issues.", 0.85),
        ("data-extract-v1", "Data Extractor", "Extract structured data from text.", 0.90),
    ]):
        store.add(AgentCapability(
            capability_id=cap_id,
            name=name,
            description=desc,
            category=CapabilityCategory.NLP,
            pricing_model=PricingModel.PER_TOKEN,
            price_per_unit=0.000002,
            provider_id=f"provider-{i}",
            latency_profile=LatencyProfile.LOW,
            quality_metrics=QualityMetrics(accuracy=accuracy, reliability=0.97),
        ))

    # Step 2: Discover capabilities matching a request
    request = CapabilityRequest(
        requester_id="langchain-orchestrator",
        description="Summarise and extract data from documents",
        category=CapabilityCategory.NLP,
        max_price_per_unit=0.000005,
        min_accuracy=0.80,
    )
    engine = MatchingEngine(store=store)
    matches = engine.match(request)
    print(f"\nDiscovered {len(matches)} matching capabilities.")

    # Step 3: Convert to LangChain tools (if available)
    if _LANGCHAIN_AVAILABLE:
        lc_tools = [capability_to_tool(m.capability) for m in matches]
        lc_tools = [t for t in lc_tools if t is not None]
        print(f"Converted {len(lc_tools)} capabilities to LangChain tools.")
        for tool in lc_tools:
            result = tool.run("summarise the quarterly earnings report")
            print(f"  [{tool.name}] -> {result[:60]}")

    # Step 4: Track usage
    tracker = UsageTracker()
    for match in matches:
        record = UsageRecord(
            capability_id=match.capability.capability_id,
            requester_id=request.requester_id,
            tokens_consumed=500,
            cost_usd=500 * match.capability.price_per_unit,
        )
        tracker.record(record)

    summary = tracker.summary(requester_id=request.requester_id)
    print(f"\nUsage summary for '{request.requester_id}':")
    print(f"  Total calls: {summary.total_calls}")
    print(f"  Total cost: ${summary.total_cost_usd:.6f}")


if __name__ == "__main__":
    main()
