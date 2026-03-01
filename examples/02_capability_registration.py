#!/usr/bin/env python3
"""Example: Capability Registration

Demonstrates registering multiple capabilities with metadata,
quality metrics, and provider information.

Usage:
    python examples/02_capability_registration.py

Requirements:
    pip install agent-marketplace
"""
from __future__ import annotations

import agent_marketplace
from agent_marketplace import (
    AgentCapability,
    CapabilityCategory,
    LatencyProfile,
    PricingModel,
    ProviderInfo,
    QualityMetrics,
    MemoryStore,
    SchemaValidator,
)


def build_capabilities() -> list[AgentCapability]:
    """Build a set of sample capabilities."""
    return [
        AgentCapability(
            capability_id="nlp-summarise-v2",
            name="Document Summarisation",
            description="Summarise long-form documents using LLMs.",
            category=CapabilityCategory.NLP,
            pricing_model=PricingModel.PER_TOKEN,
            price_per_unit=0.000002,
            provider_id="nlp-labs",
            latency_profile=LatencyProfile.LOW,
            quality_metrics=QualityMetrics(accuracy=0.92, reliability=0.99),
        ),
        AgentCapability(
            capability_id="vision-classify-v1",
            name="Image Classification",
            description="Classify images into predefined categories.",
            category=CapabilityCategory.VISION,
            pricing_model=PricingModel.PER_REQUEST,
            price_per_unit=0.005,
            provider_id="vision-ai",
            latency_profile=LatencyProfile.MEDIUM,
            quality_metrics=QualityMetrics(accuracy=0.88, reliability=0.97),
        ),
        AgentCapability(
            capability_id="code-review-v1",
            name="Code Review",
            description="Automated code review with security and quality checks.",
            category=CapabilityCategory.CODE,
            pricing_model=PricingModel.PER_REQUEST,
            price_per_unit=0.02,
            provider_id="devtools-ai",
            latency_profile=LatencyProfile.HIGH,
            quality_metrics=QualityMetrics(accuracy=0.85, reliability=0.95),
        ),
    ]


def main() -> None:
    print(f"agent-marketplace version: {agent_marketplace.__version__}")

    # Step 1: Create registry store
    store = MemoryStore()
    validator = SchemaValidator()

    # Step 2: Register capabilities
    capabilities = build_capabilities()
    for cap in capabilities:
        validation = validator.validate(cap)
        if validation.is_valid:
            store.add(cap)
            print(f"  Registered: [{cap.category.value}] {cap.name} | ${cap.price_per_unit:.6f}/unit")
        else:
            print(f"  Invalid: {cap.name} — {validation.errors}")

    print(f"\nRegistry: {store.count()} capabilities")

    # Step 3: Retrieve by category
    for category in [CapabilityCategory.NLP, CapabilityCategory.CODE]:
        caps = store.filter_by_category(category)
        print(f"  {category.value}: {len(caps)} capability(ies)")


if __name__ == "__main__":
    main()
