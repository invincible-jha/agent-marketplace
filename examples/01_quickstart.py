#!/usr/bin/env python3
"""Example: Quickstart

Demonstrates the minimal setup for agent-marketplace using the
Marketplace convenience class to register and discover capabilities.

Usage:
    python examples/01_quickstart.py

Requirements:
    pip install agent-marketplace
"""
from __future__ import annotations

import agent_marketplace
from agent_marketplace import (
    Marketplace,
    AgentCapability,
    CapabilityCategory,
    PricingModel,
)


def main() -> None:
    print(f"agent-marketplace version: {agent_marketplace.__version__}")

    # Step 1: Create a marketplace instance
    marketplace = Marketplace()
    print(f"Marketplace created: {marketplace}")

    # Step 2: Register a capability
    capability = AgentCapability(
        capability_id="text-summarisation-v1",
        name="Text Summarisation",
        description="Summarises long documents into concise bullet points.",
        category=CapabilityCategory.NLP,
        pricing_model=PricingModel.PER_TOKEN,
        price_per_unit=0.000002,
        provider_id="provider-alpha",
    )
    marketplace.register(capability)
    print(f"\nRegistered: '{capability.name}' (id={capability.capability_id})")

    # Step 3: Discover capabilities
    results = marketplace.discover(query="summarise documents")
    print(f"\nDiscovery results for 'summarise documents': {len(results)} match(es)")
    for result in results:
        print(f"  [{result.capability_id}] {result.name} — ${result.price_per_unit:.6f}/unit")

    print("\nQuickstart complete.")


if __name__ == "__main__":
    main()
