"""Shared pytest fixtures for agent-marketplace unit tests.

All fixtures are deterministic and side-effect-free.  Heavy objects
(stores, engines) are constructed fresh per test via function scope to
prevent inter-test contamination.
"""
from __future__ import annotations

import pytest

from agent_marketplace.schema.capability import (
    AgentCapability,
    CapabilityCategory,
    LatencyProfile,
    PricingModel,
    QualityMetrics,
)
from agent_marketplace.schema.provider import ProviderInfo


# ---------------------------------------------------------------------------
# Provider fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def acme_provider() -> ProviderInfo:
    """A verified provider representing 'Acme Corp'."""
    return ProviderInfo(
        name="Acme Corp",
        organization="Acme Corporation",
        contact_email="support@acme.example.com",
        website="https://acme.example.com",
        github_handle="acme-corp",
        verified=True,
    )


@pytest.fixture()
def unknown_provider() -> ProviderInfo:
    """An unverified provider with minimal fields."""
    return ProviderInfo(name="Unknown Dev")


# ---------------------------------------------------------------------------
# Capability fixtures
# ---------------------------------------------------------------------------


def make_capability(
    name: str = "pdf-extractor",
    version: str = "1.0.0",
    description: str = "Extracts structured data from PDF documents.",
    category: CapabilityCategory = CapabilityCategory.EXTRACTION,
    tags: list[str] | None = None,
    input_types: list[str] | None = None,
    output_type: str = "application/json",
    pricing_model: PricingModel = PricingModel.FREE,
    cost: float = 0.0,
    trust_level: float = 0.0,
    provider_name: str = "Acme Corp",
    supported_languages: list[str] | None = None,
    supported_frameworks: list[str] | None = None,
) -> AgentCapability:
    """Factory function to build an ``AgentCapability`` with sensible defaults."""
    return AgentCapability(
        name=name,
        version=version,
        description=description,
        category=category,
        tags=tags or ["pdf", "extraction", "ocr"],
        input_types=input_types or ["application/pdf"],
        output_type=output_type,
        pricing_model=pricing_model,
        cost=cost,
        trust_level=trust_level,
        supported_languages=supported_languages or ["en"],
        supported_frameworks=supported_frameworks or ["langchain"],
        provider=ProviderInfo(name=provider_name, verified=True),
    )


@pytest.fixture()
def pdf_extractor_capability() -> AgentCapability:
    """A fully populated extraction capability."""
    return make_capability()


@pytest.fixture()
def image_generation_capability() -> AgentCapability:
    """A generation capability with per-call pricing."""
    return make_capability(
        name="image-generator",
        version="2.1.0",
        description="Generates high-resolution images from text prompts.",
        category=CapabilityCategory.GENERATION,
        tags=["image", "generation", "diffusion"],
        input_types=["text/plain"],
        output_type="image/png",
        pricing_model=PricingModel.PER_CALL,
        cost=0.005,
        trust_level=0.75,
        provider_name="Creative AI",
    )


@pytest.fixture()
def analysis_capability() -> AgentCapability:
    """A data analysis capability with quality metrics."""
    cap = make_capability(
        name="data-analyser",
        version="3.0.0",
        description="Performs statistical analysis on structured datasets.",
        category=CapabilityCategory.ANALYSIS,
        tags=["analysis", "statistics", "data"],
        input_types=["application/json", "text/csv"],
        output_type="application/json",
        pricing_model=PricingModel.PER_TOKEN,
        cost=0.002,
        trust_level=0.90,
        provider_name="DataCo",
    )
    # Attach quality metrics after construction
    return cap.model_copy(
        update={
            "quality_metrics": QualityMetrics(
                metrics={"accuracy": 0.94, "f1": 0.91},
                benchmark_source="internal-eval-2025",
                benchmark_date="2025-01-15",
                verified=True,
            )
        }
    )


@pytest.fixture()
def three_capabilities(
    pdf_extractor_capability: AgentCapability,
    image_generation_capability: AgentCapability,
    analysis_capability: AgentCapability,
) -> list[AgentCapability]:
    """Convenience fixture returning all three base capabilities."""
    return [
        pdf_extractor_capability,
        image_generation_capability,
        analysis_capability,
    ]
