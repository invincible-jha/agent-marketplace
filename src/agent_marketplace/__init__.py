"""agent-marketplace — Agent capability registry, discovery, and semantic matching.

Public API
----------
The stable public surface is everything exported from this module.
Anything inside submodules not re-exported here is considered private
and may change without notice.

Sub-packages
------------
- ``schema``     — ``AgentCapability``, ``ProviderInfo``, schema validators.
- ``registry``   — ``RegistryStore`` and backend implementations.
- ``discovery``  — ``DiscoveryClient``, ``SearchEngine``, ranker, filters.
- ``trust``      — ``TrustScorer``, ``ReviewStore``, ``ReputationTracker``.
- ``matching``   — ``MatchingEngine``, ``CapabilityRequest``, ``PriceNegotiator``.
- ``adapters``   — ``OpenAPIAdapter``, ``AsyncAPIAdapter``, ``MCPAdapter``.
- ``server``     — ``MarketplaceAPI``, ``HealthEndpoint``.
- ``analytics``  — ``UsageTracker``, ``MarketplaceReporter``.
- ``plugins``    — ``PluginRegistry`` for third-party extensions.

Example
-------
>>> import agent_marketplace
>>> agent_marketplace.__version__
'0.1.0'
"""
from __future__ import annotations

__version__: str = "0.1.0"

from agent_marketplace.convenience import Marketplace

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------
from agent_marketplace.schema.capability import (
    AgentCapability,
    CapabilityCategory,
    LatencyProfile,
    PricingModel,
    QualityMetrics,
)
from agent_marketplace.schema.provider import ProviderInfo
from agent_marketplace.schema.validator import (
    SchemaValidator,
    ValidationError,
    ValidationResult,
)

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
from agent_marketplace.registry.memory_store import MemoryStore
from agent_marketplace.registry.store import RegistryStore, SearchQuery

# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------
from agent_marketplace.discovery.client import DiscoveryClient, DiscoveryResult
from agent_marketplace.discovery.filter import ConstraintFilter, FilterConstraints
from agent_marketplace.discovery.ranker import FitnessRanker, RankedCapability
from agent_marketplace.discovery.search import SearchEngine

# ---------------------------------------------------------------------------
# Trust
# ---------------------------------------------------------------------------
from agent_marketplace.trust.reputation import ReputationTracker
from agent_marketplace.trust.reviews import Review, ReviewStore
from agent_marketplace.trust.scorer import ProviderTrustData, TrustScorer

# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------
from agent_marketplace.matching.engine import MatchResult, MatchingEngine
from agent_marketplace.matching.negotiator import (
    NegotiationResult,
    PriceNegotiator,
    PriceOffer,
)
from agent_marketplace.matching.request import CapabilityRequest

# ---------------------------------------------------------------------------
# Adapters
# ---------------------------------------------------------------------------
from agent_marketplace.adapters.asyncapi import AsyncAPIAdapter
from agent_marketplace.adapters.mcp_adapter import MCPAdapter
from agent_marketplace.adapters.openapi import OpenAPIAdapter

# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------
from agent_marketplace.server.api import MarketplaceAPI
from agent_marketplace.server.health import HealthEndpoint

# ---------------------------------------------------------------------------
# Analytics
# ---------------------------------------------------------------------------
from agent_marketplace.analytics.reporter import MarketplaceReporter
from agent_marketplace.analytics.usage import UsageRecord, UsageTracker

# ---------------------------------------------------------------------------
# Plugins
# ---------------------------------------------------------------------------
from agent_marketplace.plugins.registry import PluginRegistry

# ---------------------------------------------------------------------------
# Catalog
# ---------------------------------------------------------------------------
from agent_marketplace.catalog import (
    ToolAlreadyRegisteredError,
    ToolCatalog,
    ToolCatalogError,
    ToolEntry,
    ToolNotFoundError,
)

# ---------------------------------------------------------------------------
# Semantic — embedding search (optional extras guarded inside the modules)
# ---------------------------------------------------------------------------
from agent_marketplace.semantic.embedding_backend import (
    EmbeddingBackend,
    SentenceTransformerEmbedder,
)
from agent_marketplace.semantic.vector_index import (
    InMemoryCosineIndex,
    SearchHit,
)

__all__ = [
    "__version__",
    "Marketplace",
    # Schema
    "AgentCapability",
    "CapabilityCategory",
    "LatencyProfile",
    "PricingModel",
    "ProviderInfo",
    "QualityMetrics",
    "SchemaValidator",
    "ValidationError",
    "ValidationResult",
    # Registry
    "MemoryStore",
    "RegistryStore",
    "SearchQuery",
    # Discovery
    "ConstraintFilter",
    "DiscoveryClient",
    "DiscoveryResult",
    "FilterConstraints",
    "FitnessRanker",
    "RankedCapability",
    "SearchEngine",
    # Trust
    "ProviderTrustData",
    "ReputationTracker",
    "Review",
    "ReviewStore",
    "TrustScorer",
    # Matching
    "CapabilityRequest",
    "MatchResult",
    "MatchingEngine",
    "NegotiationResult",
    "PriceNegotiator",
    "PriceOffer",
    # Adapters
    "AsyncAPIAdapter",
    "MCPAdapter",
    "OpenAPIAdapter",
    # Server
    "HealthEndpoint",
    "MarketplaceAPI",
    # Analytics
    "MarketplaceReporter",
    "UsageRecord",
    "UsageTracker",
    # Plugins
    "PluginRegistry",
    # Catalog
    "ToolAlreadyRegisteredError",
    "ToolCatalog",
    "ToolCatalogError",
    "ToolEntry",
    "ToolNotFoundError",
    # Semantic embedding search
    "EmbeddingBackend",
    "SentenceTransformerEmbedder",
    "InMemoryCosineIndex",
    "SearchHit",
]
