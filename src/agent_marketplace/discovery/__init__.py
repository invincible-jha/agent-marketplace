"""Discovery package for agent-marketplace.

Exports the search engine, embedding search, constraint filter,
fitness ranker, and high-level discovery client.
"""
from __future__ import annotations

from agent_marketplace.discovery.client import DiscoveryClient, DiscoveryResult
from agent_marketplace.discovery.embeddings import EmbeddingSearch
from agent_marketplace.discovery.filter import ConstraintFilter, FilterConstraints
from agent_marketplace.discovery.ranker import FitnessRanker, RankedCapability
from agent_marketplace.discovery.search import SearchEngine

__all__ = [
    "ConstraintFilter",
    "DiscoveryClient",
    "DiscoveryResult",
    "EmbeddingSearch",
    "FilterConstraints",
    "FitnessRanker",
    "RankedCapability",
    "SearchEngine",
]
