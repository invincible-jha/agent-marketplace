"""Semantic capability matching package for agent-marketplace.

Provides TF-IDF based semantic search over capability descriptions
without any external ML dependencies.

Classes
-------
TFIDFEmbedder         — bag-of-words / TF-IDF vectoriser
SemanticMatcher       — cosine similarity matching between query and capabilities
CapabilityIndex       — in-memory capability index with add/search/remove
"""
from __future__ import annotations

from agent_marketplace.semantic.embedder import (
    TFIDFEmbedder,
    TFIDFVector,
    EmbedderConfig,
)
from agent_marketplace.semantic.matcher import (
    MatchResult,
    SemanticMatcher,
    SemanticMatcherConfig,
)
from agent_marketplace.semantic.index import (
    CapabilityIndex,
    CapabilityIndexConfig,
    IndexedCapability,
    SearchResult,
)

__all__ = [
    # Embedder
    "TFIDFEmbedder",
    "TFIDFVector",
    "EmbedderConfig",
    # Matcher
    "MatchResult",
    "SemanticMatcher",
    "SemanticMatcherConfig",
    # Index
    "CapabilityIndex",
    "CapabilityIndexConfig",
    "IndexedCapability",
    "SearchResult",
]
