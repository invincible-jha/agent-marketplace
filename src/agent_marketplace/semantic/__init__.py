"""Semantic capability matching package for agent-marketplace.

Provides TF-IDF based semantic search over capability descriptions
without any external ML dependencies, plus an optional dense-vector
embedding path when ``sentence-transformers`` (or any custom
EmbeddingBackend) is installed.

Classes
-------
TFIDFEmbedder             — bag-of-words / TF-IDF vectoriser
SemanticMatcher           — cosine similarity matching between query and capabilities
CapabilityIndex           — in-memory capability index with add/search/remove
EmbeddingBackend          — abstract base for pluggable embedding backends
SentenceTransformerEmbedder — optional sentence-transformers backend
InMemoryCosineIndex       — pure-Python dense-vector cosine search index
SearchHit                 — result type from InMemoryCosineIndex.search
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
from agent_marketplace.semantic.embedding_backend import (
    EmbeddingBackend,
    SentenceTransformerEmbedder,
    cosine_similarity,
)
from agent_marketplace.semantic.vector_index import (
    InMemoryCosineIndex,
    SearchHit,
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
    # Embedding backend
    "EmbeddingBackend",
    "SentenceTransformerEmbedder",
    "cosine_similarity",
    # Vector index
    "InMemoryCosineIndex",
    "SearchHit",
]
