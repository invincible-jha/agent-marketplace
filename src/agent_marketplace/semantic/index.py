"""Capability index — in-memory capability index with add, search, remove.

CapabilityIndex provides a self-managing index over capability descriptions.
When capabilities are added or removed, the index automatically re-fits the
TF-IDF embedder so that all subsequent searches are up-to-date.

This is intentionally a simple in-memory implementation.  For persistence,
capabilities can be exported to a dict and re-imported.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from agent_marketplace.semantic.embedder import EmbedderConfig, TFIDFEmbedder
from agent_marketplace.semantic.matcher import (
    MatchResult,
    SemanticMatcher,
    SemanticMatcherConfig,
)


# ---------------------------------------------------------------------------
# IndexedCapability
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class IndexedCapability:
    """A capability registered in the semantic index.

    Attributes
    ----------
    capability_id:
        Unique identifier for this capability.
    name:
        Human-readable capability name.
    description:
        Full prose description used for semantic matching.
    tags:
        Additional keyword tags that supplement the description.
    metadata:
        Arbitrary extra data.
    """

    capability_id: str
    name: str
    description: str
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def index_text(self) -> str:
        """Combined text for TF-IDF indexing (name + description + tags)."""
        tag_str = " ".join(self.tags)
        return f"{self.name} {self.description} {tag_str}".strip()

    def to_dict(self) -> dict[str, object]:
        """Serialise to a plain dict."""
        return {
            "capability_id": self.capability_id,
            "name": self.name,
            "description": self.description,
            "tags": list(self.tags),
            "metadata": dict(self.metadata),
        }


# ---------------------------------------------------------------------------
# SearchResult
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SearchResult:
    """A search result from CapabilityIndex.

    Attributes
    ----------
    capability:
        The matched capability.
    similarity:
        Cosine similarity score in [0.0, 1.0].
    rank:
        Zero-based rank within the search results.
    """

    capability: IndexedCapability
    similarity: float
    rank: int

    def to_dict(self) -> dict[str, object]:
        """Serialise to a plain dict."""
        return {
            "capability": self.capability.to_dict(),
            "similarity": self.similarity,
            "rank": self.rank,
        }


# ---------------------------------------------------------------------------
# CapabilityIndexConfig
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CapabilityIndexConfig:
    """Configuration for CapabilityIndex.

    Attributes
    ----------
    embedder_config:
        Configuration for the underlying TF-IDF embedder.
    matcher_config:
        Configuration for the semantic matcher.
    auto_refit:
        When True (default), the index automatically re-fits the embedder
        whenever capabilities are added or removed.
    """

    embedder_config: EmbedderConfig = field(default_factory=EmbedderConfig)
    matcher_config: SemanticMatcherConfig = field(
        default_factory=SemanticMatcherConfig
    )
    auto_refit: bool = True


# ---------------------------------------------------------------------------
# CapabilityIndex
# ---------------------------------------------------------------------------


class CapabilityIndex:
    """In-memory semantic capability index with add, search, and remove.

    Parameters
    ----------
    config:
        Index configuration.  Defaults to standard settings.

    Example
    -------
    >>> index = CapabilityIndex()
    >>> index.add(IndexedCapability(
    ...     capability_id="c1",
    ...     name="PDF Extractor",
    ...     description="Extract tables and text from PDF documents",
    ... ))
    >>> results = index.search("extract data from PDF")
    >>> results[0].capability.capability_id == "c1"
    True
    """

    def __init__(self, config: CapabilityIndexConfig | None = None) -> None:
        self._config = config if config is not None else CapabilityIndexConfig()
        self._capabilities: dict[str, IndexedCapability] = {}
        self._embedder = TFIDFEmbedder(self._config.embedder_config)
        self._matcher = SemanticMatcher(self._embedder, self._config.matcher_config)

    @property
    def size(self) -> int:
        """Number of capabilities currently in the index."""
        return len(self._capabilities)

    @property
    def is_empty(self) -> bool:
        """True when no capabilities are indexed."""
        return len(self._capabilities) == 0

    def add(self, capability: IndexedCapability) -> None:
        """Add or replace a capability in the index.

        Parameters
        ----------
        capability:
            The capability to index.
        """
        self._capabilities[capability.capability_id] = capability
        if self._config.auto_refit:
            self._refit()

    def add_many(self, capabilities: list[IndexedCapability]) -> None:
        """Add multiple capabilities and refit once.

        Parameters
        ----------
        capabilities:
            List of capabilities to add.
        """
        for cap in capabilities:
            self._capabilities[cap.capability_id] = cap
        self._refit()

    def remove(self, capability_id: str) -> None:
        """Remove a capability from the index.

        Parameters
        ----------
        capability_id:
            The ID of the capability to remove.

        Raises
        ------
        KeyError
            If no capability with ``capability_id`` exists.
        """
        if capability_id not in self._capabilities:
            raise KeyError(
                f"Capability {capability_id!r} not found in index."
            )
        del self._capabilities[capability_id]
        if self._config.auto_refit:
            self._refit()

    def get(self, capability_id: str) -> IndexedCapability | None:
        """Retrieve a capability by ID.

        Parameters
        ----------
        capability_id:
            The ID to look up.

        Returns
        -------
        IndexedCapability | None
            The capability, or None if not found.
        """
        return self._capabilities.get(capability_id)

    def search(
        self,
        query: str,
        *,
        top_k: int | None = None,
        min_similarity: float | None = None,
    ) -> list[SearchResult]:
        """Search for capabilities matching the query.

        Parameters
        ----------
        query:
            Free-text query describing the desired capability.
        top_k:
            Maximum number of results to return.
        min_similarity:
            Minimum cosine similarity threshold.

        Returns
        -------
        list[SearchResult]
            Ranked search results (best match first).
        """
        if self.is_empty or not self._embedder.is_fitted:
            return []

        match_results: list[MatchResult] = self._matcher.match(
            query, top_k=top_k, min_similarity=min_similarity
        )

        results: list[SearchResult] = []
        for mr in match_results:
            cap = self._capabilities.get(mr.capability_id)
            if cap is not None:
                results.append(
                    SearchResult(
                        capability=cap,
                        similarity=mr.similarity,
                        rank=mr.rank,
                    )
                )
        return results

    def all_capabilities(self) -> list[IndexedCapability]:
        """Return all indexed capabilities.

        Returns
        -------
        list[IndexedCapability]
            All capabilities in the index (unordered).
        """
        return list(self._capabilities.values())

    def clear(self) -> None:
        """Remove all capabilities from the index and reset the embedder."""
        self._capabilities.clear()
        self._embedder = TFIDFEmbedder(self._config.embedder_config)
        self._matcher = SemanticMatcher(self._embedder, self._config.matcher_config)

    def _refit(self) -> None:
        """Re-fit the TF-IDF embedder with the current capability corpus."""
        corpus = {
            cap_id: cap.index_text
            for cap_id, cap in self._capabilities.items()
        }
        self._embedder.fit(corpus)
        self._matcher = SemanticMatcher(self._embedder, self._config.matcher_config)
