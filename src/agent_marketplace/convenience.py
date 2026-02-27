"""Convenience API for agent-marketplace — 3-line quickstart.

Example
-------
::

    from agent_marketplace import Marketplace
    mp = Marketplace()
    results = mp.find("document summarization")

"""
from __future__ import annotations

from typing import Any


class Marketplace:
    """Zero-config agent capability marketplace for the 80% use case.

    Wraps MemoryStore + SearchEngine with no external dependencies.
    Register capabilities in-memory and search with plain text queries.

    Example
    -------
    ::

        from agent_marketplace import Marketplace
        mp = Marketplace()
        mp.register("doc-summarizer", "Summarizes long documents", ["summarization"])
        results = mp.find("document summarization")
        print(results[0].capability.name if results else "no results")
    """

    def __init__(self) -> None:
        from agent_marketplace.registry.memory_store import MemoryStore
        from agent_marketplace.discovery.search import SearchEngine

        self._store = MemoryStore()
        self._search = SearchEngine(self._store)

    def register(
        self,
        name: str,
        description: str,
        tags: list[str] | None = None,
        provider_id: str = "quickstart",
    ) -> Any:
        """Register a capability in the marketplace.

        Parameters
        ----------
        name:
            Unique capability name (e.g. ``"document-summarizer"``).
        description:
            Human-readable description of what this capability does.
        tags:
            Optional list of tag strings for search/filtering.
        provider_id:
            Provider identifier (default ``"quickstart"``).

        Returns
        -------
        AgentCapability
            The registered capability object.
        """
        from agent_marketplace.schema.capability import AgentCapability, CapabilityCategory
        from agent_marketplace.schema.provider import ProviderInfo

        capability = AgentCapability(
            name=name,
            version="1.0.0",
            description=description,
            category=CapabilityCategory.SPECIALIZED,
            provider=ProviderInfo(name=provider_id),
            tags=tags or [],
        )
        self._store.register(capability)
        return capability

    def find(self, query: str, limit: int = 10) -> list[Any]:
        """Search for capabilities matching a text query.

        Parameters
        ----------
        query:
            Free-text search query.
        limit:
            Maximum number of results to return (default 10).

        Returns
        -------
        list[RankedCapability]
            Ranked list of matching capabilities. Each has a
            ``.capability`` attribute and a ``.score`` float.

        Example
        -------
        ::

            mp = Marketplace()
            mp.register("summarizer", "Summarizes documents", ["nlp"])
            results = mp.find("document summary")
        """
        return self._search.search(keyword=query, limit=limit)

    @property
    def store(self) -> Any:
        """The underlying MemoryStore instance."""
        return self._store

    def __repr__(self) -> str:
        count = len(self._store.list_all())
        return f"Marketplace(capabilities={count})"
