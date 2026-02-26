"""Keyword and tag search engine for agent-marketplace.

Provides fast in-memory keyword matching combined with optional
constraint filtering and fitness ranking.
"""
from __future__ import annotations

from agent_marketplace.discovery.filter import ConstraintFilter, FilterConstraints
from agent_marketplace.discovery.ranker import FitnessRanker, RankedCapability
from agent_marketplace.registry.store import RegistryStore, SearchQuery
from agent_marketplace.schema.capability import AgentCapability


class SearchEngine:
    """High-level keyword and tag search engine.

    Combines the registry's built-in ``search()`` with constraint
    filtering and fitness ranking to return a relevance-ordered result list.

    Parameters
    ----------
    store:
        The registry backend to query.
    ranker:
        Optional custom ``FitnessRanker``.  A default-weight ranker is
        created when omitted.
    """

    def __init__(
        self,
        store: RegistryStore,
        ranker: FitnessRanker | None = None,
    ) -> None:
        self._store = store
        self._ranker = ranker or FitnessRanker()

    def search(
        self,
        keyword: str = "",
        tags: list[str] | None = None,
        constraints: FilterConstraints | None = None,
        limit: int = 20,
        offset: int = 0,
    ) -> list[RankedCapability]:
        """Search for capabilities matching *keyword* and *tags*.

        Parameters
        ----------
        keyword:
            Free-text keyword.  Matched as a case-insensitive substring
            against capability name, description, and tags.
        tags:
            Additional tag filters (AND semantics).
        constraints:
            Optional hard constraint filter applied after keyword search.
        limit:
            Maximum number of results.
        offset:
            Number of results to skip for pagination.

        Returns
        -------
        list[RankedCapability]
            Results in descending fitness order.
        """
        query = SearchQuery(
            keyword=keyword,
            tags=tags or [],
            limit=0,  # Fetch all; apply pagination after ranking
            offset=0,
        )
        candidates = self._store.search(query)

        if constraints is not None:
            constraint_filter = ConstraintFilter(constraints)
            candidates = constraint_filter.apply(candidates)

        ranked = self._ranker.rank(
            candidates,
            relevance_scores=self._compute_relevance_scores(candidates, keyword),
        )

        return ranked[offset : offset + limit] if limit > 0 else ranked[offset:]

    def search_by_capability_type(
        self,
        category_value: str,
        constraints: FilterConstraints | None = None,
        limit: int = 20,
    ) -> list[RankedCapability]:
        """Search for capabilities by category string.

        Parameters
        ----------
        category_value:
            A ``CapabilityCategory`` string value (e.g. ``"analysis"``).
        constraints:
            Optional additional filters.
        limit:
            Maximum result count.
        """
        from agent_marketplace.schema.capability import CapabilityCategory

        try:
            category = CapabilityCategory(category_value)
        except ValueError:
            return []

        query = SearchQuery(category=category, limit=0)
        candidates = self._store.search(query)

        if constraints is not None:
            candidates = ConstraintFilter(constraints).apply(candidates)

        return self._ranker.rank(candidates)[:limit]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_relevance_scores(
        capabilities: list[AgentCapability],
        keyword: str,
    ) -> dict[str, float]:
        """Compute a simple term-frequency relevance score for each capability."""
        if not keyword or not capabilities:
            return {}

        keyword_lower = keyword.lower()
        scores: dict[str, float] = {}
        for cap in capabilities:
            score = 0.0
            text = cap.name.lower() + " " + cap.description.lower()
            tag_text = " ".join(cap.tags).lower()

            # Exact name match is highest signal
            if keyword_lower in cap.name.lower():
                score += 1.0
            # Description match
            occurrences = text.count(keyword_lower)
            score += min(occurrences * 0.1, 0.5)
            # Tag match
            if keyword_lower in tag_text:
                score += 0.3

            scores[cap.capability_id] = min(score, 1.0)

        return scores
