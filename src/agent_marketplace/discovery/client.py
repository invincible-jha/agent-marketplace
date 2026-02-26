"""High-level discovery client for agent-marketplace.

Provides the unified API that agent frameworks call to discover
capabilities.  Combines keyword search, embedding similarity, constraint
filtering, and trust scoring into a single ``discover()`` call.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from agent_marketplace.discovery.embeddings import EmbeddingSearch
from agent_marketplace.discovery.filter import ConstraintFilter, FilterConstraints
from agent_marketplace.discovery.ranker import FitnessRanker, RankedCapability
from agent_marketplace.discovery.search import SearchEngine
from agent_marketplace.registry.store import RegistryStore
from agent_marketplace.schema.capability import AgentCapability


@dataclass
class DiscoveryResult:
    """The outcome of a ``DiscoveryClient.discover()`` call.

    Attributes
    ----------
    ranked_capabilities:
        Capabilities in descending fitness order.
    total_found:
        Total number of capabilities matched (before pagination).
    query_keyword:
        The keyword used for this search.
    """

    ranked_capabilities: list[RankedCapability]
    total_found: int
    query_keyword: str = ""

    @property
    def capabilities(self) -> list[AgentCapability]:
        """Return just the capability objects (without scores)."""
        return [r.capability for r in self.ranked_capabilities]

    def best(self) -> AgentCapability | None:
        """Return the highest-ranked capability, or None if empty."""
        if not self.ranked_capabilities:
            return None
        return self.ranked_capabilities[0].capability


class DiscoveryClient:
    """Unified high-level API for discovering agent capabilities.

    This is the primary entry point for agent frameworks.  It orchestrates
    keyword search, TF-IDF semantic similarity, constraint filtering, and
    fitness ranking into a single ``discover()`` call.

    Parameters
    ----------
    store:
        The registry backend to query.
    use_embeddings:
        If True, supplement keyword search with TF-IDF similarity scoring.
    ranker:
        Optional custom ``FitnessRanker``.
    """

    def __init__(
        self,
        store: RegistryStore,
        use_embeddings: bool = True,
        ranker: FitnessRanker | None = None,
    ) -> None:
        self._store = store
        self._use_embeddings = use_embeddings
        self._ranker = ranker or FitnessRanker()
        self._search_engine = SearchEngine(store, self._ranker)
        self._embedding_search = EmbeddingSearch()
        self._index_built = False

    def discover(
        self,
        query: str,
        constraints: FilterConstraints | None = None,
        limit: int = 10,
        offset: int = 0,
    ) -> DiscoveryResult:
        """Discover capabilities matching *query*.

        The method:
        1. Performs keyword search via ``SearchEngine``.
        2. Optionally computes TF-IDF similarity scores.
        3. Applies ``ConstraintFilter`` if constraints are provided.
        4. Ranks by composite fitness.
        5. Returns paginated ``DiscoveryResult``.

        Parameters
        ----------
        query:
            Free-text description of what capability is needed.
        constraints:
            Optional hard threshold filters.
        limit:
            Maximum results per page.
        offset:
            Pagination offset.

        Returns
        -------
        DiscoveryResult
        """
        # Build/refresh embedding index on first call
        if self._use_embeddings and not self._index_built:
            self._build_embedding_index()

        # Step 1 — keyword candidates
        all_candidates = self._store.list_all()

        if constraints is not None:
            all_candidates = ConstraintFilter(constraints).apply(all_candidates)

        # Step 2 — relevance scores
        relevance_scores: dict[str, float] = {}
        if query.strip():
            relevance_scores.update(
                self._keyword_relevance_scores(all_candidates, query)
            )
            if self._use_embeddings and self._index_built:
                embedding_results = self._embedding_search.query(query, top_k=len(all_candidates))
                for cap_id, sim_score in embedding_results:
                    # Blend keyword and embedding scores
                    keyword_score = relevance_scores.get(cap_id, 0.0)
                    relevance_scores[cap_id] = 0.5 * keyword_score + 0.5 * sim_score

        # Step 3 — rank
        total = len(all_candidates)
        ranked = self._ranker.rank(all_candidates, relevance_scores=relevance_scores)

        # Step 4 — paginate
        paginated = ranked[offset : offset + limit] if limit > 0 else ranked[offset:]

        return DiscoveryResult(
            ranked_capabilities=paginated,
            total_found=total,
            query_keyword=query,
        )

    def refresh_index(self) -> None:
        """Rebuild the TF-IDF index from current registry contents."""
        self._build_embedding_index()

    def get_by_id(self, capability_id: str) -> AgentCapability:
        """Fetch a single capability by its unique identifier.

        Raises
        ------
        KeyError
            If the capability is not found.
        """
        return self._store.get(capability_id)

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _build_embedding_index(self) -> None:
        capabilities = self._store.list_all()
        documents = {
            cap.capability_id: self._capability_to_text(cap)
            for cap in capabilities
        }
        self._embedding_search.fit(documents)
        self._index_built = True

    @staticmethod
    def _capability_to_text(capability: AgentCapability) -> str:
        parts = [
            capability.name,
            capability.description,
            capability.category.value,
            " ".join(capability.tags),
            " ".join(capability.input_types),
            capability.output_type,
        ]
        return " ".join(p for p in parts if p)

    @staticmethod
    def _keyword_relevance_scores(
        capabilities: list[AgentCapability],
        keyword: str,
    ) -> dict[str, float]:
        keyword_lower = keyword.lower()
        scores: dict[str, float] = {}
        for cap in capabilities:
            score = 0.0
            if keyword_lower in cap.name.lower():
                score += 1.0
            occurrences = (cap.name + " " + cap.description).lower().count(keyword_lower)
            score += min(occurrences * 0.1, 0.5)
            if any(keyword_lower in tag.lower() for tag in cap.tags):
                score += 0.3
            scores[cap.capability_id] = min(score, 1.0)
        return scores
