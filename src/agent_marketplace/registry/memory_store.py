"""In-memory registry store for agent-marketplace.

Suitable for testing and development.  All data is lost when the
process exits.
"""
from __future__ import annotations

from agent_marketplace.registry.store import RegistryStore, SearchQuery
from agent_marketplace.schema.capability import AgentCapability


class MemoryStore(RegistryStore):
    """Thread-unsafe, in-memory capability store.

    Intended for testing and single-threaded development workflows.
    """

    def __init__(self) -> None:
        self._store: dict[str, AgentCapability] = {}

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def register(self, capability: AgentCapability) -> None:
        if capability.capability_id in self._store:
            raise ValueError(
                f"Capability {capability.capability_id!r} is already registered. "
                "Use update() to replace it."
            )
        self._store[capability.capability_id] = capability

    def update(self, capability: AgentCapability) -> None:
        if capability.capability_id not in self._store:
            raise KeyError(
                f"Capability {capability.capability_id!r} not found. "
                "Use register() to add a new capability."
            )
        self._store[capability.capability_id] = capability

    def get(self, capability_id: str) -> AgentCapability:
        try:
            return self._store[capability_id]
        except KeyError:
            raise KeyError(f"Capability {capability_id!r} not found in registry.") from None

    def delete(self, capability_id: str) -> None:
        if capability_id not in self._store:
            raise KeyError(f"Capability {capability_id!r} not found in registry.")
        del self._store[capability_id]

    # ------------------------------------------------------------------
    # Listing / search
    # ------------------------------------------------------------------

    def list_all(self) -> list[AgentCapability]:
        return list(self._store.values())

    def search(self, query: SearchQuery) -> list[AgentCapability]:
        results: list[AgentCapability] = []

        for capability in self._store.values():
            if not self._matches(capability, query):
                continue
            results.append(capability)

        # Apply pagination
        start = query.offset
        end = start + query.limit if query.limit > 0 else None
        return results[start:end]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _matches(capability: AgentCapability, query: SearchQuery) -> bool:
        """Return True if *capability* satisfies all query constraints."""
        # Keyword match (case-insensitive substring against name + description + tags)
        if query.keyword:
            keyword_lower = query.keyword.lower()
            searchable = (
                capability.name.lower()
                + " "
                + capability.description.lower()
                + " "
                + " ".join(t.lower() for t in capability.tags)
            )
            if keyword_lower not in searchable:
                return False

        # Category filter
        if query.category is not None and capability.category != query.category:
            return False

        # Tags filter (AND semantics)
        if query.tags:
            capability_tags = {t.lower() for t in capability.tags}
            for required_tag in query.tags:
                if required_tag.lower() not in capability_tags:
                    return False

        # Trust floor
        if capability.trust_level < query.min_trust:
            return False

        # Cost ceiling
        if capability.cost > query.max_cost:
            return False

        # Pricing model filter
        if query.pricing_model and capability.pricing_model.value != query.pricing_model:
            return False

        # Language filter
        if query.supported_language:
            lang_lower = query.supported_language.lower()
            if lang_lower not in [lang.lower() for lang in capability.supported_languages]:
                return False

        # Framework filter
        if query.supported_framework:
            fw_lower = query.supported_framework.lower()
            if fw_lower not in [fw.lower() for fw in capability.supported_frameworks]:
                return False

        return True
