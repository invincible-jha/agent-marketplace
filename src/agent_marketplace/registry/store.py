"""Abstract registry store interface for agent-marketplace.

All backend implementations (SQLite, Redis, in-memory) must satisfy
this interface.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

from agent_marketplace.schema.capability import AgentCapability, CapabilityCategory


# ---------------------------------------------------------------------------
# Query model
# ---------------------------------------------------------------------------


@dataclass
class SearchQuery:
    """Parameters for a capability search operation.

    Attributes
    ----------
    keyword:
        Free-text keyword(s) matched against name, description, and tags.
    category:
        Optional category filter.
    tags:
        Tags that must all be present on the result (AND semantics).
    min_trust:
        Minimum trust_level score (inclusive).
    max_cost:
        Maximum cost (inclusive).
    pricing_model:
        Optional pricing model filter.
    supported_language:
        Optional language code filter.
    supported_framework:
        Optional framework name filter.
    limit:
        Maximum number of results to return (0 = unlimited).
    offset:
        Number of results to skip for pagination.
    """

    keyword: str = ""
    category: Optional[CapabilityCategory] = None
    tags: list[str] = field(default_factory=list)
    min_trust: float = 0.0
    max_cost: float = float("inf")
    pricing_model: Optional[str] = None
    supported_language: str = ""
    supported_framework: str = ""
    limit: int = 50
    offset: int = 0


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class RegistryStore(ABC):
    """Abstract base class for capability registry storage backends.

    All methods are synchronous to avoid mandating an async runtime.
    Implementors are free to add async wrappers on top.
    """

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    @abstractmethod
    def register(self, capability: AgentCapability) -> None:
        """Persist a new capability.

        Parameters
        ----------
        capability:
            The capability to store.  ``capability.capability_id`` must be
            unique within this store.

        Raises
        ------
        ValueError
            If a capability with the same ``capability_id`` already exists.
        """

    @abstractmethod
    def update(self, capability: AgentCapability) -> None:
        """Replace an existing capability record.

        Parameters
        ----------
        capability:
            The updated capability.  Must already exist in the store.

        Raises
        ------
        KeyError
            If no capability with this ``capability_id`` exists.
        """

    @abstractmethod
    def get(self, capability_id: str) -> AgentCapability:
        """Retrieve a single capability by its unique identifier.

        Parameters
        ----------
        capability_id:
            The auto-generated capability identifier.

        Returns
        -------
        AgentCapability
            The stored capability.

        Raises
        ------
        KeyError
            If the capability does not exist.
        """

    @abstractmethod
    def delete(self, capability_id: str) -> None:
        """Remove a capability from the store.

        Parameters
        ----------
        capability_id:
            Identifier of the capability to remove.

        Raises
        ------
        KeyError
            If the capability does not exist.
        """

    # ------------------------------------------------------------------
    # Listing / search
    # ------------------------------------------------------------------

    @abstractmethod
    def list_all(self) -> list[AgentCapability]:
        """Return every capability in the store."""

    @abstractmethod
    def search(self, query: SearchQuery) -> list[AgentCapability]:
        """Return capabilities matching the given query.

        Parameters
        ----------
        query:
            A ``SearchQuery`` describing the filtering and pagination params.

        Returns
        -------
        list[AgentCapability]
            Matching capabilities, respecting *limit* and *offset*.
        """

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def count(self) -> int:
        """Return the total number of capabilities in the store."""
        return len(self.list_all())

    def exists(self, capability_id: str) -> bool:
        """Return True if a capability with this id exists."""
        try:
            self.get(capability_id)
            return True
        except KeyError:
            return False
