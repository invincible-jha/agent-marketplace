"""Registry package for agent-marketplace.

Exports the registry store abstraction and all built-in backends.
"""
from __future__ import annotations

from agent_marketplace.registry.memory_store import MemoryStore
from agent_marketplace.registry.namespace import Namespace, NamespaceManager
from agent_marketplace.registry.sqlite_store import SQLiteStore
from agent_marketplace.registry.store import RegistryStore, SearchQuery

__all__ = [
    "MemoryStore",
    "Namespace",
    "NamespaceManager",
    "RegistryStore",
    "SQLiteStore",
    "SearchQuery",
]
