"""SQLite-backed registry store for agent-marketplace.

Provides durable persistence with no external service dependencies.
Uses Python's built-in ``sqlite3`` module.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from agent_marketplace.registry.store import RegistryStore, SearchQuery
from agent_marketplace.schema.capability import AgentCapability


_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS capabilities (
    capability_id TEXT PRIMARY KEY,
    name          TEXT NOT NULL,
    version       TEXT NOT NULL,
    description   TEXT NOT NULL,
    category      TEXT NOT NULL,
    tags          TEXT NOT NULL,
    trust_level   REAL NOT NULL DEFAULT 0.0,
    cost          REAL NOT NULL DEFAULT 0.0,
    pricing_model TEXT NOT NULL,
    provider_name TEXT NOT NULL,
    data          TEXT NOT NULL
)
"""

_INDEX_NAME_SQL = "CREATE INDEX IF NOT EXISTS idx_name ON capabilities(name)"
_INDEX_CATEGORY_SQL = "CREATE INDEX IF NOT EXISTS idx_category ON capabilities(category)"
_INDEX_TRUST_SQL = "CREATE INDEX IF NOT EXISTS idx_trust ON capabilities(trust_level)"


class SQLiteStore(RegistryStore):
    """Persistent registry store backed by SQLite.

    Parameters
    ----------
    db_path:
        File system path for the SQLite database file.
        Use ``":memory:"`` for an in-memory SQLite database (testing).
    """

    def __init__(self, db_path: str | Path = ":memory:") -> None:
        self._db_path = str(db_path)
        self._connection = sqlite3.connect(self._db_path, check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        cursor = self._connection.cursor()
        cursor.execute(_CREATE_TABLE_SQL)
        cursor.execute(_INDEX_NAME_SQL)
        cursor.execute(_INDEX_CATEGORY_SQL)
        cursor.execute(_INDEX_TRUST_SQL)
        self._connection.commit()

    def close(self) -> None:
        """Close the underlying SQLite connection."""
        self._connection.close()

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def register(self, capability: AgentCapability) -> None:
        if self.exists(capability.capability_id):
            raise ValueError(
                f"Capability {capability.capability_id!r} is already registered."
            )
        self._upsert(capability)

    def update(self, capability: AgentCapability) -> None:
        if not self.exists(capability.capability_id):
            raise KeyError(
                f"Capability {capability.capability_id!r} not found. "
                "Use register() to add a new capability."
            )
        self._upsert(capability)

    def get(self, capability_id: str) -> AgentCapability:
        cursor = self._connection.cursor()
        cursor.execute(
            "SELECT data FROM capabilities WHERE capability_id = ?", (capability_id,)
        )
        row = cursor.fetchone()
        if row is None:
            raise KeyError(f"Capability {capability_id!r} not found in registry.")
        return AgentCapability.model_validate_json(row["data"])

    def delete(self, capability_id: str) -> None:
        if not self.exists(capability_id):
            raise KeyError(f"Capability {capability_id!r} not found in registry.")
        cursor = self._connection.cursor()
        cursor.execute(
            "DELETE FROM capabilities WHERE capability_id = ?", (capability_id,)
        )
        self._connection.commit()

    # ------------------------------------------------------------------
    # Listing / search
    # ------------------------------------------------------------------

    def list_all(self) -> list[AgentCapability]:
        cursor = self._connection.cursor()
        cursor.execute("SELECT data FROM capabilities")
        return [AgentCapability.model_validate_json(row["data"]) for row in cursor.fetchall()]

    def search(self, query: SearchQuery) -> list[AgentCapability]:
        sql_parts = ["SELECT data FROM capabilities WHERE 1=1"]
        params: list[object] = []

        if query.category is not None:
            sql_parts.append("AND category = ?")
            params.append(query.category.value)

        if query.min_trust > 0.0:
            sql_parts.append("AND trust_level >= ?")
            params.append(query.min_trust)

        if query.max_cost < float("inf"):
            sql_parts.append("AND cost <= ?")
            params.append(query.max_cost)

        if query.pricing_model:
            sql_parts.append("AND pricing_model = ?")
            params.append(query.pricing_model)

        if query.limit > 0:
            sql_parts.append("LIMIT ?")
            params.append(query.limit)
            sql_parts.append("OFFSET ?")
            params.append(query.offset)

        sql = " ".join(sql_parts)
        cursor = self._connection.cursor()
        cursor.execute(sql, params)
        candidates = [
            AgentCapability.model_validate_json(row["data"]) for row in cursor.fetchall()
        ]

        # Apply in-Python filters for fields that don't have SQL columns
        results: list[AgentCapability] = []
        for capability in candidates:
            if not self._python_filter(capability, query):
                continue
            results.append(capability)
        return results

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _upsert(self, capability: AgentCapability) -> None:
        cursor = self._connection.cursor()
        cursor.execute(
            """
            INSERT OR REPLACE INTO capabilities
                (capability_id, name, version, description, category, tags,
                 trust_level, cost, pricing_model, provider_name, data)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                capability.capability_id,
                capability.name,
                capability.version,
                capability.description,
                capability.category.value,
                json.dumps(capability.tags),
                capability.trust_level,
                capability.cost,
                capability.pricing_model.value,
                capability.provider.name,
                capability.model_dump_json(),
            ),
        )
        self._connection.commit()

    @staticmethod
    def _python_filter(capability: AgentCapability, query: SearchQuery) -> bool:
        """Apply filters that are cheaper in Python than in SQL."""
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

        if query.tags:
            capability_tags = {t.lower() for t in capability.tags}
            for required_tag in query.tags:
                if required_tag.lower() not in capability_tags:
                    return False

        if query.supported_language:
            lang_lower = query.supported_language.lower()
            if lang_lower not in [lang.lower() for lang in capability.supported_languages]:
                return False

        if query.supported_framework:
            fw_lower = query.supported_framework.lower()
            if fw_lower not in [fw.lower() for fw in capability.supported_frameworks]:
                return False

        return True
