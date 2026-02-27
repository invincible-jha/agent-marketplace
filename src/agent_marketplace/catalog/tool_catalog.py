"""ToolCatalog — tool integration registry with search and OpenAPI support.

Provides a central in-memory registry for tool definitions that agents
can discover and invoke.  Extension points allow replacement of the
in-memory backend with a persistent store.

Commodity note
--------------
Search uses simple keyword matching (case-insensitive substring check
against name + description).  No vector search, no semantic similarity.

Classes
-------
ToolCatalogError
    Base exception for this module.
ToolAlreadyRegisteredError
    Raised on duplicate registration.
ToolNotFoundError
    Raised when a requested tool is not found.
ToolEntry
    Immutable descriptor for a registered tool.
ToolStorageBackend
    In-memory storage backend (extension point for persistent stores).
ToolCatalog
    Main catalog class.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class ToolCatalogError(Exception):
    """Base exception for all tool catalog errors."""


class ToolAlreadyRegisteredError(ToolCatalogError):
    """Raised when a tool name is already registered in the catalog."""

    def __init__(self, name: str) -> None:
        self.name = name
        super().__init__(f"Tool '{name}' is already registered in the catalog.")


class ToolNotFoundError(ToolCatalogError):
    """Raised when a requested tool is not found in the catalog."""

    def __init__(self, name: str) -> None:
        self.name = name
        super().__init__(f"Tool '{name}' not found in the catalog.")


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class ToolEntry:
    """A registered tool definition.

    Attributes
    ----------
    name:
        Unique tool name (slug-style, e.g. ``"web_search"``).
    description:
        Human-readable description of what the tool does.
    category:
        Logical grouping (e.g. ``"search"``, ``"data"``, ``"communication"``).
    parameters_schema:
        JSON Schema dict describing the tool's input parameters.
    handler:
        Callable that implements the tool.  May be ``None`` for catalog-only
        entries (e.g. imported from an OpenAPI spec without a local impl).
    version:
        Semantic version string (e.g. ``"1.0.0"``).
    """

    name: str
    description: str
    category: str
    parameters_schema: dict[str, object]
    handler: Callable[..., object] | None
    version: str = "1.0.0"


# ---------------------------------------------------------------------------
# Backend (extension point)
# ---------------------------------------------------------------------------


class ToolStorageBackend:
    """In-memory tool storage backend.

    Replace with a database-backed subclass to persist the catalog across
    process restarts.  All methods must be thread-safe in subclasses.

    Attributes
    ----------
    _tools:
        Mapping of tool name → ToolEntry.
    """

    def __init__(self) -> None:
        self._tools: dict[str, ToolEntry] = {}

    def save(self, entry: ToolEntry) -> None:
        """Store a tool entry.

        Parameters
        ----------
        entry:
            Tool entry to persist.
        """
        self._tools[entry.name] = entry

    def get(self, name: str) -> ToolEntry | None:
        """Retrieve a tool entry by name.

        Parameters
        ----------
        name:
            Tool name.

        Returns
        -------
        ToolEntry | None
            The entry, or None if not found.
        """
        return self._tools.get(name)

    def all(self) -> list[ToolEntry]:
        """Return all stored tool entries.

        Returns
        -------
        list[ToolEntry]
            All entries, in insertion order.
        """
        return list(self._tools.values())

    def contains(self, name: str) -> bool:
        """Return True if a tool with *name* exists.

        Parameters
        ----------
        name:
            Tool name.

        Returns
        -------
        bool
        """
        return name in self._tools


# ---------------------------------------------------------------------------
# Main catalog
# ---------------------------------------------------------------------------


class ToolCatalog:
    """Central registry for tool definitions with search and OpenAPI support.

    Parameters
    ----------
    backend:
        Storage backend.  Defaults to :class:`ToolStorageBackend` (in-memory).

    Example
    -------
    ::

        catalog = ToolCatalog()
        catalog.register_tool(
            name="weather",
            description="Get current weather for a location",
            parameters_schema={
                "type": "object",
                "properties": {"location": {"type": "string"}},
                "required": ["location"],
            },
            handler=get_weather,
            category="data",
        )
        results = catalog.search("weather")
        spec = catalog.export_openapi()
    """

    def __init__(self, backend: ToolStorageBackend | None = None) -> None:
        self._backend: ToolStorageBackend = (
            backend if backend is not None else ToolStorageBackend()
        )

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register_tool(
        self,
        name: str,
        description: str,
        parameters_schema: dict[str, object],
        handler: Callable[..., object] | None = None,
        category: str = "general",
        version: str = "1.0.0",
    ) -> ToolEntry:
        """Register a new tool in the catalog.

        Parameters
        ----------
        name:
            Unique tool name.
        description:
            Human-readable description.
        parameters_schema:
            JSON Schema describing the tool's input parameters.
        handler:
            Callable that implements the tool.  Pass ``None`` for
            catalog-only entries (e.g. imported from OpenAPI spec).
        category:
            Logical category for grouping and filtering.
        version:
            Semantic version string.

        Returns
        -------
        ToolEntry
            The newly created tool entry.

        Raises
        ------
        ToolAlreadyRegisteredError
            If a tool with *name* is already registered.
        """
        if self._backend.contains(name):
            raise ToolAlreadyRegisteredError(name)

        entry = ToolEntry(
            name=name,
            description=description,
            category=category,
            parameters_schema=parameters_schema,
            handler=handler,
            version=version,
        )
        self._backend.save(entry)
        return entry

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def get(self, name: str) -> ToolEntry | None:
        """Retrieve a tool entry by exact name.

        Parameters
        ----------
        name:
            Tool name.

        Returns
        -------
        ToolEntry | None
            The entry, or None if not found.
        """
        return self._backend.get(name)

    def get_or_raise(self, name: str) -> ToolEntry:
        """Retrieve a tool entry by name, raising if not found.

        Parameters
        ----------
        name:
            Tool name.

        Returns
        -------
        ToolEntry
            The entry.

        Raises
        ------
        ToolNotFoundError
            If *name* is not registered.
        """
        entry = self._backend.get(name)
        if entry is None:
            raise ToolNotFoundError(name)
        return entry

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        category: str | None = None,
    ) -> list[ToolEntry]:
        """Search for tools by keyword and optional category filter.

        Algorithm: case-insensitive substring match against name and
        description.  Results are sorted alphabetically by tool name.

        Parameters
        ----------
        query:
            Search terms (space-separated for multi-keyword search).
        category:
            When provided, only return tools in this category.

        Returns
        -------
        list[ToolEntry]
            Matching tool entries, sorted by name.
        """
        query_tokens = [t.lower() for t in query.split() if t.strip()]
        results: list[ToolEntry] = []

        for entry in self._backend.all():
            if category is not None and entry.category.lower() != category.lower():
                continue

            searchable = f"{entry.name} {entry.description}".lower()
            if all(token in searchable for token in query_tokens):
                results.append(entry)

        return sorted(results, key=lambda e: e.name)

    # ------------------------------------------------------------------
    # Category listing
    # ------------------------------------------------------------------

    def list_categories(self) -> list[str]:
        """Return a sorted list of all registered categories.

        Returns
        -------
        list[str]
            Unique category names in alphabetical order.
        """
        return sorted({entry.category for entry in self._backend.all()})

    def list_tools(self) -> list[ToolEntry]:
        """Return all registered tools sorted by name.

        Returns
        -------
        list[ToolEntry]
            All entries, sorted alphabetically.
        """
        return sorted(self._backend.all(), key=lambda e: e.name)

    # ------------------------------------------------------------------
    # OpenAPI export/import
    # ------------------------------------------------------------------

    def export_openapi(self) -> dict[str, object]:
        """Export the catalog as an OpenAPI-compatible schema.

        Produces a minimal OpenAPI 3.0.3 document where each tool is
        represented as a POST endpoint under ``/tools/{tool_name}``.

        Returns
        -------
        dict[str, object]
            OpenAPI 3.0.3 document dict.
        """
        paths: dict[str, object] = {}

        for entry in self._backend.all():
            path_key = f"/tools/{entry.name}"
            paths[path_key] = {
                "post": {
                    "operationId": entry.name,
                    "summary": entry.description,
                    "tags": [entry.category],
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": entry.parameters_schema,
                            }
                        },
                    },
                    "responses": {
                        "200": {
                            "description": "Successful tool invocation",
                            "content": {
                                "application/json": {
                                    "schema": {"type": "object"},
                                }
                            },
                        }
                    },
                    "x-tool-version": entry.version,
                }
            }

        return {
            "openapi": "3.0.3",
            "info": {
                "title": "AumOS Tool Catalog",
                "version": "1.0.0",
                "description": "Auto-generated tool catalog OpenAPI specification.",
            },
            "paths": paths,
        }

    def import_from_openapi(self, spec: dict[str, object]) -> list[ToolEntry]:
        """Import tools from an OpenAPI spec dict.

        Parses POST endpoints from ``spec["paths"]`` and registers each as
        a tool.  Tools that are already registered are skipped (not an error).

        Parameters
        ----------
        spec:
            OpenAPI 3.0.x specification dict.

        Returns
        -------
        list[ToolEntry]
            List of newly registered tool entries (excludes skipped duplicates).
        """
        imported: list[ToolEntry] = []
        paths = spec.get("paths", {})

        if not isinstance(paths, dict):
            return imported

        for path_key, path_item in paths.items():
            if not isinstance(path_item, dict):
                continue

            post_op = path_item.get("post")
            if not isinstance(post_op, dict):
                continue

            # Derive tool name from operationId or path
            tool_name: str = post_op.get("operationId") or _path_to_name(str(path_key))
            description: str = str(post_op.get("summary", f"Tool imported from {path_key}"))
            tags: list[str] = post_op.get("tags", [])  # type: ignore[assignment]
            category: str = tags[0] if tags else "imported"
            version: str = str(post_op.get("x-tool-version", "1.0.0"))

            # Extract parameters schema from requestBody
            parameters_schema: dict[str, object] = {"type": "object"}
            request_body = post_op.get("requestBody", {})
            if isinstance(request_body, dict):
                content = request_body.get("content", {})
                if isinstance(content, dict):
                    json_content = content.get("application/json", {})
                    if isinstance(json_content, dict):
                        schema = json_content.get("schema", {})
                        if isinstance(schema, dict):
                            parameters_schema = schema

            # Skip duplicates silently
            if self._backend.contains(tool_name):
                continue

            try:
                entry = self.register_tool(
                    name=tool_name,
                    description=description,
                    parameters_schema=parameters_schema,
                    handler=None,
                    category=category,
                    version=version,
                )
                imported.append(entry)
            except ToolAlreadyRegisteredError:
                pass  # Race condition guard

        return imported


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _path_to_name(path: str) -> str:
    """Convert an OpenAPI path string to a tool name slug.

    Example: ``"/tools/web_search"`` → ``"web_search"``
             ``"/api/v1/weather-lookup"`` → ``"weather_lookup"``

    Parameters
    ----------
    path:
        OpenAPI path string.

    Returns
    -------
    str
        Slug-style tool name.
    """
    # Take the last path segment and convert to snake_case
    segment = path.rstrip("/").rsplit("/", 1)[-1]
    # Replace hyphens and other non-alphanumeric chars with underscores
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", segment).strip("_")
    return slug.lower() if slug else "unknown_tool"
