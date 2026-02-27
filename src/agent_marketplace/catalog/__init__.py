"""agent_marketplace.catalog — Tool integration catalog.

Provides a central registry for tool definitions that agents can discover
and invoke.  Supports keyword search, category filtering, and OpenAPI
schema import/export.

Public surface
--------------
ToolCatalog
    Central registry for tool entries.
ToolEntry
    A registered tool definition.
ToolCatalogError
    Base exception for this sub-package.
ToolAlreadyRegisteredError
    Raised when a tool name is already registered.
ToolNotFoundError
    Raised when a requested tool is not found.

Example
-------
::

    from agent_marketplace.catalog import ToolCatalog

    catalog = ToolCatalog()
    catalog.register_tool(
        name="web_search",
        description="Search the web for information",
        parameters_schema={"type": "object", "properties": {"query": {"type": "string"}}},
        handler=my_search_fn,
        category="search",
    )
    results = catalog.search("search")
    spec = catalog.export_openapi()
"""
from __future__ import annotations

from agent_marketplace.catalog.tool_catalog import (
    ToolAlreadyRegisteredError,
    ToolCatalog,
    ToolCatalogError,
    ToolEntry,
    ToolNotFoundError,
)

__all__ = [
    "ToolAlreadyRegisteredError",
    "ToolCatalog",
    "ToolCatalogError",
    "ToolEntry",
    "ToolNotFoundError",
]
