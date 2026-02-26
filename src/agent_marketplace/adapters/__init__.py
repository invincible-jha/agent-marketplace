"""Adapters sub-package for agent-marketplace.

Provides import adapters that convert external API specification formats
into ``AgentCapability`` objects suitable for registration in the marketplace.

Supported formats
-----------------
- OpenAPI 3.x (JSON and YAML) via ``OpenAPIAdapter``
- AsyncAPI 2.x/3.x (JSON and YAML) via ``AsyncAPIAdapter``
- MCP tool manifests (JSON and YAML) via ``MCPAdapter``

Public API
----------
- ``OpenAPIAdapter``  — imports capabilities from OpenAPI specs.
- ``AsyncAPIAdapter`` — imports capabilities from AsyncAPI specs.
- ``MCPAdapter``      — imports capabilities from MCP tool manifests.
"""
from __future__ import annotations

from agent_marketplace.adapters.asyncapi import AsyncAPIAdapter
from agent_marketplace.adapters.mcp_adapter import MCPAdapter
from agent_marketplace.adapters.openapi import OpenAPIAdapter

__all__ = [
    "AsyncAPIAdapter",
    "MCPAdapter",
    "OpenAPIAdapter",
]
