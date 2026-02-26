"""Server sub-package for agent-marketplace.

Provides framework-agnostic HTTP route handlers and a lightweight health
check endpoint.  No HTTP framework dependency is introduced here — callers
wire the handlers into their framework of choice.

Public API
----------
- ``MarketplaceAPI``  — route handlers as plain Python methods.
- ``HealthEndpoint``  — returns a structured health status dict.
"""
from __future__ import annotations

from agent_marketplace.server.api import MarketplaceAPI
from agent_marketplace.server.health import HealthEndpoint

__all__ = [
    "HealthEndpoint",
    "MarketplaceAPI",
]
