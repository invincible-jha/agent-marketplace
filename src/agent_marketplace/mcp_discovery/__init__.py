"""MCP server discovery with trust integration.

Extends the base MCP scanner with trust-score integration so that
discovered servers and their capabilities are annotated with initial
trust levels derived from the marketplace trust layer.

Classes
-------
TrustedMCPDiscovery
    Wraps the base scanner; assigns trust levels from the trust scorer.
TrustedServerRecord
    A discovered MCP server with its capabilities and trust score.
"""
from __future__ import annotations

from agent_marketplace.mcp_discovery.trust_integration import (
    TrustedMCPDiscovery,
    TrustedServerRecord,
)

__all__ = [
    "TrustedMCPDiscovery",
    "TrustedServerRecord",
]
