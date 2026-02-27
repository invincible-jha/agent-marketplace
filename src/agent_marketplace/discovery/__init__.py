"""Discovery package for agent-marketplace.

Exports the search engine, embedding search, constraint filter,
fitness ranker, high-level discovery client, MCP scanner, and
auto-registrar for Phase 6C MCP capability discovery.
"""
from __future__ import annotations

from agent_marketplace.discovery.auto_register import AutoRegistrar, CapabilityRegistration
from agent_marketplace.discovery.client import DiscoveryClient, DiscoveryResult
from agent_marketplace.discovery.embeddings import EmbeddingSearch
from agent_marketplace.discovery.filter import ConstraintFilter, FilterConstraints
from agent_marketplace.discovery.mcp_scanner import MCPScanner, MCPServerInfo, MCPToolDefinition
from agent_marketplace.discovery.ranker import FitnessRanker, RankedCapability
from agent_marketplace.discovery.search import SearchEngine

__all__ = [
    "AutoRegistrar",
    "CapabilityRegistration",
    "ConstraintFilter",
    "DiscoveryClient",
    "DiscoveryResult",
    "EmbeddingSearch",
    "FilterConstraints",
    "FitnessRanker",
    "MCPScanner",
    "MCPServerInfo",
    "MCPToolDefinition",
    "RankedCapability",
    "SearchEngine",
]
