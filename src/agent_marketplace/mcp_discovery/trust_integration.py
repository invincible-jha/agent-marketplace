"""MCP server discovery with marketplace trust integration.

Design
------
:class:`TrustedMCPDiscovery` wraps the base :class:`MCPScanner` and
enriches each discovered server with a trust score computed from a
simple heuristic: known servers score higher; servers with no tools
score lower; auth-required servers get a small premium.

:class:`TrustedServerRecord` pairs an :class:`MCPServerInfo` snapshot
with its computed trust score.

No external network calls are made — trust is computed from the server
metadata available at scan time.  Callers can supply a custom trust
function to override the default heuristics.

Usage
-----
::

    from pathlib import Path
    from agent_marketplace.mcp_discovery import TrustedMCPDiscovery

    discovery = TrustedMCPDiscovery()
    records = discovery.discover_from_file(Path("claude_desktop_config.json"))
    for record in records:
        print(record.server_name, record.trust_score, record.tool_count)
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from agent_marketplace.discovery.mcp_scanner import (
    MCPScanner,
    MCPServerInfo,
)


# ---------------------------------------------------------------------------
# TrustedServerRecord
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TrustedServerRecord:
    """A discovered MCP server annotated with a trust score.

    Parameters
    ----------
    server_info:
        The raw scan result from :class:`MCPScanner`.
    trust_score:
        Computed trust score in [0.0, 1.0].
    trust_reasoning:
        Human-readable explanation of how the score was derived.
    is_trusted:
        True when trust_score >= the discovery's min_trust_threshold.
    discovered_at:
        UTC timestamp of discovery + trust computation.
    """

    server_info: MCPServerInfo
    trust_score: float
    trust_reasoning: str
    is_trusted: bool
    discovered_at: datetime

    @property
    def server_name(self) -> str:
        """Shorthand for ``server_info.server_name``."""
        return self.server_info.server_name

    @property
    def tool_count(self) -> int:
        """Number of tools exposed by this server."""
        return len(self.server_info.tools)

    @property
    def transport(self) -> str:
        """Server transport mechanism."""
        return self.server_info.transport

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serialisable dictionary."""
        return {
            "server_name": self.server_name,
            "trust_score": self.trust_score,
            "trust_reasoning": self.trust_reasoning,
            "is_trusted": self.is_trusted,
            "tool_count": self.tool_count,
            "transport": self.transport,
            "version": self.server_info.version,
            "discovered_at": self.discovered_at.isoformat(),
        }


# ---------------------------------------------------------------------------
# Default trust heuristic
# ---------------------------------------------------------------------------

# Servers known to be from reputable sources get a base trust boost.
_KNOWN_TRUSTED_SERVERS: frozenset[str] = frozenset({
    "filesystem",
    "memory",
    "brave-search",
    "fetch",
    "puppeteer",
    "sqlite",
    "github",
    "gitlab",
    "google-drive",
    "slack",
})


def _default_trust_scorer(server: MCPServerInfo) -> tuple[float, str]:
    """Compute trust score for *server* using simple heuristics.

    Rules (each contributes to a 0-1 score):
    1. Known-server bonus: +0.3 if server_name in known list.
    2. Tool count: +0.2 if >= 3 tools, +0.1 if >= 1 tool.
    3. Auth requirement: +0.1 if any tool requires auth (indicates controls).
    4. Has description: +0.2 if any tool has a non-empty description.
    5. Version declared: +0.1 if version != "unknown".
    6. Stdio transport (local): +0.1 (lower attack surface).

    Score is clamped to [0.0, 1.0].
    """
    score = 0.0
    reasons: list[str] = []

    # Known server bonus
    if server.server_name.lower() in _KNOWN_TRUSTED_SERVERS:
        score += 0.3
        reasons.append("known-server(+0.30)")

    # Tool count
    tool_count = len(server.tools)
    if tool_count >= 3:
        score += 0.2
        reasons.append(f"tool-count>={tool_count}(+0.20)")
    elif tool_count >= 1:
        score += 0.1
        reasons.append(f"tool-count={tool_count}(+0.10)")

    # Auth requirement
    if any(t.auth_required for t in server.tools):
        score += 0.1
        reasons.append("auth-required(+0.10)")

    # Has descriptions
    if any(t.description.strip() for t in server.tools):
        score += 0.2
        reasons.append("has-descriptions(+0.20)")

    # Version declared
    if server.version and server.version.lower() != "unknown":
        score += 0.1
        reasons.append("versioned(+0.10)")

    # Local stdio transport
    if server.transport == "stdio":
        score += 0.1
        reasons.append("stdio-local(+0.10)")

    score = max(0.0, min(1.0, score))
    reasoning = "; ".join(reasons) if reasons else "no-signals(0.0)"
    return score, reasoning


# ---------------------------------------------------------------------------
# TrustedMCPDiscovery
# ---------------------------------------------------------------------------


class TrustedMCPDiscovery:
    """Discover MCP servers and annotate them with trust scores.

    Parameters
    ----------
    min_trust_threshold:
        Servers with ``trust_score >= min_trust_threshold`` have
        ``is_trusted=True``.  Default 0.4.
    trust_scorer:
        Optional custom function ``(MCPServerInfo) -> (float, str)``.
        If None, the default heuristic is used.

    Example
    -------
    ::

        discovery = TrustedMCPDiscovery(min_trust_threshold=0.5)
        records = discovery.discover_from_file(Path("mcp_config.json"))
        trusted = discovery.filter_trusted(records)
    """

    def __init__(
        self,
        min_trust_threshold: float = 0.4,
        trust_scorer: Callable[[MCPServerInfo], tuple[float, str]] | None = None,
    ) -> None:
        if not (0.0 <= min_trust_threshold <= 1.0):
            raise ValueError(
                f"min_trust_threshold must be in [0.0, 1.0], "
                f"got {min_trust_threshold!r}."
            )
        self._min_trust = min_trust_threshold
        self._scorer = trust_scorer or _default_trust_scorer
        self._scanner = MCPScanner()

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    def discover_from_file(self, config_path: Path) -> list[TrustedServerRecord]:
        """Scan *config_path* and return trusted records.

        Parameters
        ----------
        config_path:
            Path to an MCP configuration file (JSON or YAML).

        Returns
        -------
        list[TrustedServerRecord]
            All discovered servers with trust annotations.

        Raises
        ------
        FileNotFoundError
            If *config_path* does not exist.
        ValueError
            If the file cannot be parsed.
        """
        servers = self._scanner.scan_file(config_path)
        return [self._annotate(server) for server in servers]

    def discover_from_dict(self, config: dict[str, object]) -> list[TrustedServerRecord]:
        """Scan a config dict and return trusted records.

        Parameters
        ----------
        config:
            MCP configuration dict (same format as a parsed config file).

        Returns
        -------
        list[TrustedServerRecord]
        """
        servers = self._scanner._extract_servers(config)
        return [self._annotate(server) for server in servers]

    def discover_from_server_info(
        self, servers: list[MCPServerInfo]
    ) -> list[TrustedServerRecord]:
        """Annotate pre-scanned :class:`MCPServerInfo` with trust scores.

        Parameters
        ----------
        servers:
            List of already-scanned server info objects.

        Returns
        -------
        list[TrustedServerRecord]
        """
        return [self._annotate(server) for server in servers]

    # ------------------------------------------------------------------
    # Filtering
    # ------------------------------------------------------------------

    def filter_trusted(
        self, records: list[TrustedServerRecord]
    ) -> list[TrustedServerRecord]:
        """Return only records with ``is_trusted=True``.

        Parameters
        ----------
        records:
            Records to filter.

        Returns
        -------
        list[TrustedServerRecord]
        """
        return [r for r in records if r.is_trusted]

    def sort_by_trust(
        self, records: list[TrustedServerRecord], descending: bool = True
    ) -> list[TrustedServerRecord]:
        """Return records sorted by trust_score.

        Parameters
        ----------
        records:
            Records to sort.
        descending:
            Highest trust first when True.

        Returns
        -------
        list[TrustedServerRecord]
        """
        return sorted(records, key=lambda r: r.trust_score, reverse=descending)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def min_trust_threshold(self) -> float:
        """Configured minimum trust threshold."""
        return self._min_trust

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _annotate(self, server: MCPServerInfo) -> TrustedServerRecord:
        trust_score, reasoning = self._scorer(server)
        return TrustedServerRecord(
            server_info=server,
            trust_score=trust_score,
            trust_reasoning=reasoning,
            is_trusted=trust_score >= self._min_trust,
            discovered_at=datetime.now(timezone.utc),
        )
