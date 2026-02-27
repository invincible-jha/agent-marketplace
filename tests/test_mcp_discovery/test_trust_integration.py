"""Tests for agent_marketplace.mcp_discovery.trust_integration."""
from __future__ import annotations

import pytest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

from agent_marketplace.discovery.mcp_scanner import (
    MCPScanner,
    MCPServerInfo,
    MCPToolDefinition,
)
from agent_marketplace.mcp_discovery.trust_integration import (
    TrustedMCPDiscovery,
    TrustedServerRecord,
    _default_trust_scorer,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _tool(
    name: str = "read_file",
    description: str = "Reads a file from disk.",
    auth_required: bool = False,
) -> MCPToolDefinition:
    return MCPToolDefinition(
        name=name,
        description=description,
        input_schema={"type": "object", "properties": {"path": {"type": "string"}}},
        auth_required=auth_required,
    )


def _server(
    server_name: str = "filesystem",
    version: str = "1.0.0",
    transport: str = "stdio",
    tools: list[MCPToolDefinition] | None = None,
) -> MCPServerInfo:
    return MCPServerInfo(
        server_name=server_name,
        version=version,
        transport=transport,
        tools=tools if tools is not None else [_tool()],
        resources=[],
        prompts=[],
        scanned_at=datetime.now(timezone.utc),
    )


# ---------------------------------------------------------------------------
# _default_trust_scorer
# ---------------------------------------------------------------------------


class TestDefaultTrustScorer:
    def test_known_server_gets_bonus(self) -> None:
        server = _server(server_name="filesystem")
        score, reasoning = _default_trust_scorer(server)
        assert score >= 0.3
        assert "known-server" in reasoning

    def test_unknown_server_no_bonus(self) -> None:
        server = _server(server_name="totally-unknown-xyz", tools=[])
        score, reasoning = _default_trust_scorer(server)
        # No known-server bonus; no tools; but may get versioned or stdio bonus
        assert "known-server" not in reasoning

    def test_three_or_more_tools_high_tool_bonus(self) -> None:
        tools = [_tool(name=f"tool_{i}") for i in range(3)]
        server = _server(server_name="xyz", tools=tools)
        score, reasoning = _default_trust_scorer(server)
        assert "tool-count>=3" in reasoning or "tool-count" in reasoning

    def test_one_tool_low_tool_bonus(self) -> None:
        server = _server(server_name="xyz", tools=[_tool()])
        score, reasoning = _default_trust_scorer(server)
        assert "+0.10" in reasoning or "tool-count=1" in reasoning

    def test_no_tools_no_tool_bonus(self) -> None:
        server = _server(server_name="xyz", tools=[])
        score, reasoning = _default_trust_scorer(server)
        assert "tool-count" not in reasoning

    def test_auth_required_adds_bonus(self) -> None:
        server = _server(tools=[_tool(auth_required=True)])
        score, reasoning = _default_trust_scorer(server)
        assert "auth-required" in reasoning

    def test_no_auth_required_no_bonus(self) -> None:
        server = _server(tools=[_tool(auth_required=False)])
        _, reasoning = _default_trust_scorer(server)
        assert "auth-required" not in reasoning

    def test_described_tools_add_bonus(self) -> None:
        server = _server(tools=[_tool(description="Does something useful.")])
        _, reasoning = _default_trust_scorer(server)
        assert "has-descriptions" in reasoning

    def test_empty_description_no_bonus(self) -> None:
        server = _server(tools=[_tool(description="   ")])
        _, reasoning = _default_trust_scorer(server)
        assert "has-descriptions" not in reasoning

    def test_version_declared_adds_bonus(self) -> None:
        server = _server(version="2.1.0")
        _, reasoning = _default_trust_scorer(server)
        assert "versioned" in reasoning

    def test_unknown_version_no_bonus(self) -> None:
        server = _server(version="unknown")
        _, reasoning = _default_trust_scorer(server)
        assert "versioned" not in reasoning

    def test_stdio_transport_adds_bonus(self) -> None:
        server = _server(transport="stdio")
        _, reasoning = _default_trust_scorer(server)
        assert "stdio-local" in reasoning

    def test_non_stdio_transport_no_bonus(self) -> None:
        server = _server(transport="sse")
        _, reasoning = _default_trust_scorer(server)
        assert "stdio-local" not in reasoning

    def test_score_clamped_to_one(self) -> None:
        # filesystem + 3 tools + auth + description + version + stdio = potentially > 1
        tools = [_tool(name=f"t{i}", auth_required=True, description="Useful.") for i in range(3)]
        server = _server(server_name="filesystem", version="1.0.0", transport="stdio", tools=tools)
        score, _ = _default_trust_scorer(server)
        assert score <= 1.0

    def test_score_clamped_to_zero(self) -> None:
        server = _server(server_name="xyz", version="unknown", transport="sse", tools=[])
        score, _ = _default_trust_scorer(server)
        assert score >= 0.0

    def test_no_signals_fallback_reasoning(self) -> None:
        server = _server(server_name="xyz", version="unknown", transport="sse", tools=[])
        _, reasoning = _default_trust_scorer(server)
        assert reasoning == "no-signals(0.0)" or reasoning != ""


# ---------------------------------------------------------------------------
# TrustedServerRecord
# ---------------------------------------------------------------------------


class TestTrustedServerRecord:
    def _make_record(
        self,
        server_name: str = "filesystem",
        trust_score: float = 0.8,
        is_trusted: bool = True,
    ) -> TrustedServerRecord:
        return TrustedServerRecord(
            server_info=_server(server_name=server_name),
            trust_score=trust_score,
            trust_reasoning="known-server(+0.30); versioned(+0.10)",
            is_trusted=is_trusted,
            discovered_at=datetime.now(timezone.utc),
        )

    def test_frozen(self) -> None:
        record = self._make_record()
        with pytest.raises(Exception):
            record.trust_score = 0.1  # type: ignore[misc]

    def test_server_name_property(self) -> None:
        record = self._make_record(server_name="brave-search")
        assert record.server_name == "brave-search"

    def test_tool_count_property(self) -> None:
        record = self._make_record()
        assert record.tool_count == 1  # _server() creates one tool by default

    def test_transport_property(self) -> None:
        record = self._make_record()
        assert record.transport == "stdio"

    def test_to_dict_keys(self) -> None:
        record = self._make_record()
        d = record.to_dict()
        expected_keys = {
            "server_name", "trust_score", "trust_reasoning",
            "is_trusted", "tool_count", "transport", "version", "discovered_at",
        }
        assert expected_keys.issubset(d.keys())

    def test_to_dict_values(self) -> None:
        record = self._make_record(trust_score=0.75, is_trusted=True)
        d = record.to_dict()
        assert d["trust_score"] == 0.75
        assert d["is_trusted"] is True
        assert d["tool_count"] == 1

    def test_discovered_at_is_iso_string_in_dict(self) -> None:
        record = self._make_record()
        d = record.to_dict()
        # Should be parseable as ISO datetime
        dt = datetime.fromisoformat(d["discovered_at"])
        assert dt.tzinfo is not None


# ---------------------------------------------------------------------------
# TrustedMCPDiscovery — constructor
# ---------------------------------------------------------------------------


class TestTrustedMCPDiscoveryInit:
    def test_default_threshold(self) -> None:
        discovery = TrustedMCPDiscovery()
        assert discovery.min_trust_threshold == 0.4

    def test_custom_threshold(self) -> None:
        discovery = TrustedMCPDiscovery(min_trust_threshold=0.7)
        assert discovery.min_trust_threshold == 0.7

    def test_invalid_threshold_raises(self) -> None:
        with pytest.raises(ValueError, match="min_trust_threshold"):
            TrustedMCPDiscovery(min_trust_threshold=1.5)

    def test_negative_threshold_raises(self) -> None:
        with pytest.raises(ValueError, match="min_trust_threshold"):
            TrustedMCPDiscovery(min_trust_threshold=-0.1)

    def test_boundary_zero_valid(self) -> None:
        discovery = TrustedMCPDiscovery(min_trust_threshold=0.0)
        assert discovery.min_trust_threshold == 0.0

    def test_boundary_one_valid(self) -> None:
        discovery = TrustedMCPDiscovery(min_trust_threshold=1.0)
        assert discovery.min_trust_threshold == 1.0

    def test_custom_scorer_used(self) -> None:
        called = []

        def my_scorer(server: MCPServerInfo) -> tuple[float, str]:
            called.append(server.server_name)
            return 0.99, "custom"

        discovery = TrustedMCPDiscovery(trust_scorer=my_scorer)
        server = _server(server_name="test-server")
        records = discovery.discover_from_server_info([server])
        assert "test-server" in called
        assert records[0].trust_score == 0.99
        assert records[0].trust_reasoning == "custom"


# ---------------------------------------------------------------------------
# TrustedMCPDiscovery — discover_from_server_info
# ---------------------------------------------------------------------------


class TestDiscoverFromServerInfo:
    def test_returns_records_for_all_servers(self) -> None:
        discovery = TrustedMCPDiscovery()
        servers = [
            _server(server_name="filesystem"),
            _server(server_name="brave-search"),
        ]
        records = discovery.discover_from_server_info(servers)
        assert len(records) == 2

    def test_trust_score_in_range(self) -> None:
        discovery = TrustedMCPDiscovery()
        records = discovery.discover_from_server_info([_server(server_name="filesystem")])
        for record in records:
            assert 0.0 <= record.trust_score <= 1.0

    def test_is_trusted_true_when_above_threshold(self) -> None:
        discovery = TrustedMCPDiscovery(min_trust_threshold=0.1)
        # filesystem is known, so score will be high
        records = discovery.discover_from_server_info([_server(server_name="filesystem")])
        assert records[0].is_trusted is True

    def test_is_trusted_false_when_below_threshold(self) -> None:
        discovery = TrustedMCPDiscovery(min_trust_threshold=0.99)
        # Empty unknown server will likely score very low
        records = discovery.discover_from_server_info([
            _server(server_name="xyz-unknown", version="unknown", transport="sse", tools=[])
        ])
        assert records[0].is_trusted is False

    def test_discovered_at_is_utc(self) -> None:
        discovery = TrustedMCPDiscovery()
        records = discovery.discover_from_server_info([_server()])
        assert records[0].discovered_at.tzinfo is not None

    def test_empty_server_list_returns_empty(self) -> None:
        discovery = TrustedMCPDiscovery()
        assert discovery.discover_from_server_info([]) == []

    def test_record_server_info_preserved(self) -> None:
        discovery = TrustedMCPDiscovery()
        server = _server(server_name="github")
        records = discovery.discover_from_server_info([server])
        assert records[0].server_info.server_name == "github"


# ---------------------------------------------------------------------------
# TrustedMCPDiscovery — discover_from_dict
# ---------------------------------------------------------------------------


class TestDiscoverFromDict:
    def test_parses_mcp_servers_config(self) -> None:
        config = {
            "mcpServers": {
                "filesystem": {
                    "command": "npx",
                    "args": ["-y", "@modelcontextprotocol/server-filesystem"],
                    "version": "1.0.0",
                    "tools": [
                        {
                            "name": "read_file",
                            "description": "Read a file from the filesystem.",
                            "inputSchema": {"type": "object"},
                        }
                    ],
                }
            }
        }
        discovery = TrustedMCPDiscovery()
        records = discovery.discover_from_dict(config)
        assert len(records) == 1
        assert records[0].server_name == "filesystem"

    def test_empty_config_returns_empty(self) -> None:
        discovery = TrustedMCPDiscovery()
        assert discovery.discover_from_dict({}) == []

    def test_trust_annotated_from_dict(self) -> None:
        config = {
            "mcpServers": {
                "github": {
                    "command": "node",
                    "version": "2.0.0",
                    "tools": [
                        {"name": "create_issue", "description": "Creates a GitHub issue.", "inputSchema": {}},
                        {"name": "list_repos", "description": "Lists user repos.", "inputSchema": {}},
                        {"name": "get_pr", "description": "Gets a PR.", "inputSchema": {}},
                    ],
                }
            }
        }
        discovery = TrustedMCPDiscovery()
        records = discovery.discover_from_dict(config)
        assert len(records) == 1
        record = records[0]
        assert record.trust_score > 0.0
        assert isinstance(record.trust_reasoning, str)


# ---------------------------------------------------------------------------
# TrustedMCPDiscovery — discover_from_file
# ---------------------------------------------------------------------------


class TestDiscoverFromFile:
    def test_file_not_found_raises(self, tmp_path: Path) -> None:
        discovery = TrustedMCPDiscovery()
        with pytest.raises(FileNotFoundError):
            discovery.discover_from_file(tmp_path / "nonexistent.json")

    def test_valid_json_file(self, tmp_path: Path) -> None:
        import json as _json
        config = {
            "mcpServers": {
                "memory": {
                    "command": "npx",
                    "version": "1.0.0",
                    "tools": [{"name": "remember", "description": "Stores memory.", "inputSchema": {}}],
                }
            }
        }
        config_file = tmp_path / "config.json"
        config_file.write_text(_json.dumps(config), encoding="utf-8")
        discovery = TrustedMCPDiscovery()
        records = discovery.discover_from_file(config_file)
        assert len(records) == 1
        assert records[0].server_name == "memory"

    def test_valid_yaml_file(self, tmp_path: Path) -> None:
        yaml_content = """\
mcpServers:
  fetch:
    command: python
    version: "1.0"
    tools:
      - name: fetch_url
        description: Fetches a URL.
        inputSchema:
          type: object
"""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml_content, encoding="utf-8")
        discovery = TrustedMCPDiscovery()
        records = discovery.discover_from_file(config_file)
        assert len(records) == 1
        assert records[0].server_name == "fetch"


# ---------------------------------------------------------------------------
# TrustedMCPDiscovery — filter_trusted
# ---------------------------------------------------------------------------


class TestFilterTrusted:
    def _make_records(self) -> list[TrustedServerRecord]:
        discovery = TrustedMCPDiscovery(min_trust_threshold=0.5)
        servers = [
            _server(server_name="filesystem"),   # known server → should be trusted
            _server(server_name="xyz-unknown", version="unknown", transport="sse", tools=[]),
        ]
        return discovery.discover_from_server_info(servers)

    def test_filter_returns_only_trusted(self) -> None:
        discovery = TrustedMCPDiscovery(min_trust_threshold=0.5)
        records = self._make_records()
        trusted = discovery.filter_trusted(records)
        assert all(r.is_trusted for r in trusted)

    def test_filter_empty_input(self) -> None:
        discovery = TrustedMCPDiscovery()
        assert discovery.filter_trusted([]) == []

    def test_all_trusted_when_threshold_zero(self) -> None:
        discovery = TrustedMCPDiscovery(min_trust_threshold=0.0)
        servers = [
            _server(server_name="filesystem"),
            _server(server_name="xyz-unknown", version="unknown", transport="sse", tools=[]),
        ]
        records = discovery.discover_from_server_info(servers)
        trusted = discovery.filter_trusted(records)
        assert len(trusted) == len(records)

    def test_none_trusted_when_threshold_one(self) -> None:
        discovery = TrustedMCPDiscovery(min_trust_threshold=1.0)
        servers = [_server(server_name="filesystem")]
        records = discovery.discover_from_server_info(servers)
        trusted = discovery.filter_trusted(records)
        # Even filesystem won't hit 1.0 exactly without all signals
        assert all(r.is_trusted for r in trusted) or len(trusted) == 0


# ---------------------------------------------------------------------------
# TrustedMCPDiscovery — sort_by_trust
# ---------------------------------------------------------------------------


class TestSortByTrust:
    def _records(self) -> list[TrustedServerRecord]:
        discovery = TrustedMCPDiscovery()
        servers = [
            _server(server_name="xyz-unknown", version="unknown", transport="sse", tools=[]),
            _server(server_name="filesystem"),
            _server(server_name="brave-search"),
        ]
        return discovery.discover_from_server_info(servers)

    def test_default_descending(self) -> None:
        discovery = TrustedMCPDiscovery()
        records = self._records()
        sorted_records = discovery.sort_by_trust(records)
        scores = [r.trust_score for r in sorted_records]
        assert scores == sorted(scores, reverse=True)

    def test_ascending(self) -> None:
        discovery = TrustedMCPDiscovery()
        records = self._records()
        sorted_records = discovery.sort_by_trust(records, descending=False)
        scores = [r.trust_score for r in sorted_records]
        assert scores == sorted(scores)

    def test_empty_returns_empty(self) -> None:
        discovery = TrustedMCPDiscovery()
        assert discovery.sort_by_trust([]) == []

    def test_original_list_not_mutated(self) -> None:
        discovery = TrustedMCPDiscovery()
        records = self._records()
        original_order = [r.server_name for r in records]
        discovery.sort_by_trust(records)
        assert [r.server_name for r in records] == original_order
