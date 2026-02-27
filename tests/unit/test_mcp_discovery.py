"""Unit tests for Phase 6C MCP capability discovery.

Covers:
- MCPToolDefinition and MCPServerInfo construction
- MCPScanner with various MCP config formats (Claude Desktop, Cursor, bare list)
- Tool categorisation heuristics
- Quality scoring algorithm
- AutoRegistrar: registration, deduplication, import/export round-trip
- CLI discover sub-commands (scan, register, list)
- Edge cases: empty tools, missing fields, malformed configs
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

import pytest
from click.testing import CliRunner

from agent_marketplace.discovery.auto_register import AutoRegistrar, CapabilityRegistration
from agent_marketplace.discovery.mcp_scanner import (
    MCPScanner,
    MCPServerInfo,
    MCPToolDefinition,
)


# ---------------------------------------------------------------------------
# Helpers / factories
# ---------------------------------------------------------------------------


def _now() -> datetime:
    return datetime.now(tz=timezone.utc)


def _tool(
    name: str = "my-tool",
    description: str = "A helpful tool for testing things.",
    input_schema: dict | None = None,
    output_schema: dict | None = None,
    auth_required: bool = False,
    tags: list[str] | None = None,
) -> MCPToolDefinition:
    return MCPToolDefinition(
        name=name,
        description=description,
        input_schema=input_schema or {"type": "object", "properties": {}},
        output_schema=output_schema,
        auth_required=auth_required,
        tags=tags or [],
    )


def _server(
    server_name: str = "test-server",
    tools: list[MCPToolDefinition] | None = None,
    transport: str = "stdio",
    version: str = "1.0.0",
) -> MCPServerInfo:
    return MCPServerInfo(
        server_name=server_name,
        version=version,
        transport=transport,
        tools=tools or [],
        resources=[],
        prompts=[],
        scanned_at=_now(),
    )


# ---------------------------------------------------------------------------
# MCPToolDefinition — construction
# ---------------------------------------------------------------------------


class TestMCPToolDefinitionConstruction:
    def test_minimal_construction(self) -> None:
        tool = MCPToolDefinition(name="t", description="d", input_schema={})
        assert tool.name == "t"
        assert tool.description == "d"
        assert tool.input_schema == {}
        assert tool.output_schema is None
        assert tool.auth_required is False
        assert tool.tags == []

    def test_full_construction(self) -> None:
        schema = {"type": "object", "properties": {"q": {"type": "string"}}}
        tool = MCPToolDefinition(
            name="search",
            description="Searches the web",
            input_schema=schema,
            output_schema={"type": "array"},
            auth_required=True,
            tags=["search", "web"],
        )
        assert tool.auth_required is True
        assert tool.tags == ["search", "web"]
        assert tool.output_schema == {"type": "array"}

    def test_frozen_prevents_mutation(self) -> None:
        tool = _tool()
        with pytest.raises((AttributeError, TypeError)):
            tool.name = "other"  # type: ignore[misc]

    def test_non_list_tags_coerced(self) -> None:
        # Pass a tuple; __post_init__ should convert to list
        tool = MCPToolDefinition(
            name="t",
            description="d",
            input_schema={},
            tags=("a", "b"),  # type: ignore[arg-type]
        )
        assert isinstance(tool.tags, list)

    def test_tags_with_non_list_in_raw_parse(self) -> None:
        # Cover the non-list branch in _parse_tool for tags
        scanner = MCPScanner()
        defn = {
            "command": "python",
            "tools": [{"name": "t", "description": "d", "inputSchema": {}, "tags": "not-a-list"}],
        }
        info = scanner.scan_definition(defn)
        assert info.tools[0].tags == []

    def test_empty_input_schema(self) -> None:
        tool = MCPToolDefinition(name="t", description="d", input_schema={})
        assert tool.input_schema == {}

    def test_rich_input_schema_preserved(self) -> None:
        schema = {
            "type": "object",
            "required": ["path"],
            "properties": {"path": {"type": "string"}},
        }
        tool = _tool(input_schema=schema)
        assert tool.input_schema["required"] == ["path"]


# ---------------------------------------------------------------------------
# MCPServerInfo — construction
# ---------------------------------------------------------------------------


class TestMCPServerInfoConstruction:
    def test_minimal_construction(self) -> None:
        server = _server()
        assert server.server_name == "test-server"
        assert server.tools == []
        assert server.resources == []
        assert server.prompts == []

    def test_full_construction(self) -> None:
        tools = [_tool("t1"), _tool("t2")]
        server = MCPServerInfo(
            server_name="fs",
            version="2.0.0",
            transport="sse",
            tools=tools,
            resources=[{"uri": "file://tmp"}],
            prompts=[{"name": "p1"}],
            scanned_at=_now(),
        )
        assert len(server.tools) == 2
        assert server.transport == "sse"
        assert server.resources[0]["uri"] == "file://tmp"

    def test_frozen_prevents_mutation(self) -> None:
        server = _server()
        with pytest.raises((AttributeError, TypeError)):
            server.server_name = "other"  # type: ignore[misc]

    def test_scanned_at_is_datetime(self) -> None:
        server = _server()
        assert isinstance(server.scanned_at, datetime)

    def test_version_unknown_default(self) -> None:
        server = _server(version="unknown")
        assert server.version == "unknown"


# ---------------------------------------------------------------------------
# MCPScanner — scan_definition
# ---------------------------------------------------------------------------


class TestMCPScannerScanDefinition:
    def setup_method(self) -> None:
        self.scanner = MCPScanner()

    def test_minimal_definition(self) -> None:
        defn = {"command": "npx", "args": ["-y", "@mcp/server-fs"]}
        info = self.scanner.scan_definition(defn)
        assert info.transport == "stdio"
        assert info.tools == []

    def test_name_extracted(self) -> None:
        defn = {"name": "my-server", "command": "python", "args": ["-m", "mcp_server"]}
        info = self.scanner.scan_definition(defn)
        assert info.server_name == "my-server"

    def test_server_name_key_used_as_fallback(self) -> None:
        defn = {"server_name": "fs-server", "command": "node", "args": ["server.js"]}
        info = self.scanner.scan_definition(defn)
        assert info.server_name == "fs-server"

    def test_unnamed_server_gets_default(self) -> None:
        defn = {"command": "node", "args": ["server.js"]}
        info = self.scanner.scan_definition(defn)
        assert info.server_name == "unnamed"

    def test_transport_explicit_sse(self) -> None:
        defn = {"transport": "sse", "url": "http://localhost:8080/sse"}
        info = self.scanner.scan_definition(defn)
        assert info.transport == "sse"

    def test_transport_explicit_stdio(self) -> None:
        defn = {"transport": "stdio", "command": "python"}
        info = self.scanner.scan_definition(defn)
        assert info.transport == "stdio"

    def test_transport_explicit_streamable_http(self) -> None:
        defn = {"transport": "streamable-http", "url": "http://localhost:8080"}
        info = self.scanner.scan_definition(defn)
        assert info.transport == "streamable-http"

    def test_transport_inferred_from_command(self) -> None:
        defn = {"command": "npx", "args": ["-y", "@mcp/server"]}
        info = self.scanner.scan_definition(defn)
        assert info.transport == "stdio"

    def test_transport_inferred_from_url(self) -> None:
        defn = {"url": "http://localhost:3000"}
        info = self.scanner.scan_definition(defn)
        assert info.transport == "streamable-http"

    def test_transport_sse_inferred_from_url(self) -> None:
        defn = {"url": "http://localhost:3000/sse/events"}
        info = self.scanner.scan_definition(defn)
        assert info.transport == "sse"

    def test_transport_defaults_to_stdio_when_no_hints(self) -> None:
        # No command, no url, no explicit transport -> final fallback "stdio"
        defn = {"name": "bare-server", "version": "1.0"}
        info = self.scanner.scan_definition(defn)
        assert info.transport == "stdio"

    def test_tools_parsed(self) -> None:
        defn = {
            "command": "python",
            "tools": [
                {
                    "name": "read_file",
                    "description": "Reads a file from disk.",
                    "inputSchema": {"type": "object", "properties": {"path": {"type": "string"}}},
                }
            ],
        }
        info = self.scanner.scan_definition(defn)
        assert len(info.tools) == 1
        assert info.tools[0].name == "read_file"

    def test_tool_with_output_schema(self) -> None:
        defn = {
            "command": "python",
            "tools": [
                {
                    "name": "query",
                    "description": "Runs a query.",
                    "inputSchema": {"type": "object"},
                    "outputSchema": {"type": "array"},
                }
            ],
        }
        info = self.scanner.scan_definition(defn)
        assert info.tools[0].output_schema == {"type": "array"}

    def test_tool_with_snake_case_schema_key(self) -> None:
        defn = {
            "command": "python",
            "tools": [
                {
                    "name": "t",
                    "description": "d",
                    "input_schema": {"type": "object"},
                }
            ],
        }
        info = self.scanner.scan_definition(defn)
        assert info.tools[0].input_schema == {"type": "object"}

    def test_auth_required_explicit_flag(self) -> None:
        defn = {
            "command": "node",
            "tools": [{"name": "t", "description": "d", "inputSchema": {}, "auth_required": True}],
        }
        info = self.scanner.scan_definition(defn)
        assert info.tools[0].auth_required is True

    def test_auth_required_inferred_from_schema_property(self) -> None:
        defn = {
            "command": "node",
            "tools": [
                {
                    "name": "t",
                    "description": "d",
                    "inputSchema": {
                        "type": "object",
                        "properties": {"api_key": {"type": "string"}},
                    },
                }
            ],
        }
        info = self.scanner.scan_definition(defn)
        assert info.tools[0].auth_required is True

    def test_auth_not_required_by_default(self) -> None:
        defn = {
            "command": "node",
            "tools": [{"name": "t", "description": "d", "inputSchema": {"type": "object", "properties": {"path": {"type": "string"}}}}],
        }
        info = self.scanner.scan_definition(defn)
        assert info.tools[0].auth_required is False

    def test_resources_preserved(self) -> None:
        defn = {"command": "node", "resources": [{"uri": "file:///tmp"}]}
        info = self.scanner.scan_definition(defn)
        assert info.resources == [{"uri": "file:///tmp"}]

    def test_prompts_preserved(self) -> None:
        defn = {"command": "node", "prompts": [{"name": "greet"}]}
        info = self.scanner.scan_definition(defn)
        assert info.prompts == [{"name": "greet"}]

    def test_version_extracted(self) -> None:
        defn = {"command": "node", "version": "3.0.0"}
        info = self.scanner.scan_definition(defn)
        assert info.version == "3.0.0"

    def test_version_defaults_to_unknown(self) -> None:
        defn = {"command": "node"}
        info = self.scanner.scan_definition(defn)
        assert info.version == "unknown"

    def test_scanned_at_is_recent_utc(self) -> None:
        before = datetime.now(tz=timezone.utc)
        defn = {"command": "node"}
        info = self.scanner.scan_definition(defn)
        after = datetime.now(tz=timezone.utc)
        assert before <= info.scanned_at <= after

    def test_non_dict_tools_skipped(self) -> None:
        defn = {"command": "node", "tools": ["not-a-dict", 42]}
        info = self.scanner.scan_definition(defn)
        assert info.tools == []

    def test_tags_on_tool(self) -> None:
        defn = {
            "command": "node",
            "tools": [{"name": "t", "description": "d", "inputSchema": {}, "tags": ["search", "web"]}],
        }
        info = self.scanner.scan_definition(defn)
        assert info.tools[0].tags == ["search", "web"]


# ---------------------------------------------------------------------------
# MCPScanner — scan_file
# ---------------------------------------------------------------------------


class TestMCPScannerScanFile:
    def setup_method(self) -> None:
        self.scanner = MCPScanner()

    def _write(self, tmp_path: Path, name: str, content: str, suffix: str = ".json") -> Path:
        p = tmp_path / (name + suffix)
        p.write_text(content, encoding="utf-8")
        return p

    def test_claude_desktop_format(self, tmp_path: Path) -> None:
        config = {
            "mcpServers": {
                "filesystem": {
                    "command": "npx",
                    "args": ["-y", "@modelcontextprotocol/server-filesystem"],
                    "tools": [
                        {"name": "read_file", "description": "Reads a file", "inputSchema": {"type": "object"}},
                    ],
                },
                "fetch": {
                    "command": "npx",
                    "args": ["-y", "@modelcontextprotocol/server-fetch"],
                },
            }
        }
        path = self._write(tmp_path, "claude", json.dumps(config))
        servers = self.scanner.scan_file(path)
        assert len(servers) == 2
        names = {s.server_name for s in servers}
        assert "filesystem" in names
        assert "fetch" in names

    def test_servers_list_format(self, tmp_path: Path) -> None:
        config = {
            "servers": [
                {"name": "alpha", "command": "python"},
                {"name": "beta", "command": "node"},
            ]
        }
        path = self._write(tmp_path, "servers_list", json.dumps(config))
        servers = self.scanner.scan_file(path)
        assert len(servers) == 2

    def test_bare_list_format(self, tmp_path: Path) -> None:
        config = [
            {"name": "s1", "command": "python"},
            {"name": "s2", "command": "node"},
        ]
        path = self._write(tmp_path, "bare_list", json.dumps(config))
        servers = self.scanner.scan_file(path)
        assert len(servers) == 2

    def test_single_server_dict(self, tmp_path: Path) -> None:
        config = {"command": "python", "args": ["-m", "mcp"]}
        path = self._write(tmp_path, "single", json.dumps(config))
        servers = self.scanner.scan_file(path)
        assert len(servers) == 1

    def test_yaml_file_parsed(self, tmp_path: Path) -> None:
        yaml_text = """
mcpServers:
  yaml-server:
    command: python
    args: ["-m", "mcp_server"]
"""
        path = self._write(tmp_path, "config", yaml_text, suffix=".yaml")
        servers = self.scanner.scan_file(path)
        assert len(servers) == 1
        assert servers[0].server_name == "yaml-server"

    def test_yml_extension_parsed(self, tmp_path: Path) -> None:
        yaml_text = "mcpServers:\n  s:\n    command: python\n"
        path = self._write(tmp_path, "config", yaml_text, suffix=".yml")
        servers = self.scanner.scan_file(path)
        assert len(servers) == 1

    def test_empty_json_file(self, tmp_path: Path) -> None:
        path = self._write(tmp_path, "empty", "null")
        servers = self.scanner.scan_file(path)
        assert servers == []

    def test_empty_mcpservers_dict(self, tmp_path: Path) -> None:
        path = self._write(tmp_path, "no_servers", json.dumps({"mcpServers": {}}))
        servers = self.scanner.scan_file(path)
        assert servers == []

    def test_malformed_json_raises(self, tmp_path: Path) -> None:
        path = self._write(tmp_path, "bad", "{not valid json")
        with pytest.raises(ValueError):
            self.scanner.scan_file(path)

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "nonexistent.json"
        with pytest.raises(FileNotFoundError):
            self.scanner.scan_file(path)

    def test_mcp_servers_tools_extracted(self, tmp_path: Path) -> None:
        config = {
            "mcpServers": {
                "fs": {
                    "command": "node",
                    "tools": [
                        {"name": "write_file", "description": "Writes content to a file.", "inputSchema": {}},
                    ],
                }
            }
        }
        path = self._write(tmp_path, "with_tools", json.dumps(config))
        servers = self.scanner.scan_file(path)
        assert len(servers[0].tools) == 1
        assert servers[0].tools[0].name == "write_file"

    def test_cursor_style_mcp_servers(self, tmp_path: Path) -> None:
        # Cursor uses the same mcpServers format as Claude Desktop
        config = {
            "mcpServers": {
                "brave-search": {
                    "command": "npx",
                    "args": ["-y", "@modelcontextprotocol/server-brave-search"],
                    "env": {"BRAVE_API_KEY": "key123"},
                }
            }
        }
        path = self._write(tmp_path, "cursor", json.dumps(config))
        servers = self.scanner.scan_file(path)
        assert servers[0].server_name == "brave-search"

    def test_no_tools_key_gives_empty_list(self, tmp_path: Path) -> None:
        config = {"mcpServers": {"minimal": {"command": "python"}}}
        path = self._write(tmp_path, "no_tools", json.dumps(config))
        servers = self.scanner.scan_file(path)
        assert servers[0].tools == []

    def test_servers_resources_and_prompts(self, tmp_path: Path) -> None:
        config = {
            "mcpServers": {
                "rich": {
                    "command": "python",
                    "resources": [{"uri": "file:///data"}],
                    "prompts": [{"name": "summarize"}],
                }
            }
        }
        path = self._write(tmp_path, "rich", json.dumps(config))
        servers = self.scanner.scan_file(path)
        assert servers[0].resources == [{"uri": "file:///data"}]
        assert servers[0].prompts == [{"name": "summarize"}]

    def test_unknown_dict_shape_returns_empty(self, tmp_path: Path) -> None:
        # A JSON dict that has none of the expected shapes
        config = {"someOtherKey": {"deeply": "nested"}}
        path = self._write(tmp_path, "unknown_shape", json.dumps(config))
        servers = self.scanner.scan_file(path)
        assert servers == []


# ---------------------------------------------------------------------------
# MCPScanner — extract_capabilities
# ---------------------------------------------------------------------------


class TestMCPScannerExtractCapabilities:
    def setup_method(self) -> None:
        self.scanner = MCPScanner()

    def test_returns_one_dict_per_tool(self) -> None:
        server = _server(tools=[_tool("a"), _tool("b")])
        caps = self.scanner.extract_capabilities(server)
        assert len(caps) == 2

    def test_dict_keys_present(self) -> None:
        server = _server(tools=[_tool()])
        cap = self.scanner.extract_capabilities(server)[0]
        expected_keys = {"server_name", "tool_name", "category", "description", "input_schema", "output_schema", "auth_required", "tags"}
        assert expected_keys.issubset(cap.keys())

    def test_server_name_propagated(self) -> None:
        server = _server(server_name="my-srv", tools=[_tool()])
        cap = self.scanner.extract_capabilities(server)[0]
        assert cap["server_name"] == "my-srv"

    def test_tool_name_propagated(self) -> None:
        server = _server(tools=[_tool("read_file")])
        cap = self.scanner.extract_capabilities(server)[0]
        assert cap["tool_name"] == "read_file"

    def test_empty_tools_returns_empty(self) -> None:
        server = _server(tools=[])
        assert self.scanner.extract_capabilities(server) == []

    def test_category_included(self) -> None:
        server = _server(tools=[_tool("search_web", "Searches the web for information.")])
        cap = self.scanner.extract_capabilities(server)[0]
        assert cap["category"] == "search"

    def test_auth_required_propagated(self) -> None:
        server = _server(tools=[_tool(auth_required=True)])
        cap = self.scanner.extract_capabilities(server)[0]
        assert cap["auth_required"] is True

    def test_tags_copied(self) -> None:
        server = _server(tools=[_tool(tags=["a", "b"])])
        cap = self.scanner.extract_capabilities(server)[0]
        assert cap["tags"] == ["a", "b"]


# ---------------------------------------------------------------------------
# MCPScanner — categorize_tool
# ---------------------------------------------------------------------------


class TestCategorizeTool:
    def setup_method(self) -> None:
        self.scanner = MCPScanner()

    def _cat(self, name: str, desc: str = "", tags: list[str] | None = None) -> str:
        return self.scanner.categorize_tool(_tool(name=name, description=desc, tags=tags or []))

    def test_search_by_name(self) -> None:
        assert self._cat("web_search") == "search"

    def test_search_by_description(self) -> None:
        assert self._cat("tool", "Searches the internet for results.") == "search"

    def test_search_by_tag(self) -> None:
        assert self._cat("tool", "", ["search"]) == "search"

    def test_file_management_read(self) -> None:
        assert self._cat("read_file") == "file_management"

    def test_file_management_write(self) -> None:
        assert self._cat("write_file") == "file_management"

    def test_file_management_delete(self) -> None:
        assert self._cat("delete_file") == "file_management"

    def test_file_management_directory(self) -> None:
        assert self._cat("list_directory") == "file_management"

    def test_code_execute(self) -> None:
        assert self._cat("execute_code") == "code"

    def test_code_run(self) -> None:
        assert self._cat("run_script") == "code"

    def test_code_shell(self) -> None:
        assert self._cat("bash_command") == "code"

    def test_data_csv(self) -> None:
        assert self._cat("parse_csv") == "data"

    def test_data_database(self) -> None:
        assert self._cat("query_database") == "data"

    def test_communication_email(self) -> None:
        assert self._cat("send_email") == "communication"

    def test_communication_slack(self) -> None:
        assert self._cat("post_to_slack") == "communication"

    def test_utility_generate(self) -> None:
        assert self._cat("generate_uuid") == "utility"

    def test_utility_encode(self) -> None:
        assert self._cat("base64_encode") == "utility"

    def test_unknown_category(self) -> None:
        assert self._cat("xyzzy_frobnicate", "Does something mysterious with the blort.") == "unknown"

    def test_search_priority_over_file(self) -> None:
        # "search" should win over "file" when both present
        result = self._cat("search_files", "Searches through files in a directory.")
        assert result == "search"

    def test_case_insensitive_matching(self) -> None:
        assert self._cat("SEARCH_WEB") == "search"

    def test_description_wins_when_name_ambiguous(self) -> None:
        result = self._cat("do_thing", "Sends an email notification to recipients.")
        assert result == "communication"


# ---------------------------------------------------------------------------
# AutoRegistrar — compute_quality_score
# ---------------------------------------------------------------------------


class TestComputeQualityScore:
    def setup_method(self) -> None:
        self.registrar = AutoRegistrar()

    def test_empty_tool_scores_zero(self) -> None:
        tool = MCPToolDefinition(name="t", description="", input_schema={})
        assert self.registrar.compute_quality_score(tool) == pytest.approx(0.0)

    def test_description_only_gives_point_three(self) -> None:
        # Use bare constructor with empty input_schema to isolate description bonus
        tool = MCPToolDefinition(name="t", description="Short.", input_schema={})
        score = self.registrar.compute_quality_score(tool)
        assert score == pytest.approx(0.30)

    def test_long_description_adds_point_one_five(self) -> None:
        tool = MCPToolDefinition(
            name="t",
            description="This description is definitely longer than twenty characters.",
            input_schema={},
        )
        score = self.registrar.compute_quality_score(tool)
        assert score >= 0.45  # 0.30 + 0.15

    def test_input_schema_adds_point_two(self) -> None:
        tool = MCPToolDefinition(name="t", description="", input_schema={"type": "object"})
        score = self.registrar.compute_quality_score(tool)
        assert score == pytest.approx(0.20)

    def test_output_schema_adds_point_two(self) -> None:
        tool = MCPToolDefinition(name="t", description="", input_schema={}, output_schema={"type": "string"})
        score = self.registrar.compute_quality_score(tool)
        assert score == pytest.approx(0.20)

    def test_tags_add_point_one_five(self) -> None:
        tool = MCPToolDefinition(name="t", description="", input_schema={}, tags=["x"])
        score = self.registrar.compute_quality_score(tool)
        assert score == pytest.approx(0.15)

    def test_perfect_score_is_one(self) -> None:
        tool = MCPToolDefinition(
            name="t",
            description="A description that is very clearly longer than twenty chars.",
            input_schema={"type": "object"},
            output_schema={"type": "string"},
            tags=["tag"],
        )
        score = self.registrar.compute_quality_score(tool)
        assert score == pytest.approx(1.0)

    def test_score_never_exceeds_one(self) -> None:
        tool = MCPToolDefinition(
            name="t",
            description="x" * 100,
            input_schema={"a": "b"},
            output_schema={"c": "d"},
            tags=["t1", "t2"],
        )
        assert self.registrar.compute_quality_score(tool) <= 1.0

    def test_score_non_negative(self) -> None:
        tool = MCPToolDefinition(name="t", description="", input_schema={})
        assert self.registrar.compute_quality_score(tool) >= 0.0

    def test_exactly_twenty_chars_not_bonus(self) -> None:
        # Exactly 20 chars does NOT trigger the > 20 bonus
        desc = "x" * 20
        tool = MCPToolDefinition(name="t", description=desc, input_schema={})
        score = self.registrar.compute_quality_score(tool)
        assert score == pytest.approx(0.30)  # only base description bonus

    def test_twenty_one_chars_triggers_bonus(self) -> None:
        desc = "x" * 21
        tool = MCPToolDefinition(name="t", description=desc, input_schema={})
        score = self.registrar.compute_quality_score(tool)
        assert score == pytest.approx(0.45)


# ---------------------------------------------------------------------------
# AutoRegistrar — register_from_scan / register_all
# ---------------------------------------------------------------------------


class TestAutoRegistrarRegistration:
    def setup_method(self) -> None:
        self.registrar = AutoRegistrar()

    def test_register_from_scan_returns_list(self) -> None:
        server = _server(tools=[_tool("t1"), _tool("t2")])
        regs = self.registrar.register_from_scan(server)
        assert len(regs) == 2

    def test_register_from_scan_empty_tools(self) -> None:
        server = _server(tools=[])
        assert self.registrar.register_from_scan(server) == []

    def test_registration_source_server_set(self) -> None:
        server = _server(server_name="alpha", tools=[_tool()])
        reg = self.registrar.register_from_scan(server)[0]
        assert reg.source_server == "alpha"

    def test_registration_tool_name_set(self) -> None:
        server = _server(tools=[_tool("read_file")])
        reg = self.registrar.register_from_scan(server)[0]
        assert reg.tool_name == "read_file"

    def test_registration_category_set(self) -> None:
        server = _server(tools=[_tool("search_docs", "Searches documentation.")])
        reg = self.registrar.register_from_scan(server)[0]
        assert reg.category == "search"

    def test_registration_quality_score_set(self) -> None:
        server = _server(tools=[_tool()])
        reg = self.registrar.register_from_scan(server)[0]
        assert 0.0 <= reg.quality_score <= 1.0

    def test_registration_capability_id_is_16_chars(self) -> None:
        server = _server(tools=[_tool()])
        reg = self.registrar.register_from_scan(server)[0]
        assert len(reg.capability_id) == 16

    def test_capability_id_deterministic(self) -> None:
        server = _server(server_name="s", tools=[_tool("t")])
        reg1 = self.registrar.register_from_scan(server)[0]
        reg2 = self.registrar.register_from_scan(server)[0]
        assert reg1.capability_id == reg2.capability_id

    def test_capability_id_differs_for_different_names(self) -> None:
        server = _server(server_name="s", tools=[_tool("t1"), _tool("t2")])
        regs = self.registrar.register_from_scan(server)
        assert regs[0].capability_id != regs[1].capability_id

    def test_registration_is_frozen(self) -> None:
        server = _server(tools=[_tool()])
        reg = self.registrar.register_from_scan(server)[0]
        with pytest.raises((AttributeError, TypeError)):
            reg.tool_name = "other"  # type: ignore[misc]

    def test_register_all_flattens_servers(self) -> None:
        servers = [
            _server(server_name="s1", tools=[_tool("t1"), _tool("t2")]),
            _server(server_name="s2", tools=[_tool("t3")]),
        ]
        regs = self.registrar.register_all(servers)
        assert len(regs) == 3

    def test_register_all_empty_list(self) -> None:
        assert self.registrar.register_all([]) == []

    def test_registered_at_is_utc(self) -> None:
        server = _server(tools=[_tool()])
        reg = self.registrar.register_from_scan(server)[0]
        assert reg.registered_at.tzinfo is not None

    def test_input_schema_copied(self) -> None:
        schema = {"type": "object", "properties": {"x": {"type": "string"}}}
        server = _server(tools=[_tool(input_schema=schema)])
        reg = self.registrar.register_from_scan(server)[0]
        assert reg.input_schema == schema


# ---------------------------------------------------------------------------
# AutoRegistrar — deduplication
# ---------------------------------------------------------------------------


class TestAutoRegistrarDeduplicate:
    def setup_method(self) -> None:
        self.registrar = AutoRegistrar()

    def _make_reg(
        self,
        tool_name: str = "t",
        source_server: str = "s",
        quality_score: float = 0.5,
    ) -> CapabilityRegistration:
        return CapabilityRegistration(
            capability_id=f"{source_server}-{tool_name}-{quality_score}",
            source_server=source_server,
            tool_name=tool_name,
            category="utility",
            description="desc",
            input_schema={},
            registered_at=_now(),
            quality_score=quality_score,
        )

    def test_no_duplicates_unchanged(self) -> None:
        regs = [self._make_reg("t1"), self._make_reg("t2")]
        result = AutoRegistrar.deduplicate(regs)
        assert len(result) == 2

    def test_exact_duplicates_reduced_to_one(self) -> None:
        reg = self._make_reg()
        result = AutoRegistrar.deduplicate([reg, reg])
        assert len(result) == 1

    def test_higher_quality_wins(self) -> None:
        low = self._make_reg(quality_score=0.3)
        high = self._make_reg(quality_score=0.9)
        result = AutoRegistrar.deduplicate([low, high])
        assert len(result) == 1
        assert result[0].quality_score == pytest.approx(0.9)

    def test_different_servers_not_deduplicated(self) -> None:
        r1 = self._make_reg(tool_name="t", source_server="s1")
        r2 = self._make_reg(tool_name="t", source_server="s2")
        result = AutoRegistrar.deduplicate([r1, r2])
        assert len(result) == 2

    def test_empty_list(self) -> None:
        assert AutoRegistrar.deduplicate([]) == []

    def test_single_item_unchanged(self) -> None:
        reg = self._make_reg()
        result = AutoRegistrar.deduplicate([reg])
        assert result == [reg]

    def test_preserves_order_of_survivors(self) -> None:
        r1 = self._make_reg("a")
        r2 = self._make_reg("b")
        r3 = self._make_reg("c")
        result = AutoRegistrar.deduplicate([r1, r2, r3])
        assert [r.tool_name for r in result] == ["a", "b", "c"]

    def test_three_duplicates_one_survives(self) -> None:
        regs = [
            self._make_reg(quality_score=0.1),
            self._make_reg(quality_score=0.9),
            self._make_reg(quality_score=0.5),
        ]
        result = AutoRegistrar.deduplicate(regs)
        assert len(result) == 1
        assert result[0].quality_score == pytest.approx(0.9)


# ---------------------------------------------------------------------------
# AutoRegistrar — export / import round-trip
# ---------------------------------------------------------------------------


class TestAutoRegistrarRoundTrip:
    def setup_method(self) -> None:
        self.registrar = AutoRegistrar()

    def _build_regs(self, count: int = 2) -> list[CapabilityRegistration]:
        server = _server(tools=[_tool(f"t{i}", f"Tool {i} does useful things for testing.") for i in range(count)])
        return self.registrar.register_from_scan(server)

    def test_export_creates_file(self, tmp_path: Path) -> None:
        regs = self._build_regs()
        path = tmp_path / "registry.json"
        self.registrar.export_registry(regs, path)
        assert path.exists()

    def test_export_valid_json(self, tmp_path: Path) -> None:
        regs = self._build_regs()
        path = tmp_path / "registry.json"
        self.registrar.export_registry(regs, path)
        data = json.loads(path.read_text())
        assert isinstance(data, list)

    def test_import_returns_same_count(self, tmp_path: Path) -> None:
        regs = self._build_regs(3)
        path = tmp_path / "r.json"
        self.registrar.export_registry(regs, path)
        loaded = AutoRegistrar.import_registry(path)
        assert len(loaded) == 3

    def test_round_trip_preserves_tool_name(self, tmp_path: Path) -> None:
        regs = self._build_regs()
        path = tmp_path / "r.json"
        self.registrar.export_registry(regs, path)
        loaded = AutoRegistrar.import_registry(path)
        original_names = {r.tool_name for r in regs}
        loaded_names = {r.tool_name for r in loaded}
        assert original_names == loaded_names

    def test_round_trip_preserves_quality_score(self, tmp_path: Path) -> None:
        regs = self._build_regs()
        path = tmp_path / "r.json"
        self.registrar.export_registry(regs, path)
        loaded = AutoRegistrar.import_registry(path)
        for orig, load in zip(regs, loaded):
            assert orig.quality_score == pytest.approx(load.quality_score)

    def test_round_trip_preserves_category(self, tmp_path: Path) -> None:
        regs = self._build_regs()
        path = tmp_path / "r.json"
        self.registrar.export_registry(regs, path)
        loaded = AutoRegistrar.import_registry(path)
        for orig, load in zip(regs, loaded):
            assert orig.category == load.category

    def test_round_trip_preserves_registered_at(self, tmp_path: Path) -> None:
        regs = self._build_regs()
        path = tmp_path / "r.json"
        self.registrar.export_registry(regs, path)
        loaded = AutoRegistrar.import_registry(path)
        for orig, load in zip(regs, loaded):
            # Compare to the second (ISO format truncation)
            assert abs((orig.registered_at - load.registered_at).total_seconds()) < 1

    def test_import_empty_file(self, tmp_path: Path) -> None:
        path = tmp_path / "empty.json"
        path.write_text("[]", encoding="utf-8")
        loaded = AutoRegistrar.import_registry(path)
        assert loaded == []

    def test_import_invalid_json_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.json"
        path.write_text("{not valid}", encoding="utf-8")
        with pytest.raises(ValueError):
            AutoRegistrar.import_registry(path)

    def test_import_non_array_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "obj.json"
        path.write_text('{"key": "value"}', encoding="utf-8")
        with pytest.raises(ValueError):
            AutoRegistrar.import_registry(path)

    def test_import_missing_file_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "ghost.json"
        with pytest.raises(FileNotFoundError):
            AutoRegistrar.import_registry(path)

    def test_export_with_string_path(self, tmp_path: Path) -> None:
        regs = self._build_regs()
        path_str = str(tmp_path / "registry.json")
        self.registrar.export_registry(regs, path_str)
        assert Path(path_str).exists()

    def test_import_with_string_path(self, tmp_path: Path) -> None:
        regs = self._build_regs()
        path = tmp_path / "r.json"
        self.registrar.export_registry(regs, path)
        loaded = AutoRegistrar.import_registry(str(path))
        assert len(loaded) == len(regs)


# ---------------------------------------------------------------------------
# CLI — discover scan
# ---------------------------------------------------------------------------


class TestCLIDiscoverScan:
    def setup_method(self) -> None:
        from agent_marketplace.cli.main import cli
        self.runner = CliRunner()
        self.cli = cli

    def _write_config(self, tmp_path: Path, data: dict) -> Path:
        path = tmp_path / "config.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        return path

    def test_scan_basic_success(self, tmp_path: Path) -> None:
        config = {
            "mcpServers": {
                "fs": {
                    "command": "npx",
                    "tools": [
                        {"name": "read_file", "description": "Reads a file from disk.", "inputSchema": {}},
                    ],
                }
            }
        }
        path = self._write_config(tmp_path, config)
        result = self.runner.invoke(self.cli, ["discover", "scan", "--config", str(path)])
        assert result.exit_code == 0
        assert "fs" in result.output

    def test_scan_json_output(self, tmp_path: Path) -> None:
        config = {"mcpServers": {"srv": {"command": "python"}}}
        path = self._write_config(tmp_path, config)
        result = self.runner.invoke(self.cli, ["discover", "scan", "--config", str(path), "--json-output"])
        assert result.exit_code == 0
        parsed = json.loads(result.output)
        assert isinstance(parsed, list)

    def test_scan_empty_servers(self, tmp_path: Path) -> None:
        path = tmp_path / "empty.json"
        path.write_text("null", encoding="utf-8")
        result = self.runner.invoke(self.cli, ["discover", "scan", "--config", str(path)])
        assert result.exit_code == 0
        assert "No MCP servers" in result.output

    def test_scan_reports_tool_count(self, tmp_path: Path) -> None:
        config = {
            "mcpServers": {
                "srv": {
                    "command": "python",
                    "tools": [
                        {"name": "t1", "description": "Tool one.", "inputSchema": {}},
                        {"name": "t2", "description": "Tool two.", "inputSchema": {}},
                    ],
                }
            }
        }
        path = self._write_config(tmp_path, config)
        result = self.runner.invoke(self.cli, ["discover", "scan", "--config", str(path)])
        assert result.exit_code == 0
        assert "2" in result.output  # 2 tools

    def test_scan_missing_config_fails(self) -> None:
        result = self.runner.invoke(self.cli, ["discover", "scan", "--config", "/no/such/file.json"])
        assert result.exit_code != 0

    def test_scan_no_tools_shows_message(self, tmp_path: Path) -> None:
        config = {"mcpServers": {"empty-srv": {"command": "python"}}}
        path = self._write_config(tmp_path, config)
        result = self.runner.invoke(self.cli, ["discover", "scan", "--config", str(path)])
        assert result.exit_code == 0
        assert "no tools declared" in result.output


# ---------------------------------------------------------------------------
# CLI — discover register
# ---------------------------------------------------------------------------


class TestCLIDiscoverRegister:
    def setup_method(self) -> None:
        from agent_marketplace.cli.main import cli
        self.runner = CliRunner()
        self.cli = cli

    def test_register_creates_output_file(self, tmp_path: Path) -> None:
        config = {
            "mcpServers": {
                "srv": {
                    "command": "python",
                    "tools": [{"name": "t1", "description": "Does things.", "inputSchema": {}}],
                }
            }
        }
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps(config), encoding="utf-8")
        output_path = tmp_path / "registry.json"

        result = self.runner.invoke(self.cli, [
            "discover", "register",
            "--config", str(config_path),
            "--output", str(output_path),
        ])
        assert result.exit_code == 0
        assert output_path.exists()

    def test_register_output_is_valid_json(self, tmp_path: Path) -> None:
        config = {
            "mcpServers": {"srv": {"command": "python", "tools": [{"name": "t", "description": "d", "inputSchema": {}}]}}
        }
        config_path = tmp_path / "c.json"
        config_path.write_text(json.dumps(config), encoding="utf-8")
        output_path = tmp_path / "r.json"

        self.runner.invoke(self.cli, ["discover", "register", "--config", str(config_path), "--output", str(output_path)])
        data = json.loads(output_path.read_text())
        assert isinstance(data, list)

    def test_register_deduplicate_flag(self, tmp_path: Path) -> None:
        config = {
            "mcpServers": {
                "s1": {"command": "python", "tools": [{"name": "t", "description": "d", "inputSchema": {}}]},
                "s2": {"command": "node", "tools": [{"name": "t", "description": "d", "inputSchema": {}}]},
            }
        }
        config_path = tmp_path / "c.json"
        config_path.write_text(json.dumps(config), encoding="utf-8")
        output_path = tmp_path / "r.json"

        result = self.runner.invoke(self.cli, [
            "discover", "register",
            "--config", str(config_path),
            "--output", str(output_path),
            "--deduplicate",
        ])
        assert result.exit_code == 0
        # Different servers — should not be deduplicated (tool_name + source_server key)
        data = json.loads(output_path.read_text())
        assert len(data) == 2

    def test_register_missing_config_fails(self, tmp_path: Path) -> None:
        result = self.runner.invoke(self.cli, [
            "discover", "register",
            "--config", "/no/such/config.json",
            "--output", str(tmp_path / "r.json"),
        ])
        assert result.exit_code != 0


# ---------------------------------------------------------------------------
# CLI — discover list
# ---------------------------------------------------------------------------


class TestCLIDiscoverList:
    def setup_method(self) -> None:
        from agent_marketplace.cli.main import cli
        self.runner = CliRunner()
        self.cli = cli

    def _make_registry(self, tmp_path: Path, count: int = 2) -> Path:
        registrar = AutoRegistrar()
        scanner = MCPScanner()
        server = _server(
            tools=[
                _tool(f"tool_{i}", f"Tool {i} helps you do useful searching things efficiently.")
                for i in range(count)
            ]
        )
        regs = registrar.register_from_scan(server)
        path = tmp_path / "registry.json"
        registrar.export_registry(regs, path)
        return path

    def test_list_shows_tools(self, tmp_path: Path) -> None:
        path = self._make_registry(tmp_path)
        result = self.runner.invoke(self.cli, ["discover", "list", "--registry", str(path)])
        assert result.exit_code == 0
        assert "tool_0" in result.output

    def test_list_json_output(self, tmp_path: Path) -> None:
        path = self._make_registry(tmp_path)
        result = self.runner.invoke(self.cli, ["discover", "list", "--registry", str(path), "--json-output"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert isinstance(data, list)
        assert len(data) == 2

    def test_list_category_filter(self, tmp_path: Path) -> None:
        path = self._make_registry(tmp_path)
        # All tools will be categorised as "search" due to description keyword
        result = self.runner.invoke(self.cli, ["discover", "list", "--registry", str(path), "--category", "search"])
        assert result.exit_code == 0

    def test_list_empty_registry(self, tmp_path: Path) -> None:
        path = tmp_path / "empty.json"
        path.write_text("[]", encoding="utf-8")
        result = self.runner.invoke(self.cli, ["discover", "list", "--registry", str(path)])
        assert result.exit_code == 0
        assert "No capabilities" in result.output

    def test_list_missing_registry_fails(self) -> None:
        result = self.runner.invoke(self.cli, ["discover", "list", "--registry", "/no/such.json"])
        assert result.exit_code != 0

    def test_list_unknown_category_returns_empty_message(self, tmp_path: Path) -> None:
        path = self._make_registry(tmp_path)
        result = self.runner.invoke(self.cli, ["discover", "list", "--registry", str(path), "--category", "zzznomatch"])
        assert result.exit_code == 0
        assert "No capabilities" in result.output


# ---------------------------------------------------------------------------
# Integration — scan → register → list pipeline
# ---------------------------------------------------------------------------


class TestEndToEndPipeline:
    def test_scan_register_list_round_trip(self, tmp_path: Path) -> None:
        """Full pipeline: write config, scan, register, re-import."""
        config = {
            "mcpServers": {
                "brave-search": {
                    "command": "npx",
                    "args": ["-y", "@mcp/server-brave-search"],
                    "tools": [
                        {
                            "name": "brave_web_search",
                            "description": "Searches the web using Brave Search API.",
                            "inputSchema": {
                                "type": "object",
                                "required": ["query"],
                                "properties": {
                                    "query": {"type": "string"},
                                    "api_key": {"type": "string"},
                                },
                            },
                            "outputSchema": {"type": "array"},
                            "tags": ["search", "web"],
                        }
                    ],
                }
            }
        }
        config_path = tmp_path / "claude_desktop_config.json"
        config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

        scanner = MCPScanner()
        servers = scanner.scan_file(config_path)
        assert len(servers) == 1
        server = servers[0]
        assert server.server_name == "brave-search"
        assert len(server.tools) == 1
        assert server.tools[0].auth_required is True  # api_key in schema

        caps = scanner.extract_capabilities(server)
        assert caps[0]["category"] == "search"

        registrar = AutoRegistrar(scanner)
        regs = registrar.register_from_scan(server)
        assert len(regs) == 1
        reg = regs[0]
        assert reg.quality_score == pytest.approx(1.0)  # full score

        registry_path = tmp_path / "registry.json"
        registrar.export_registry(regs, registry_path)

        loaded = AutoRegistrar.import_registry(registry_path)
        assert len(loaded) == 1
        assert loaded[0].tool_name == "brave_web_search"
        assert loaded[0].source_server == "brave-search"

    def test_multiple_servers_aggregated(self, tmp_path: Path) -> None:
        config = {
            "mcpServers": {
                "filesystem": {
                    "command": "npx",
                    "tools": [
                        {"name": "read_file", "description": "Reads a file.", "inputSchema": {}},
                        {"name": "write_file", "description": "Writes a file.", "inputSchema": {}},
                    ],
                },
                "database": {
                    "command": "python",
                    "tools": [
                        {"name": "query_db", "description": "Queries the database.", "inputSchema": {}},
                    ],
                },
            }
        }
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps(config), encoding="utf-8")

        scanner = MCPScanner()
        servers = scanner.scan_file(config_path)
        registrar = AutoRegistrar(scanner)
        regs = registrar.register_all(servers)
        assert len(regs) == 3

        categories = {r.category for r in regs}
        assert "file_management" in categories
        assert "data" in categories
