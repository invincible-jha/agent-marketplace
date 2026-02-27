"""MCP server capability scanner for agent-marketplace.

Parses MCP server definition files (JSON/YAML) in the format used by
Claude Desktop, Cursor, and similar clients, then extracts structured
capability metadata from each server's tool list.

Supported config layouts
------------------------
- ``mcpServers`` dict  (Claude Desktop / Cursor format)
- Bare ``servers`` list
- Single server definition dict

Example usage
-------------
::

    from pathlib import Path
    from agent_marketplace.discovery.mcp_scanner import MCPScanner

    scanner = MCPScanner()
    servers = scanner.scan_file(Path("~/.config/claude/claude_desktop_config.json"))
    for server in servers:
        caps = scanner.extract_capabilities(server)
        print(server.server_name, len(caps))
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import yaml


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MCPToolDefinition:
    """Immutable description of a single MCP tool exposed by a server.

    Attributes
    ----------
    name:
        Tool name as declared by the MCP server.
    description:
        Human-readable description of what the tool does.
    input_schema:
        JSON Schema object describing the tool's input parameters.
    output_schema:
        Optional JSON Schema object describing the tool's return value.
    auth_required:
        Whether calling this tool requires authentication credentials.
    tags:
        Categorisation tags inferred or explicitly provided.
    """

    name: str
    description: str
    input_schema: dict[str, object]
    output_schema: dict[str, object] | None = None
    auth_required: bool = False
    tags: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        # frozen=True means we cannot assign directly; use object.__setattr__
        # for mutable defaults that must be copied to avoid aliasing.
        if not isinstance(self.tags, list):
            object.__setattr__(self, "tags", list(self.tags))


@dataclass(frozen=True)
class MCPServerInfo:
    """Immutable snapshot of an MCP server scanned at a point in time.

    Attributes
    ----------
    server_name:
        Logical name of the server (key in the mcpServers dict, or
        derived from the server definition).
    version:
        Server version string, defaulting to ``"unknown"`` when not declared.
    transport:
        Transport mechanism: ``"stdio"``, ``"sse"``, or
        ``"streamable-http"``.
    tools:
        All tools discovered on this server.
    resources:
        Raw resource definitions (schema varies by server).
    prompts:
        Raw prompt template definitions.
    scanned_at:
        UTC datetime when the scan was performed.
    """

    server_name: str
    version: str
    transport: str
    tools: list[MCPToolDefinition]
    resources: list[dict[str, object]]
    prompts: list[dict[str, object]]
    scanned_at: datetime

    def __post_init__(self) -> None:
        # Ensure lists are not aliased across instances (frozen prevents
        # direct assignment, so object.__setattr__ is used).
        if not isinstance(self.tools, list):
            object.__setattr__(self, "tools", list(self.tools))
        if not isinstance(self.resources, list):
            object.__setattr__(self, "resources", list(self.resources))
        if not isinstance(self.prompts, list):
            object.__setattr__(self, "prompts", list(self.prompts))


# ---------------------------------------------------------------------------
# Categorisation heuristics
# ---------------------------------------------------------------------------

# Maps tool-category label -> list of lowercase keyword fragments that
# suggest the tool belongs to that category.  Earlier entries take priority.
_CATEGORY_KEYWORDS: list[tuple[str, list[str]]] = [
    ("search", ["search", "find", "lookup", "fetch", "retrieve", "browse"]),
    ("file_management", ["file", "read", "write", "delete", "move", "copy", "fs", "directory", "path", "upload", "download"]),
    ("code", ["code", "execute", "run", "compile", "lint", "debug", "script", "bash", "shell", "eval", "python", "javascript"]),
    ("data", ["data", "csv", "json", "xml", "parse", "transform", "convert", "schema", "sql", "database", "table", "spreadsheet", "query"]),
    ("communication", ["email", "send", "message", "chat", "notify", "slack", "webhook", "sms", "calendar", "meet"]),
    ("utility", ["format", "generate", "random", "hash", "encode", "decode", "base64", "uuid", "timestamp", "date", "time", "log", "help"]),
]

_KNOWN_AUTH_KEYWORDS = frozenset(["token", "key", "secret", "auth", "credential", "oauth", "api_key", "bearer"])


# ---------------------------------------------------------------------------
# Scanner
# ---------------------------------------------------------------------------


class MCPScanner:
    """Parses MCP server configuration files and extracts capability metadata.

    All public methods are stateless and thread-safe.

    Example
    -------
    ::

        scanner = MCPScanner()
        infos = scanner.scan_file(Path("claude_desktop_config.json"))
        for info in infos:
            print(info.server_name, [t.name for t in info.tools])
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def scan_definition(self, definition: dict[str, object]) -> MCPServerInfo:
        """Parse a single MCP server definition dict into ``MCPServerInfo``.

        The *definition* dict is one entry from a ``mcpServers`` mapping or
        an element of a ``servers`` list.  It typically has the shape::

            {
                "command": "npx",
                "args": ["-y", "@modelcontextprotocol/server-filesystem"],
                "env": {"API_KEY": "..."},
                "tools": [...],   # optional — pre-populated tool list
                "transport": "stdio"
            }

        Parameters
        ----------
        definition:
            Raw server definition dict.

        Returns
        -------
        MCPServerInfo
            Parsed server info with scanned_at set to the current UTC time.
        """
        server_name: str = definition.get("name", definition.get("server_name", "unnamed"))
        version: str = definition.get("version", "unknown")
        transport: str = self._extract_transport(definition)

        raw_tools: list[dict[str, object]] = definition.get("tools", [])
        tools = [self._parse_tool(t) for t in raw_tools if isinstance(t, dict)]

        resources: list[dict[str, object]] = definition.get("resources", [])
        prompts: list[dict[str, object]] = definition.get("prompts", [])

        return MCPServerInfo(
            server_name=server_name,
            version=version,
            transport=transport,
            tools=tools,
            resources=list(resources) if isinstance(resources, list) else [],
            prompts=list(prompts) if isinstance(prompts, list) else [],
            scanned_at=datetime.now(tz=timezone.utc),
        )

    def scan_file(self, path: Path) -> list[MCPServerInfo]:
        """Read a JSON or YAML config file and extract all MCP server definitions.

        Handles the following top-level shapes:

        - ``{"mcpServers": {"name": {...}, ...}}``  — Claude Desktop / Cursor
        - ``{"servers": [{...}, ...]}``              — list-style config
        - A bare dict with a ``command`` or ``transport`` key            — single server
        - A bare list of server dicts

        Parameters
        ----------
        path:
            Path to the config file. Extension (``.yaml``/``.yml``/``.json``)
            determines the parser; all other extensions are tried as JSON first.

        Returns
        -------
        list[MCPServerInfo]
            One entry per server definition found. Empty when no servers are
            found or the file is empty.

        Raises
        ------
        ValueError
            If the file cannot be parsed as JSON or YAML.
        FileNotFoundError
            If *path* does not exist.
        """
        raw_text = Path(path).read_text(encoding="utf-8")

        try:
            if path.suffix.lower() in {".yaml", ".yml"}:
                data: object = yaml.safe_load(raw_text)
            else:
                try:
                    data = json.loads(raw_text)
                except json.JSONDecodeError:
                    data = yaml.safe_load(raw_text)
        except Exception as exc:
            raise ValueError(f"Cannot parse {path}: {exc}") from exc

        return self._extract_servers(data)

    def extract_capabilities(self, server: MCPServerInfo) -> list[dict[str, object]]:
        """Extract capability metadata dicts from a scanned ``MCPServerInfo``.

        Each dict in the returned list has the shape::

            {
                "server_name": str,
                "tool_name":   str,
                "category":    str,
                "description": str,
                "input_schema": dict,
                "output_schema": dict | None,
                "auth_required": bool,
                "tags": list[str],
            }

        Parameters
        ----------
        server:
            A previously scanned server info object.

        Returns
        -------
        list[dict[str, object]]
            One entry per tool.  Empty when the server has no tools.
        """
        capabilities: list[dict[str, object]] = []
        for tool in server.tools:
            capabilities.append(
                {
                    "server_name": server.server_name,
                    "tool_name": tool.name,
                    "category": self.categorize_tool(tool),
                    "description": tool.description,
                    "input_schema": tool.input_schema,
                    "output_schema": tool.output_schema,
                    "auth_required": tool.auth_required,
                    "tags": list(tool.tags),
                }
            )
        return capabilities

    def categorize_tool(self, tool: MCPToolDefinition) -> str:
        """Assign a category string to a tool using heuristic keyword matching.

        Categories (in priority order):
        ``search``, ``file_management``, ``code``, ``data``,
        ``communication``, ``utility``, ``unknown``.

        The heuristic checks the tool name and description (lower-cased,
        split on non-alphanumeric boundaries) for keyword membership.

        Parameters
        ----------
        tool:
            The tool to categorize.

        Returns
        -------
        str
            One of the seven category strings listed above.
        """
        haystack = f"{tool.name} {tool.description}".lower()
        # Tokenise: split on whitespace and non-alphanumeric characters
        tokens = set(re.split(r"[\W_]+", haystack)) - {""}

        for category, keywords in _CATEGORY_KEYWORDS:
            for kw in keywords:
                # Token-level match (exact whole-token) OR multi-word keyword
                # phrase contained in the haystack (for compound keywords).
                if kw in tokens or (len(kw) > 4 and kw in haystack):
                    return category

        # Fall through to tags (token-level only)
        tag_tokens = set(re.split(r"[\W_]+", " ".join(tool.tags).lower())) - {""}
        for category, keywords in _CATEGORY_KEYWORDS:
            for kw in keywords:
                if kw in tag_tokens:
                    return category

        return "unknown"

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _extract_servers(self, data: object) -> list[MCPServerInfo]:
        """Dispatch on the top-level data shape and return all server infos."""
        if data is None:
            return []

        # Shape: {"mcpServers": {"name": {...}, ...}}
        if isinstance(data, dict) and "mcpServers" in data:
            mcp_servers = data["mcpServers"]
            if isinstance(mcp_servers, dict):
                return [
                    self._scan_named(name, defn)
                    for name, defn in mcp_servers.items()
                    if isinstance(defn, dict)
                ]

        # Shape: {"servers": [{...}, ...]}
        if isinstance(data, dict) and "servers" in data:
            servers = data["servers"]
            if isinstance(servers, list):
                return [
                    self.scan_definition(s)
                    for s in servers
                    if isinstance(s, dict)
                ]

        # Shape: bare list of server dicts
        if isinstance(data, list):
            return [
                self.scan_definition(s)
                for s in data
                if isinstance(s, dict)
            ]

        # Shape: single server definition dict
        if isinstance(data, dict) and ("command" in data or "transport" in data or "url" in data):
            return [self.scan_definition(data)]

        return []

    def _scan_named(self, name: str, definition: dict[str, object]) -> MCPServerInfo:
        """Scan a server definition that has an explicit name key."""
        enriched = dict(definition)
        if "name" not in enriched and "server_name" not in enriched:
            enriched["name"] = name
        return self.scan_definition(enriched)

    @staticmethod
    def _extract_transport(definition: dict[str, object]) -> str:
        """Infer the transport mechanism from the server definition."""
        transport = definition.get("transport", "")
        if transport:
            return str(transport).lower()

        # Infer from other keys
        if "url" in definition:
            url = str(definition["url"]).lower()
            if "sse" in url:
                return "sse"
            return "streamable-http"

        command = definition.get("command", "")
        if command:
            return "stdio"

        return "stdio"

    @staticmethod
    def _parse_tool(raw: dict[str, object]) -> MCPToolDefinition:
        """Convert a raw tool dict from the config into an ``MCPToolDefinition``."""
        name: str = raw.get("name", "")
        description: str = raw.get("description", "")
        input_schema: dict[str, object] = raw.get("inputSchema", raw.get("input_schema", {}))
        output_schema: dict[str, object] | None = raw.get("outputSchema", raw.get("output_schema", None))
        tags: list[str] = raw.get("tags", [])
        if not isinstance(tags, list):
            tags = []

        # Detect auth requirement from schema property names or env hints
        auth_required = MCPScanner._detect_auth_required(raw, input_schema)

        return MCPToolDefinition(
            name=name,
            description=description,
            input_schema=dict(input_schema) if input_schema else {},
            output_schema=dict(output_schema) if output_schema else None,
            auth_required=auth_required,
            tags=list(tags),
        )

    @staticmethod
    def _detect_auth_required(raw: dict[str, object], input_schema: dict[str, object]) -> bool:
        """Return True when the tool definition implies authentication is needed."""
        # Explicit flag
        if raw.get("auth_required", False):
            return True

        # Inspect input schema property names for auth hints
        properties = input_schema.get("properties", {})
        for prop_name in properties:
            if any(kw in prop_name.lower() for kw in _KNOWN_AUTH_KEYWORDS):
                return True

        return False
