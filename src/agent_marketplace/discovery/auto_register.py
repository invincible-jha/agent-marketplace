"""Auto-registration of MCP capabilities into a portable registry.

Converts ``MCPServerInfo`` objects (produced by ``MCPScanner``) into
``CapabilityRegistration`` records that can be persisted to JSON and
later reimported.

Example usage
-------------
::

    from agent_marketplace.discovery.mcp_scanner import MCPScanner
    from agent_marketplace.discovery.auto_register import AutoRegistrar

    scanner = MCPScanner()
    servers = scanner.scan_file(Path("claude_desktop_config.json"))

    registrar = AutoRegistrar(scanner)
    registrations = registrar.register_all(servers)
    registrar.export_registry(registrations, Path("registry.json"))

    # Later —
    loaded = AutoRegistrar.import_registry(Path("registry.json"))
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agent_marketplace.discovery.mcp_scanner import MCPScanner, MCPServerInfo, MCPToolDefinition


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CapabilityRegistration:
    """An immutable record describing a registered MCP tool capability.

    Attributes
    ----------
    capability_id:
        Deterministic identifier derived from ``source_server`` and
        ``tool_name`` (SHA-256 hex prefix).
    source_server:
        ``server_name`` from the originating ``MCPServerInfo``.
    tool_name:
        Name of the MCP tool.
    category:
        Heuristic category string (see ``MCPScanner.categorize_tool``).
    description:
        Tool description carried forward from the scan.
    input_schema:
        JSON Schema describing the tool's input.
    registered_at:
        UTC datetime when this record was created.
    quality_score:
        Documentation completeness score in [0.0, 1.0].
    """

    capability_id: str
    source_server: str
    tool_name: str
    category: str
    description: str
    input_schema: dict[str, Any]
    registered_at: datetime
    quality_score: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.input_schema, dict):
            object.__setattr__(self, "input_schema", {})


# ---------------------------------------------------------------------------
# Auto-registrar
# ---------------------------------------------------------------------------


class AutoRegistrar:
    """Converts ``MCPServerInfo`` objects into ``CapabilityRegistration`` records.

    Parameters
    ----------
    scanner:
        An ``MCPScanner`` instance used for categorisation.  When
        omitted a default scanner is created automatically.
    """

    def __init__(self, scanner: MCPScanner | None = None) -> None:
        self._scanner = scanner or MCPScanner()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def register_from_scan(self, server_info: MCPServerInfo) -> list[CapabilityRegistration]:
        """Convert all tools in *server_info* into ``CapabilityRegistration`` objects.

        Parameters
        ----------
        server_info:
            A scanned server.

        Returns
        -------
        list[CapabilityRegistration]
            One registration per tool.  Empty when the server has no tools.
        """
        registrations: list[CapabilityRegistration] = []
        for tool in server_info.tools:
            reg = self._build_registration(server_info.server_name, tool)
            registrations.append(reg)
        return registrations

    def register_all(self, servers: list[MCPServerInfo]) -> list[CapabilityRegistration]:
        """Register all tools from every server in *servers*.

        Parameters
        ----------
        servers:
            Sequence of scanned servers.

        Returns
        -------
        list[CapabilityRegistration]
            Flat list of all registrations across every server.
        """
        result: list[CapabilityRegistration] = []
        for server in servers:
            result.extend(self.register_from_scan(server))
        return result

    def compute_quality_score(self, tool: MCPToolDefinition) -> float:
        """Compute a documentation-completeness score for *tool*.

        Scoring breakdown
        -----------------
        +0.30 — has a non-empty description
        +0.20 — has a non-empty input schema
        +0.20 — has a non-None output schema
        +0.15 — has at least one tag
        +0.15 — description is longer than 20 characters

        Parameters
        ----------
        tool:
            The tool to score.

        Returns
        -------
        float
            Score in [0.0, 1.0].
        """
        score = 0.0

        if tool.description and tool.description.strip():
            score += 0.30

        if tool.input_schema:
            score += 0.20

        if tool.output_schema is not None:
            score += 0.20

        if tool.tags:
            score += 0.15

        if tool.description and len(tool.description.strip()) > 20:
            score += 0.15

        return round(min(score, 1.0), 10)

    def export_registry(
        self,
        registrations: list[CapabilityRegistration],
        path: Path | str,
    ) -> None:
        """Persist *registrations* to a JSON file at *path*.

        The file is human-readable with 2-space indentation.  Existing
        content is overwritten.

        Parameters
        ----------
        registrations:
            The records to serialise.
        path:
            Destination file path.
        """
        serialisable = [self._registration_to_dict(r) for r in registrations]
        Path(path).write_text(
            json.dumps(serialisable, indent=2, default=str),
            encoding="utf-8",
        )

    @staticmethod
    def import_registry(path: Path | str) -> list[CapabilityRegistration]:
        """Load ``CapabilityRegistration`` objects from a JSON file.

        Parameters
        ----------
        path:
            Path to a file previously written by ``export_registry``.

        Returns
        -------
        list[CapabilityRegistration]

        Raises
        ------
        ValueError
            If the file cannot be parsed or has an unexpected structure.
        FileNotFoundError
            If *path* does not exist.
        """
        raw = Path(path).read_text(encoding="utf-8")
        try:
            data: list[dict[str, Any]] = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Cannot parse registry file {path}: {exc}") from exc

        if not isinstance(data, list):
            raise ValueError(f"Registry file {path} must contain a JSON array at the top level.")

        registrations: list[CapabilityRegistration] = []
        for item in data:
            registrations.append(AutoRegistrar._dict_to_registration(item))
        return registrations

    @staticmethod
    def deduplicate(
        registrations: list[CapabilityRegistration],
    ) -> list[CapabilityRegistration]:
        """Remove duplicates keyed by ``(tool_name, source_server)``.

        When multiple registrations share the same key, the one with the
        highest ``quality_score`` is kept.  Ties are broken by
        ``registered_at`` (more recent wins).

        Parameters
        ----------
        registrations:
            Input list, possibly containing duplicates.

        Returns
        -------
        list[CapabilityRegistration]
            Deduplicated list preserving relative insertion order of the
            surviving records.
        """
        best: dict[tuple[str, str], CapabilityRegistration] = {}
        for reg in registrations:
            key = (reg.tool_name, reg.source_server)
            existing = best.get(key)
            if existing is None:
                best[key] = reg
            else:
                # Prefer higher quality score; fall back to more recent
                if (reg.quality_score, reg.registered_at) > (existing.quality_score, existing.registered_at):
                    best[key] = reg

        # Preserve order: iterate registrations and keep first occurrence of
        # each key as selected by the loop above.
        seen: set[tuple[str, str]] = set()
        ordered: list[CapabilityRegistration] = []
        for reg in registrations:
            key = (reg.tool_name, reg.source_server)
            selected = best[key]
            if selected.capability_id not in {r.capability_id for r in ordered}:
                if key not in seen:
                    ordered.append(selected)
                    seen.add(key)
        return ordered

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_registration(
        self, server_name: str, tool: MCPToolDefinition
    ) -> CapabilityRegistration:
        """Construct a single ``CapabilityRegistration`` from a tool."""
        cap_id = self._generate_id(server_name, tool.name)
        category = self._scanner.categorize_tool(tool)
        score = self.compute_quality_score(tool)

        return CapabilityRegistration(
            capability_id=cap_id,
            source_server=server_name,
            tool_name=tool.name,
            category=category,
            description=tool.description,
            input_schema=dict(tool.input_schema),
            registered_at=datetime.now(tz=timezone.utc),
            quality_score=score,
        )

    @staticmethod
    def _generate_id(server_name: str, tool_name: str) -> str:
        """Return a short deterministic capability ID."""
        raw = f"{server_name}::{tool_name}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    @staticmethod
    def _registration_to_dict(reg: CapabilityRegistration) -> dict[str, Any]:
        """Serialise a registration to a JSON-safe dict."""
        return {
            "capability_id": reg.capability_id,
            "source_server": reg.source_server,
            "tool_name": reg.tool_name,
            "category": reg.category,
            "description": reg.description,
            "input_schema": reg.input_schema,
            "registered_at": reg.registered_at.isoformat(),
            "quality_score": reg.quality_score,
        }

    @staticmethod
    def _dict_to_registration(data: dict[str, Any]) -> CapabilityRegistration:
        """Deserialise a dict (from JSON) into a ``CapabilityRegistration``."""
        try:
            registered_at = datetime.fromisoformat(data["registered_at"])
        except (KeyError, ValueError) as exc:
            raise ValueError(f"Invalid 'registered_at' field: {exc}") from exc

        return CapabilityRegistration(
            capability_id=data.get("capability_id", ""),
            source_server=data.get("source_server", ""),
            tool_name=data.get("tool_name", ""),
            category=data.get("category", "unknown"),
            description=data.get("description", ""),
            input_schema=data.get("input_schema", {}),
            registered_at=registered_at,
            quality_score=float(data.get("quality_score", 0.0)),
        )
