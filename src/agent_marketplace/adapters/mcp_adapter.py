"""MCP (Model Context Protocol) manifest adapter for agent-marketplace.

Imports an ``AgentCapability`` from an MCP tool manifest.  MCP manifests
describe tools exposed by an MCP server; each tool maps to one capability.

The adapter reads the manifest's ``name``, ``description``, ``inputSchema``,
and ``annotations`` fields to produce a well-formed ``AgentCapability``.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

try:
    import yaml as _yaml

    _YAML_AVAILABLE = True
except ImportError:  # pragma: no cover
    _YAML_AVAILABLE = False

if TYPE_CHECKING:
    pass


class MCPAdapter:
    """Converts an MCP tool manifest into one or more ``AgentCapability`` objects.

    An MCP manifest may describe a single tool (a flat dict with ``name``,
    ``description``, and ``inputSchema``) or a list of such tools under a
    ``tools`` key.  The adapter handles both shapes.

    Usage
    -----
    ::

        adapter = MCPAdapter(provider_name="Acme MCP Server")
        # Single tool
        capability = adapter.from_dict(manifest)
        # List of tools
        capabilities = adapter.from_dict_all(manifest)

    Parameters
    ----------
    provider_name:
        Name used for ``ProviderInfo.name``.
    server_url:
        Optional URL of the MCP server, recorded as the provider website.
    default_category:
        ``CapabilityCategory`` string for capabilities whose type cannot be
        inferred (default ``"automation"``).
    """

    def __init__(
        self,
        provider_name: str = "MCP Server",
        server_url: str = "",
        default_category: str = "automation",
    ) -> None:
        self._provider_name = provider_name
        self._server_url = server_url
        self._default_category = default_category

    # ------------------------------------------------------------------
    # Public import methods (single tool)
    # ------------------------------------------------------------------

    def from_dict(self, manifest: dict[str, object]) -> object:
        """Import the first (or only) tool from an MCP manifest dict.

        Parameters
        ----------
        manifest:
            Parsed MCP manifest as a Python dict.  May be a single-tool
            manifest or a multi-tool manifest with a ``tools`` list.

        Returns
        -------
        AgentCapability
            A populated capability derived from the first tool entry.

        Raises
        ------
        ValueError
            If no tool definition is found or ``name`` is missing.
        """
        tool = self._extract_first_tool(manifest)
        return self._tool_to_capability(tool)

    def from_json(self, json_text: str) -> object:
        """Import the first tool from a JSON MCP manifest string."""
        import json

        manifest: dict[str, object] = json.loads(json_text)
        return self.from_dict(manifest)

    def from_yaml(self, yaml_text: str) -> object:
        """Import the first tool from a YAML MCP manifest string.

        Raises
        ------
        ImportError
            If ``PyYAML`` is not installed.
        """
        if not _YAML_AVAILABLE:
            raise ImportError(
                "PyYAML is required for YAML manifest parsing. "
                "Install it with: pip install pyyaml"
            )
        manifest: dict[str, object] = _yaml.safe_load(yaml_text)
        return self.from_dict(manifest)

    def from_file(self, path: str) -> object:
        """Load and import the first tool from an MCP manifest file."""
        import json
        import pathlib

        file_path = pathlib.Path(path)
        raw_text = file_path.read_text(encoding="utf-8")

        if file_path.suffix.lower() in {".yaml", ".yml"}:
            return self.from_yaml(raw_text)
        return self.from_json(raw_text)

    # ------------------------------------------------------------------
    # Public import methods (all tools)
    # ------------------------------------------------------------------

    def from_dict_all(self, manifest: dict[str, object]) -> list[object]:
        """Import all tools from a multi-tool MCP manifest.

        Parameters
        ----------
        manifest:
            Parsed MCP manifest.  If it contains a ``tools`` key with a
            list, each entry is imported.  Otherwise the manifest itself
            is treated as a single tool.

        Returns
        -------
        list[AgentCapability]
            One capability per tool definition found.
        """
        tools = self._extract_all_tools(manifest)
        return [self._tool_to_capability(tool) for tool in tools]

    # ------------------------------------------------------------------
    # Conversion
    # ------------------------------------------------------------------

    def _tool_to_capability(self, tool: dict[str, object]) -> object:
        from agent_marketplace.schema.capability import (
            AgentCapability,
            PricingModel,
        )
        from agent_marketplace.schema.provider import ProviderInfo

        name: str = str(tool.get("name", "")).strip()
        if not name:
            raise ValueError("MCP tool manifest is missing 'name'.")

        description: str = str(tool.get("description", "")).strip()
        if not description:
            description = f"MCP tool: {name}"

        version: str = str(tool.get("version", "1.0.0")).strip() or "1.0.0"

        # Input types from inputSchema
        input_types = self._extract_input_types(tool)

        # Annotations (arbitrary metadata)
        annotations = tool.get("annotations")
        tags: list[str] = []
        if isinstance(annotations, dict):
            raw_tags = annotations.get("tags")
            if isinstance(raw_tags, list):
                tags = [str(t).strip() for t in raw_tags if str(t).strip()]

        # Mark as MCP tool
        if "mcp" not in {t.lower() for t in tags}:
            tags.append("mcp")

        category = self._infer_category(tags, name, description)

        provider = ProviderInfo(
            name=self._provider_name,
            website=self._server_url,
        )

        return AgentCapability(
            name=name,
            version=version,
            description=description,
            category=category,
            tags=tags,
            input_types=input_types or ["application/json"],
            output_type="application/json",
            pricing_model=PricingModel.FREE,
            provider=provider,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_first_tool(manifest: dict[str, object]) -> dict[str, object]:
        tools_raw = manifest.get("tools")
        if isinstance(tools_raw, list) and tools_raw:
            first = tools_raw[0]
            if isinstance(first, dict):
                return first  # type: ignore[return-value]
        # Single-tool manifest
        if "name" in manifest:
            return manifest  # type: ignore[return-value]
        raise ValueError(
            "MCP manifest does not contain a 'tools' list or a top-level 'name'."
        )

    @staticmethod
    def _extract_all_tools(manifest: dict[str, object]) -> list[dict[str, object]]:
        tools_raw = manifest.get("tools")
        if isinstance(tools_raw, list):
            return [t for t in tools_raw if isinstance(t, dict)]  # type: ignore[return-value]
        if "name" in manifest:
            return [manifest]  # type: ignore[return-value]
        return []

    @staticmethod
    def _extract_input_types(tool: dict[str, object]) -> list[str]:
        schema = tool.get("inputSchema")
        if not isinstance(schema, dict):
            return ["application/json"]
        # If schema is a JSON Schema object, the input is JSON
        schema_type = schema.get("type")
        if schema_type in ("object", "array", "string", "number", "integer", "boolean"):
            return ["application/json"]
        return ["application/json"]

    def _infer_category(
        self, tags: list[str], name: str, description: str
    ) -> object:
        from agent_marketplace.schema.capability import CapabilityCategory

        combined_lower = " ".join([*tags, name, description]).lower()
        mapping: dict[str, CapabilityCategory] = {
            "analysis": CapabilityCategory.ANALYSIS,
            "generation": CapabilityCategory.GENERATION,
            "transform": CapabilityCategory.TRANSFORMATION,
            "extract": CapabilityCategory.EXTRACTION,
            "interaction": CapabilityCategory.INTERACTION,
            "automation": CapabilityCategory.AUTOMATION,
            "evaluation": CapabilityCategory.EVALUATION,
            "research": CapabilityCategory.RESEARCH,
            "reasoning": CapabilityCategory.REASONING,
        }
        for keyword, category in mapping.items():
            if keyword in combined_lower:
                return category
        try:
            return CapabilityCategory(self._default_category)
        except ValueError:
            return CapabilityCategory.AUTOMATION
