"""OpenAPI spec adapter for agent-marketplace.

Imports an ``AgentCapability`` from an OpenAPI 3.x specification document.
The adapter reads the spec's ``info`` block, ``paths``, and ``servers`` to
derive capability metadata without requiring any third-party OpenAPI library.

Only ``application/json`` request and response bodies are inspected; other
content types are recorded as plain strings.
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


class OpenAPIAdapter:
    """Converts an OpenAPI 3.x specification into an ``AgentCapability``.

    The adapter supports both JSON and YAML spec formats.  The ``info``
    block is mapped to capability metadata; ``paths`` entries are
    summarised into ``input_types`` and ``output_type``; the first
    server URL is recorded in the provider's ``website`` field.

    Usage
    -----
    ::

        adapter = OpenAPIAdapter(provider_name="Acme Corp")
        capability = adapter.from_file("path/to/openapi.yaml")

    Parameters
    ----------
    provider_name:
        Name used for the ``ProviderInfo.name`` field when the spec does
        not contain contact information.
    default_category:
        ``CapabilityCategory`` string used when no category can be inferred
        from the spec (default ``"automation"``).
    """

    def __init__(
        self,
        provider_name: str = "Unknown Provider",
        default_category: str = "automation",
    ) -> None:
        self._provider_name = provider_name
        self._default_category = default_category

    # ------------------------------------------------------------------
    # Public import methods
    # ------------------------------------------------------------------

    def from_dict(self, spec: dict[str, object]) -> object:
        """Parse an OpenAPI spec dict and return an ``AgentCapability``.

        Parameters
        ----------
        spec:
            Parsed OpenAPI 3.x specification as a Python dict.

        Returns
        -------
        AgentCapability
            A populated capability instance derived from the spec.

        Raises
        ------
        ValueError
            If the spec is missing required fields (``info.title``).
        """
        from agent_marketplace.schema.capability import (
            AgentCapability,
            CapabilityCategory,
            PricingModel,
        )
        from agent_marketplace.schema.provider import ProviderInfo

        info = self._extract_dict(spec, "info")
        title: str = str(info.get("title", "")).strip()
        if not title:
            raise ValueError(
                "OpenAPI spec is missing 'info.title'; cannot derive capability name."
            )

        version: str = str(info.get("version", "1.0.0")).strip() or "1.0.0"
        description: str = str(info.get("description", "")).strip()
        if not description:
            description = f"Capability imported from OpenAPI spec: {title}"

        # Provider info
        contact = self._extract_dict(info, "contact")
        provider_name = str(contact.get("name", self._provider_name)).strip()
        contact_email = str(contact.get("email", "")).strip()
        website_url = self._extract_first_server_url(spec)

        provider = ProviderInfo(
            name=provider_name or self._provider_name,
            contact_email=contact_email,
            website=website_url,
        )

        # Input / output types from paths
        input_types, output_type = self._extract_io_types(spec)

        # Tags from spec-level tags list
        tags_raw: list[object] = self._extract_list(spec, "tags")
        tags: list[str] = []
        for tag_entry in tags_raw:
            if isinstance(tag_entry, dict):
                tag_name = str(tag_entry.get("name", "")).strip()
                if tag_name:
                    tags.append(tag_name)
            elif isinstance(tag_entry, str):
                tags.append(tag_entry.strip())

        # Category — attempt to infer from tags
        category = self._infer_category(tags)

        return AgentCapability(
            name=title,
            version=version,
            description=description,
            category=category,
            tags=tags,
            input_types=input_types or ["application/json"],
            output_type=output_type or "application/json",
            pricing_model=PricingModel.FREE,
            provider=provider,
        )

    def from_json(self, json_text: str) -> object:
        """Parse an OpenAPI spec from a JSON string.

        Parameters
        ----------
        json_text:
            Raw JSON content of the OpenAPI specification.
        """
        import json

        spec: dict[str, object] = json.loads(json_text)
        return self.from_dict(spec)

    def from_yaml(self, yaml_text: str) -> object:
        """Parse an OpenAPI spec from a YAML string.

        Parameters
        ----------
        yaml_text:
            Raw YAML content of the OpenAPI specification.

        Raises
        ------
        ImportError
            If ``PyYAML`` is not installed.
        """
        if not _YAML_AVAILABLE:
            raise ImportError(
                "PyYAML is required for YAML spec parsing. "
                "Install it with: pip install pyyaml"
            )
        spec: dict[str, object] = _yaml.safe_load(yaml_text)
        return self.from_dict(spec)

    def from_file(self, path: str) -> object:
        """Load and parse an OpenAPI spec from a file path.

        Both ``.json`` and ``.yaml``/``.yml`` extensions are supported.

        Parameters
        ----------
        path:
            Filesystem path to the spec file.
        """
        import json
        import pathlib

        file_path = pathlib.Path(path)
        raw_text = file_path.read_text(encoding="utf-8")

        if file_path.suffix.lower() in {".yaml", ".yml"}:
            return self.from_yaml(raw_text)
        return self.from_json(raw_text)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_dict(mapping: dict[str, object], key: str) -> dict[str, object]:
        value = mapping.get(key)
        if isinstance(value, dict):
            return value  # type: ignore[return-value]
        return {}

    @staticmethod
    def _extract_list(mapping: dict[str, object], key: str) -> list[object]:
        value = mapping.get(key)
        if isinstance(value, list):
            return value
        return []

    @staticmethod
    def _extract_first_server_url(spec: dict[str, object]) -> str:
        servers = spec.get("servers")
        if isinstance(servers, list) and servers:
            first = servers[0]
            if isinstance(first, dict):
                url = first.get("url", "")
                return str(url).strip()
        return ""

    @staticmethod
    def _extract_io_types(
        spec: dict[str, object],
    ) -> tuple[list[str], str]:
        """Scan paths for request/response content types."""
        paths = spec.get("paths")
        if not isinstance(paths, dict):
            return [], ""

        input_types: set[str] = set()
        output_types: set[str] = set()

        for _path, path_item in paths.items():
            if not isinstance(path_item, dict):
                continue
            for _method, operation in path_item.items():
                if not isinstance(operation, dict):
                    continue
                # Request body
                request_body = operation.get("requestBody")
                if isinstance(request_body, dict):
                    content = request_body.get("content")
                    if isinstance(content, dict):
                        input_types.update(content.keys())
                # Responses
                responses = operation.get("responses")
                if isinstance(responses, dict):
                    for _status, response in responses.items():
                        if isinstance(response, dict):
                            content = response.get("content")
                            if isinstance(content, dict):
                                output_types.update(content.keys())

        # Prefer JSON types; fall back to sorted set
        preferred_input = sorted(input_types) or []
        preferred_output = next(
            (t for t in ("application/json", "text/plain") if t in output_types),
            next(iter(sorted(output_types)), ""),
        )
        return preferred_input, preferred_output

    def _infer_category(self, tags: list[str]) -> object:
        from agent_marketplace.schema.capability import CapabilityCategory

        tag_lower = {t.lower() for t in tags}
        mapping: dict[str, CapabilityCategory] = {
            "analysis": CapabilityCategory.ANALYSIS,
            "generation": CapabilityCategory.GENERATION,
            "transform": CapabilityCategory.TRANSFORMATION,
            "extraction": CapabilityCategory.EXTRACTION,
            "interaction": CapabilityCategory.INTERACTION,
            "automation": CapabilityCategory.AUTOMATION,
            "evaluation": CapabilityCategory.EVALUATION,
            "research": CapabilityCategory.RESEARCH,
            "reasoning": CapabilityCategory.REASONING,
        }
        for keyword, category in mapping.items():
            if any(keyword in tag for tag in tag_lower):
                return category
        try:
            return CapabilityCategory(self._default_category)
        except ValueError:
            return CapabilityCategory.AUTOMATION
