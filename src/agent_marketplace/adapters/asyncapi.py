"""AsyncAPI spec adapter for agent-marketplace.

Imports an ``AgentCapability`` from an AsyncAPI 2.x / 3.x specification.
AsyncAPI describes event-driven (pub/sub, message-broker) APIs, so the
derived capability is categorised as ``AUTOMATION`` by default and
``input_types`` / ``output_type`` are derived from the spec's channel
message schemas.
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


class AsyncAPIAdapter:
    """Converts an AsyncAPI 2.x/3.x specification into an ``AgentCapability``.

    The adapter is intentionally lenient: fields missing from the spec
    are replaced with sensible defaults so that import always succeeds when
    a valid ``info.title`` is present.

    Usage
    -----
    ::

        adapter = AsyncAPIAdapter(provider_name="Acme Events")
        capability = adapter.from_file("path/to/asyncapi.yaml")

    Parameters
    ----------
    provider_name:
        Fallback provider name used when the spec has no contact block.
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
        """Parse an AsyncAPI spec dict and return an ``AgentCapability``.

        Parameters
        ----------
        spec:
            Parsed AsyncAPI 2.x/3.x specification as a Python dict.

        Returns
        -------
        AgentCapability
            A populated capability instance derived from the spec.

        Raises
        ------
        ValueError
            If the spec is missing ``info.title``.
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
                "AsyncAPI spec is missing 'info.title'; cannot derive capability name."
            )

        version: str = str(info.get("version", "1.0.0")).strip() or "1.0.0"
        description: str = str(info.get("description", "")).strip()
        if not description:
            description = f"Async capability imported from AsyncAPI spec: {title}"

        # Contact / provider
        contact = self._extract_dict(info, "contact")
        provider_name = str(contact.get("name", self._provider_name)).strip()
        contact_email = str(contact.get("email", "")).strip()

        provider = ProviderInfo(
            name=provider_name or self._provider_name,
            contact_email=contact_email,
        )

        # Channels → derive message types
        input_types, output_type = self._extract_message_types(spec)

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

        # Mark as async / event-driven via tags
        if "async" not in {t.lower() for t in tags}:
            tags.append("async")
        if "event-driven" not in {t.lower() for t in tags}:
            tags.append("event-driven")

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
        """Parse an AsyncAPI spec from a JSON string."""
        import json

        spec: dict[str, object] = json.loads(json_text)
        return self.from_dict(spec)

    def from_yaml(self, yaml_text: str) -> object:
        """Parse an AsyncAPI spec from a YAML string.

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
        """Load and parse an AsyncAPI spec from a file path.

        Both ``.json`` and ``.yaml``/``.yml`` extensions are supported.
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

    @classmethod
    def _extract_message_types(
        cls, spec: dict[str, object]
    ) -> tuple[list[str], str]:
        """Scan channels for message content types (AsyncAPI 2.x and 3.x)."""
        channels = spec.get("channels")
        if not isinstance(channels, dict):
            return [], ""

        content_types: set[str] = set()

        for _channel_name, channel_obj in channels.items():
            if not isinstance(channel_obj, dict):
                continue

            # AsyncAPI 2.x: subscribe/publish → message → contentType
            for direction in ("subscribe", "publish"):
                direction_obj = cls._extract_dict(channel_obj, direction)
                msg = cls._extract_dict(direction_obj, "message")
                content_type = msg.get("contentType") or channel_obj.get("bindings", {})
                if isinstance(content_type, str) and content_type:
                    content_types.add(content_type)

            # AsyncAPI 3.x: messages map
            messages = channel_obj.get("messages")
            if isinstance(messages, dict):
                for _msg_name, msg_obj in messages.items():
                    if isinstance(msg_obj, dict):
                        ct = msg_obj.get("contentType")
                        if isinstance(ct, str) and ct:
                            content_types.add(ct)

        types_list = sorted(content_types) or ["application/json"]
        output_type = next(
            (t for t in types_list if "json" in t),
            types_list[0],
        )
        return types_list, output_type

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
