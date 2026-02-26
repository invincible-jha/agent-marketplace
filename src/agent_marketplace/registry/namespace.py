"""Namespace management for agent-marketplace registry.

Namespaces follow the convention ``org/agent/capability`` and allow
multiple organizations to share the same registry without key collisions.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

_VALID_SEGMENT_RE = re.compile(r"^[a-z0-9][a-z0-9_\-]*$")
_SEPARATOR = "/"


@dataclass(frozen=True)
class Namespace:
    """A hierarchical namespace identifier.

    Attributes
    ----------
    organization:
        Top-level organization or owner segment (e.g. ``"acme-corp"``).
    agent:
        Agent name segment (e.g. ``"research-assistant"``).
    capability:
        Capability leaf name (e.g. ``"web-search"``).
    """

    organization: str
    agent: str
    capability: str

    def __post_init__(self) -> None:
        for segment_name, segment_value in (
            ("organization", self.organization),
            ("agent", self.agent),
            ("capability", self.capability),
        ):
            if not _VALID_SEGMENT_RE.match(segment_value):
                raise ValueError(
                    f"Namespace segment {segment_name!r}={segment_value!r} is invalid. "
                    "Segments must start with a lowercase letter or digit and contain only "
                    "lowercase letters, digits, hyphens, or underscores."
                )

    @property
    def path(self) -> str:
        """Return the fully-qualified namespace path (``org/agent/capability``)."""
        return _SEPARATOR.join([self.organization, self.agent, self.capability])

    @classmethod
    def from_path(cls, path: str) -> "Namespace":
        """Parse a namespace from its string representation.

        Parameters
        ----------
        path:
            A ``"org/agent/capability"`` string.

        Returns
        -------
        Namespace

        Raises
        ------
        ValueError
            If the path does not have exactly three segments.
        """
        parts = path.split(_SEPARATOR)
        if len(parts) != 3:
            raise ValueError(
                f"Namespace path {path!r} must have exactly three segments "
                "in the form 'organization/agent/capability'."
            )
        return cls(organization=parts[0], agent=parts[1], capability=parts[2])

    def __str__(self) -> str:
        return self.path


class NamespaceManager:
    """Manages a collection of namespaces and provides capability_id mapping.

    The manager maintains a two-way mapping between ``Namespace`` objects
    and ``capability_id`` strings, allowing the registry to resolve
    namespace paths to storage identifiers.
    """

    def __init__(self) -> None:
        self._ns_to_id: dict[str, str] = {}
        self._id_to_ns: dict[str, str] = {}

    def register(self, namespace: Namespace, capability_id: str) -> None:
        """Associate a namespace with a capability_id.

        Parameters
        ----------
        namespace:
            The namespace to register.
        capability_id:
            The corresponding capability identifier.

        Raises
        ------
        ValueError
            If the namespace path is already registered.
        """
        ns_path = namespace.path
        if ns_path in self._ns_to_id:
            raise ValueError(
                f"Namespace {ns_path!r} is already registered "
                f"to capability_id {self._ns_to_id[ns_path]!r}."
            )
        self._ns_to_id[ns_path] = capability_id
        self._id_to_ns[capability_id] = ns_path

    def deregister(self, namespace: Namespace) -> None:
        """Remove a namespace mapping.

        Parameters
        ----------
        namespace:
            The namespace to remove.

        Raises
        ------
        KeyError
            If the namespace is not registered.
        """
        ns_path = namespace.path
        if ns_path not in self._ns_to_id:
            raise KeyError(f"Namespace {ns_path!r} is not registered.")
        capability_id = self._ns_to_id.pop(ns_path)
        self._id_to_ns.pop(capability_id, None)

    def resolve(self, namespace: Namespace) -> str:
        """Return the capability_id for a namespace.

        Raises
        ------
        KeyError
            If the namespace is not registered.
        """
        ns_path = namespace.path
        if ns_path not in self._ns_to_id:
            raise KeyError(f"Namespace {ns_path!r} is not registered.")
        return self._ns_to_id[ns_path]

    def reverse_resolve(self, capability_id: str) -> Namespace:
        """Return the Namespace for a capability_id.

        Raises
        ------
        KeyError
            If the capability_id has no namespace mapping.
        """
        if capability_id not in self._id_to_ns:
            raise KeyError(f"capability_id {capability_id!r} has no namespace mapping.")
        return Namespace.from_path(self._id_to_ns[capability_id])

    def list_namespaces(self) -> list[Namespace]:
        """Return all registered namespaces in sorted order."""
        return [Namespace.from_path(p) for p in sorted(self._ns_to_id.keys())]

    def list_by_org(self, organization: str) -> list[Namespace]:
        """Return all namespaces belonging to *organization*."""
        return [
            Namespace.from_path(p)
            for p in self._ns_to_id
            if p.startswith(f"{organization}/")
        ]
