"""Framework-agnostic marketplace API route handlers.

``MarketplaceAPI`` exposes the full marketplace surface as plain Python
methods.  Each method accepts a request dictionary and returns a
JSON-serializable response dictionary.  This design allows the same
handlers to be wired into any HTTP framework (FastAPI, Flask, WSGI, etc.)
or called directly in tests without starting a server.

Typical usage with a custom router::

    from agent_marketplace.registry.memory_store import MemoryStore
    from agent_marketplace.server.api import MarketplaceAPI

    store = MemoryStore()
    api = MarketplaceAPI(store=store)

    # In a FastAPI route:
    @app.post("/capabilities")
    async def register(body: dict) -> dict:
        return api.register_capability(body)
"""
from __future__ import annotations

from dataclasses import asdict

from agent_marketplace.registry.store import RegistryStore, SearchQuery
from agent_marketplace.schema.capability import AgentCapability
from agent_marketplace.schema.validator import SchemaValidator


def _error(message: str, code: int = 400) -> dict[str, object]:
    return {"ok": False, "error": message, "code": code}


def _ok(data: object, **extra: object) -> dict[str, object]:
    result: dict[str, object] = {"ok": True, "data": data}
    result.update(extra)
    return result


class MarketplaceAPI:
    """Route handlers for the agent-marketplace HTTP API.

    All methods accept and return plain Python dicts (JSON-serializable).
    Framework-specific glue (request parsing, serialization, status codes)
    is left to the caller.

    Parameters
    ----------
    store:
        The registry backend to use for persistence.
    validator:
        Optional custom schema validator.  A default ``SchemaValidator``
        is created when omitted.
    """

    def __init__(
        self,
        store: RegistryStore,
        validator: SchemaValidator | None = None,
    ) -> None:
        self._store = store
        self._validator = validator or SchemaValidator()

    # ------------------------------------------------------------------
    # Capability CRUD
    # ------------------------------------------------------------------

    def register_capability(self, body: dict[str, object]) -> dict[str, object]:
        """Register a new capability from a raw request body dict.

        Parameters
        ----------
        body:
            JSON-decoded request body.  Must satisfy the ``AgentCapability``
            schema.

        Returns
        -------
        dict
            ``{"ok": True, "data": {"capability_id": ...}}`` on success, or
            ``{"ok": False, "error": "...", "code": 400}`` on validation failure.
        """
        try:
            capability = AgentCapability.model_validate(body)
        except Exception as exc:  # noqa: BLE001
            return _error(f"Schema validation failed: {exc}")

        validation_result = self._validator.validate(capability)
        if not validation_result.valid:
            return _error(
                f"Business rule validation failed: {'; '.join(validation_result.errors)}"
            )

        try:
            self._store.register(capability)
        except ValueError as exc:
            return _error(str(exc), code=409)

        return _ok(
            {"capability_id": capability.capability_id},
            warnings=validation_result.warnings,
        )

    def get_capability(self, capability_id: str) -> dict[str, object]:
        """Retrieve a single capability by ID.

        Parameters
        ----------
        capability_id:
            The capability's unique identifier.
        """
        try:
            capability = self._store.get(capability_id)
        except KeyError:
            return _error(f"Capability {capability_id!r} not found.", code=404)
        return _ok(capability.to_dict())

    def update_capability(
        self, capability_id: str, body: dict[str, object]
    ) -> dict[str, object]:
        """Update an existing capability.

        Parameters
        ----------
        capability_id:
            The capability to update.
        body:
            Updated capability data.
        """
        # Ensure the id in the path is injected into the body
        body.setdefault("capability_id", capability_id)

        try:
            capability = AgentCapability.model_validate(body)
        except Exception as exc:  # noqa: BLE001
            return _error(f"Schema validation failed: {exc}")

        if capability.capability_id != capability_id:
            return _error(
                "capability_id in body does not match path parameter.", code=422
            )

        try:
            self._store.update(capability)
        except KeyError:
            return _error(f"Capability {capability_id!r} not found.", code=404)

        return _ok({"capability_id": capability.capability_id})

    def delete_capability(self, capability_id: str) -> dict[str, object]:
        """Delete a capability by ID."""
        try:
            self._store.delete(capability_id)
        except KeyError:
            return _error(f"Capability {capability_id!r} not found.", code=404)
        return _ok({"deleted": capability_id})

    # ------------------------------------------------------------------
    # Search / list
    # ------------------------------------------------------------------

    def search_capabilities(self, params: dict[str, object]) -> dict[str, object]:
        """Search capabilities using query parameters.

        Parameters
        ----------
        params:
            Query parameters.  Recognised keys:

            - ``keyword`` (str)
            - ``category`` (str)
            - ``tags`` (list[str] or comma-separated str)
            - ``min_trust`` (float)
            - ``max_cost`` (float)
            - ``limit`` (int, default 20)
            - ``offset`` (int, default 0)

        Returns
        -------
        dict
            ``{"ok": True, "data": {"results": [...], "total": N}}``.
        """
        from agent_marketplace.schema.capability import CapabilityCategory

        keyword = str(params.get("keyword", ""))
        limit = int(params.get("limit", 20))  # type: ignore[arg-type]
        offset = int(params.get("offset", 0))  # type: ignore[arg-type]
        min_trust = float(params.get("min_trust", 0.0))  # type: ignore[arg-type]
        max_cost_raw = params.get("max_cost")
        max_cost = float(max_cost_raw) if max_cost_raw is not None else float("inf")  # type: ignore[arg-type]

        # Category
        category_value = str(params.get("category", "")).strip()
        category = None
        if category_value:
            try:
                category = CapabilityCategory(category_value)
            except ValueError:
                return _error(f"Unknown category {category_value!r}.")

        # Tags
        tags_raw = params.get("tags", [])
        tags: list[str] = []
        if isinstance(tags_raw, list):
            tags = [str(t) for t in tags_raw]
        elif isinstance(tags_raw, str) and tags_raw:
            tags = [t.strip() for t in tags_raw.split(",")]

        query = SearchQuery(
            keyword=keyword,
            category=category,
            tags=tags,
            min_trust=min_trust,
            max_cost=max_cost,
            limit=limit,
            offset=offset,
        )
        results = self._store.search(query)
        total = len(self._store.search(SearchQuery(
            keyword=keyword,
            category=category,
            tags=tags,
            min_trust=min_trust,
            max_cost=max_cost,
            limit=0,
        )))

        return _ok(
            {
                "results": [cap.to_dict() for cap in results],
                "total": total,
                "limit": limit,
                "offset": offset,
            }
        )

    def list_capabilities(self, params: dict[str, object] | None = None) -> dict[str, object]:
        """Return all registered capabilities with optional pagination.

        Parameters
        ----------
        params:
            Optional dict with ``limit`` (default 50) and ``offset`` (default 0).
        """
        if params is None:
            params = {}
        limit = int(params.get("limit", 50))  # type: ignore[arg-type]
        offset = int(params.get("offset", 0))  # type: ignore[arg-type]

        all_capabilities = self._store.list_all()
        total = len(all_capabilities)
        page = all_capabilities[offset : offset + limit] if limit > 0 else all_capabilities[offset:]

        return _ok(
            {
                "results": [cap.to_dict() for cap in page],
                "total": total,
                "limit": limit,
                "offset": offset,
            }
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @property
    def store(self) -> RegistryStore:
        """Return the underlying registry store."""
        return self._store
