"""Route handler functions for the agent-marketplace HTTP server.

Wraps the existing MarketplaceAPI for use by the stdlib HTTP handler.
Each function accepts parsed request data and returns a tuple of
(status_code, response_dict).
"""
from __future__ import annotations


from agent_marketplace.registry.memory_store import MemoryStore
from agent_marketplace.server.api import MarketplaceAPI
from agent_marketplace.server.health import HealthEndpoint
from agent_marketplace.server.models import (
    CapabilityResponse,
    ErrorResponse,
    HealthResponse,
    RegisterResponse,
    SearchResponse,
)


# Module-level shared state
_store: MemoryStore = MemoryStore()
_api: MarketplaceAPI = MarketplaceAPI(store=_store)
_health: HealthEndpoint = HealthEndpoint(registry_store=_store)


def reset_state() -> None:
    """Reset all shared state — used in tests and for clean restarts."""
    global _store, _api, _health
    _store = MemoryStore()
    _api = MarketplaceAPI(store=_store)
    _health = HealthEndpoint(registry_store=_store)


def _capability_dict_to_response(data: dict[str, object]) -> CapabilityResponse:
    """Convert a raw capability dict to a CapabilityResponse."""
    return CapabilityResponse(
        capability_id=data.get("capability_id", ""),
        name=data.get("name", ""),
        version=data.get("version", ""),
        description=data.get("description", ""),
        category=data.get("category", ""),
        tags=data.get("tags", []),
        input_types=data.get("input_types", []),
        output_type=data.get("output_type", ""),
        pricing_model=data.get("pricing_model", "free"),
        cost=data.get("cost", 0.0),
        trust_level=data.get("trust_level", 0.0),
        supported_languages=data.get("supported_languages", []),
        supported_frameworks=data.get("supported_frameworks", []),
    )


def handle_register(body: dict[str, object]) -> tuple[int, dict[str, object]]:
    """Handle POST /register.

    Parameters
    ----------
    body:
        Parsed JSON request body.

    Returns
    -------
    tuple[int, dict[str, object]]
        HTTP status code and response dictionary.
    """
    result = _api.register_capability(body)

    if not result.get("ok"):
        code = result.get("code", 400)
        return int(code), ErrorResponse(  # type: ignore[arg-type]
            error=str(result.get("error", "Registration failed")),
        ).model_dump()

    data = result.get("data", {})
    warnings = result.get("warnings", [])
    response = RegisterResponse(
        capability_id=str(data.get("capability_id", "")),  # type: ignore[union-attr]
        registered=True,
        warnings=list(warnings or []),  # type: ignore[arg-type]
    )
    return 201, response.model_dump()


def handle_search(
    keyword: str = "",
    category: str | None = None,
    limit: int = 20,
    offset: int = 0,
) -> tuple[int, dict[str, object]]:
    """Handle GET /search?q=...

    Parameters
    ----------
    keyword:
        Free-text search term.
    category:
        Optional category filter.
    limit:
        Maximum results to return.
    offset:
        Results to skip.

    Returns
    -------
    tuple[int, dict[str, object]]
        HTTP status code and response dictionary.
    """
    params: dict[str, object] = {
        "keyword": keyword,
        "limit": limit,
        "offset": offset,
    }
    if category:
        params["category"] = category

    result = _api.search_capabilities(params)

    if not result.get("ok"):
        code = result.get("code", 400)
        return int(code), ErrorResponse(  # type: ignore[arg-type]
            error=str(result.get("error", "Search failed")),
        ).model_dump()

    inner = result.get("data", {})
    raw_results = inner.get("results", [])  # type: ignore[union-attr]
    cap_responses = [_capability_dict_to_response(r) for r in raw_results]  # type: ignore[union-attr]

    response = SearchResponse(
        query=keyword,
        results=cap_responses,
        total=int(inner.get("total", 0)),  # type: ignore[union-attr,arg-type]
        limit=limit,
        offset=offset,
    )
    return 200, response.model_dump()


def handle_get_capability(capability_id: str) -> tuple[int, dict[str, object]]:
    """Handle GET /capabilities/{id}.

    Parameters
    ----------
    capability_id:
        The capability identifier from the URL path.

    Returns
    -------
    tuple[int, dict[str, object]]
        HTTP status code and response dictionary.
    """
    result = _api.get_capability(capability_id)

    if not result.get("ok"):
        code = result.get("code", 404)
        return int(code), ErrorResponse(  # type: ignore[arg-type]
            error=str(result.get("error", "Not found")),
        ).model_dump()

    data = result.get("data", {})
    response = _capability_dict_to_response(data)  # type: ignore[arg-type]
    return 200, response.model_dump()


def handle_health() -> tuple[int, dict[str, object]]:
    """Handle GET /health.

    Returns
    -------
    tuple[int, dict[str, object]]
        HTTP status code and response dictionary.
    """
    check = _health.check()
    capability_count = check.get("capabilities", 0)
    if isinstance(capability_count, str):
        capability_count = 0
    response = HealthResponse(
        status=str(check.get("status", "ok")),
        capability_count=int(capability_count),  # type: ignore[arg-type]
    )
    return 200, response.model_dump()


__all__ = [
    "reset_state",
    "handle_register",
    "handle_search",
    "handle_get_capability",
    "handle_health",
]
