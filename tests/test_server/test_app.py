"""Tests for agent_marketplace.server.app — HTTP handler integration."""
from __future__ import annotations

from http.server import HTTPServer

import pytest

from agent_marketplace.server import routes
from agent_marketplace.server.app import AgentMarketplaceHandler, create_server


_VALID_CAPABILITY = {
    "name": "data-extractor",
    "version": "2.0.0",
    "description": "Extracts structured data from unstructured text.",
    "category": "extraction",
    "tags": ["extraction", "nlp"],
    "input_types": ["text/plain"],
    "output_type": "application/json",
    "pricing_model": "free",
    "cost": 0.0,
    "provider": {"name": "TestProvider"},
}


@pytest.fixture(autouse=True)
def reset_server_state() -> None:
    """Reset module-level state before each test."""
    routes.reset_state()


class TestAgentMarketplaceHandlerHealth:
    def test_health_returns_ok(self) -> None:
        status, data = routes.handle_health()
        assert status == 200
        assert data["status"] == "ok"

    def test_health_service_name(self) -> None:
        status, data = routes.handle_health()
        assert data["service"] == "agent-marketplace"

    def test_health_zero_count_initially(self) -> None:
        status, data = routes.handle_health()
        assert data["capability_count"] == 0


class TestAgentMarketplaceHandlerRegister:
    def test_registers_valid_capability(self) -> None:
        status, data = routes.handle_register(_VALID_CAPABILITY)
        assert status == 201
        assert "capability_id" in data
        assert data["registered"] is True

    def test_register_and_health_count_updates(self) -> None:
        routes.handle_register(_VALID_CAPABILITY)

        _, health_data = routes.handle_health()
        assert health_data["capability_count"] == 1

    def test_register_multiple_capabilities(self) -> None:
        for i in range(3):
            cap = dict(_VALID_CAPABILITY)
            cap["name"] = f"tool-{i}"
            routes.handle_register(cap)

        _, health_data = routes.handle_health()
        assert health_data["capability_count"] == 3


class TestAgentMarketplaceHandlerSearch:
    def test_search_empty_results(self) -> None:
        status, data = routes.handle_search(keyword="absolutely-nothing")
        assert status == 200
        assert data["total"] == 0

    def test_search_finds_by_name(self) -> None:
        routes.handle_register(_VALID_CAPABILITY)
        status, data = routes.handle_search(keyword="extractor")
        assert status == 200
        assert data["total"] >= 1

    def test_search_finds_by_tag(self) -> None:
        routes.handle_register(_VALID_CAPABILITY)
        status, data = routes.handle_search(keyword="nlp")
        assert status == 200
        assert data["total"] >= 1

    def test_search_returns_capability_fields(self) -> None:
        routes.handle_register(_VALID_CAPABILITY)
        status, data = routes.handle_search(keyword="extractor")
        assert status == 200
        if data["results"]:
            cap = data["results"][0]
            assert "capability_id" in cap
            assert "name" in cap
            assert "version" in cap

    def test_search_limit_and_offset(self) -> None:
        for i in range(5):
            cap = dict(_VALID_CAPABILITY)
            cap["name"] = f"search-cap-{i}"
            routes.handle_register(cap)

        status, data = routes.handle_search(keyword="search-cap", limit=2, offset=0)
        assert status == 200
        assert data["limit"] == 2


class TestAgentMarketplaceHandlerGetCapability:
    def test_get_registered_capability(self) -> None:
        _, registered = routes.handle_register(_VALID_CAPABILITY)
        capability_id = registered["capability_id"]

        status, data = routes.handle_get_capability(capability_id)
        assert status == 200
        assert data["capability_id"] == capability_id

    def test_get_nonexistent_capability(self) -> None:
        status, data = routes.handle_get_capability("does-not-exist-id")
        assert status == 404

    def test_get_capability_returns_correct_name(self) -> None:
        _, registered = routes.handle_register(_VALID_CAPABILITY)
        capability_id = registered["capability_id"]

        status, data = routes.handle_get_capability(capability_id)
        assert status == 200
        assert data["name"] == "data-extractor"


class TestCreateServer:
    def test_create_server_returns_http_server(self) -> None:
        server = create_server(host="127.0.0.1", port=0)
        try:
            assert isinstance(server, HTTPServer)
        finally:
            server.server_close()

    def test_create_server_uses_correct_handler(self) -> None:
        server = create_server(host="127.0.0.1", port=0)
        try:
            assert server.RequestHandlerClass is AgentMarketplaceHandler
        finally:
            server.server_close()


class TestServerModels:
    def test_register_capability_request(self) -> None:
        from agent_marketplace.server.models import RegisterCapabilityRequest

        req = RegisterCapabilityRequest(
            name="test",
            version="1.0",
            description="Test capability",
            category="analysis",
        )
        assert req.pricing_model == "free"
        assert req.cost == 0.0

    def test_capability_response(self) -> None:
        from agent_marketplace.server.models import CapabilityResponse

        resp = CapabilityResponse(
            capability_id="cap-1",
            name="test-cap",
            version="1.0",
            description="Test",
            category="analysis",
            output_type="text/plain",
            pricing_model="free",
            cost=0.0,
            trust_level=0.0,
        )
        assert resp.capability_id == "cap-1"
        assert resp.tags == []

    def test_search_response_fields(self) -> None:
        from agent_marketplace.server.models import SearchResponse

        resp = SearchResponse(query="test", total=0)
        assert resp.query == "test"
        assert resp.limit == 20

    def test_health_response_defaults(self) -> None:
        from agent_marketplace.server.models import HealthResponse

        resp = HealthResponse()
        assert resp.status == "ok"
        assert resp.service == "agent-marketplace"
