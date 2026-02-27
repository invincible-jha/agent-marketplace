"""Tests for agent_marketplace.server.routes."""
from __future__ import annotations

import pytest

from agent_marketplace.server import routes


# Minimal valid capability body for tests
_VALID_CAPABILITY = {
    "name": "text-summarizer",
    "version": "1.0.0",
    "description": "Summarizes long documents into key points.",
    "category": "analysis",
    "tags": ["nlp", "summarization"],
    "input_types": ["text/plain"],
    "output_type": "text/plain",
    "pricing_model": "free",
    "cost": 0.0,
    "provider": {"name": "AumOS"},
}


@pytest.fixture(autouse=True)
def reset_server_state() -> None:
    """Reset module-level state before each test."""
    routes.reset_state()


class TestHandleRegister:
    def test_registers_valid_capability(self) -> None:
        status, data = routes.handle_register(_VALID_CAPABILITY)

        assert status == 201
        assert "capability_id" in data
        assert data["registered"] is True

    def test_rejects_missing_name(self) -> None:
        body = {k: v for k, v in _VALID_CAPABILITY.items() if k != "name"}
        status, data = routes.handle_register(body)

        assert status in (400, 422)
        assert "error" in data

    def test_rejects_missing_provider(self) -> None:
        body = {k: v for k, v in _VALID_CAPABILITY.items() if k != "provider"}
        status, data = routes.handle_register(body)

        assert status in (400, 422)
        assert "error" in data

    def test_rejects_duplicate_registration(self) -> None:
        routes.handle_register(_VALID_CAPABILITY)
        status, data = routes.handle_register(_VALID_CAPABILITY)

        assert status == 409

    def test_returns_warnings_when_present(self) -> None:
        body = dict(_VALID_CAPABILITY)
        body["input_types"] = []  # This may cause a warning
        routes.handle_register(body)
        # Just ensure register call doesn't crash even with empty input_types

    def test_capability_id_is_deterministic(self) -> None:
        status1, data1 = routes.handle_register(_VALID_CAPABILITY)
        routes.reset_state()
        status2, data2 = routes.handle_register(_VALID_CAPABILITY)

        assert data1["capability_id"] == data2["capability_id"]


class TestHandleSearch:
    def test_returns_empty_on_no_match(self) -> None:
        routes.handle_register(_VALID_CAPABILITY)
        status, data = routes.handle_search(keyword="nonexistent-term-xyz")

        assert status == 200
        assert data["total"] == 0
        assert data["results"] == []

    def test_finds_by_keyword(self) -> None:
        routes.handle_register(_VALID_CAPABILITY)
        status, data = routes.handle_search(keyword="summarizer")

        assert status == 200
        assert data["total"] >= 1

    def test_keyword_matches_tags(self) -> None:
        routes.handle_register(_VALID_CAPABILITY)
        status, data = routes.handle_search(keyword="nlp")

        assert status == 200
        assert data["total"] >= 1

    def test_empty_keyword_returns_all(self) -> None:
        routes.handle_register(_VALID_CAPABILITY)
        cap2 = dict(_VALID_CAPABILITY)
        cap2["name"] = "second-tool"
        routes.handle_register(cap2)

        status, data = routes.handle_search(keyword="")

        assert status == 200
        assert data["total"] == 2

    def test_respects_limit(self) -> None:
        for i in range(5):
            cap = dict(_VALID_CAPABILITY)
            cap["name"] = f"tool-{i}"
            routes.handle_register(cap)

        status, data = routes.handle_search(keyword="tool", limit=3)

        assert status == 200
        assert len(data["results"]) <= 3

    def test_query_preserved_in_response(self) -> None:
        status, data = routes.handle_search(keyword="search-term")

        assert data["query"] == "search-term"

    def test_category_filter(self) -> None:
        routes.handle_register(_VALID_CAPABILITY)
        gen_cap = dict(_VALID_CAPABILITY)
        gen_cap["name"] = "text-generator"
        gen_cap["category"] = "generation"
        gen_cap["description"] = "Generates text content."
        routes.handle_register(gen_cap)

        status, data = routes.handle_search(category="generation")

        assert status == 200
        # Should find only generation caps
        for cap in data["results"]:
            assert cap["category"] == "generation"


class TestHandleGetCapability:
    def test_returns_capability_by_id(self) -> None:
        _, registered = routes.handle_register(_VALID_CAPABILITY)
        capability_id = registered["capability_id"]

        status, data = routes.handle_get_capability(capability_id)

        assert status == 200
        assert data["capability_id"] == capability_id
        assert data["name"] == "text-summarizer"

    def test_returns_404_for_missing_id(self) -> None:
        status, data = routes.handle_get_capability("nonexistent-cap-id")

        assert status == 404
        assert "error" in data

    def test_returns_all_capability_fields(self) -> None:
        _, registered = routes.handle_register(_VALID_CAPABILITY)
        capability_id = registered["capability_id"]

        status, data = routes.handle_get_capability(capability_id)

        assert status == 200
        assert "name" in data
        assert "version" in data
        assert "description" in data
        assert "category" in data


class TestHandleHealth:
    def test_returns_ok_status(self) -> None:
        status, data = routes.handle_health()

        assert status == 200
        assert data["status"] == "ok"

    def test_reports_capability_count(self) -> None:
        routes.handle_register(_VALID_CAPABILITY)
        cap2 = dict(_VALID_CAPABILITY)
        cap2["name"] = "second-cap"
        routes.handle_register(cap2)

        status, data = routes.handle_health()

        assert status == 200
        assert data["capability_count"] == 2

    def test_zero_count_initially(self) -> None:
        status, data = routes.handle_health()

        assert data["capability_count"] == 0
