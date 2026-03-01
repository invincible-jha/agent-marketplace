"""Tests for agent_marketplace.dashboard.server."""
from __future__ import annotations

import io
import json
from http.server import HTTPServer
from unittest.mock import MagicMock

import pytest

from agent_marketplace.dashboard.server import (
    DashboardDataSource,
    DashboardServer,
    _build_handler,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_source() -> DashboardDataSource:
    source = DashboardDataSource()
    source.register_capability({
        "name": "Text Summarizer",
        "category": "nlp",
        "provider": "openai-corp",
        "version": "1.0.0",
        "trust_score": 0.92,
        "tags": "summarize text nlp",
    })
    source.register_capability({
        "name": "Image Classifier",
        "category": "vision",
        "provider": "vision-ai",
        "version": "2.1.0",
        "trust_score": 0.85,
        "tags": "image classify vision",
    })
    source.register_capability({
        "name": "Data Extractor",
        "category": "data",
        "provider": "openai-corp",
        "version": "0.5.0",
        "trust_score": 0.70,
    })
    source.register_agent({
        "name": "Research Agent",
        "capabilities": ["text-summarizer", "data-extractor"],
        "description": "Performs research tasks",
    })
    source.record_usage({"capability_id": "text-summarizer", "agent_id": "research-agent"})
    source.record_usage({"capability_id": "text-summarizer", "agent_id": "research-agent"})
    source.record_usage({"capability_id": "image-classifier", "agent_id": "vision-agent"})
    return source


def _call_get(path: str, source: DashboardDataSource | None = None) -> bytes:
    if source is None:
        source = _make_source()
    handler_cls = _build_handler(source)
    output = io.BytesIO()
    request = MagicMock()
    srv = MagicMock()
    srv.server_address = ("127.0.0.1", 8083)
    handler = handler_cls.__new__(handler_cls)
    handler.request = request
    handler.client_address = ("127.0.0.1", 9999)
    handler.server = srv
    handler.rfile = io.BytesIO(b"")
    handler.wfile = output
    handler.path = path
    # Required by BaseHTTPRequestHandler.send_response / send_header
    handler.request_version = "HTTP/1.1"
    handler.requestline = f"GET {path} HTTP/1.1"
    handler.close_connection = True
    handler.do_GET()
    return output.getvalue()


def _parse_json(path: str, source: DashboardDataSource | None = None) -> dict[str, object]:
    raw = _call_get(path, source)
    body_start = raw.find(b"\r\n\r\n")
    body = raw[body_start + 4:] if body_start != -1 else raw[raw.find(b"\n\n") + 2:]
    return json.loads(body)  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# DashboardDataSource unit tests
# ---------------------------------------------------------------------------


class TestDashboardDataSource:
    def test_register_capability_assigns_id(self) -> None:
        source = DashboardDataSource()
        cap_id = source.register_capability({"name": "Test Cap"})
        assert cap_id
        assert source.capability_count == 1

    def test_register_agent_assigns_id(self) -> None:
        source = DashboardDataSource()
        agent_id = source.register_agent({"name": "Agent A"})
        assert agent_id
        assert source.agent_count == 1

    def test_get_capabilities_all(self) -> None:
        source = _make_source()
        caps = source.get_capabilities()
        assert len(caps) == 3

    def test_get_capabilities_filter_by_category(self) -> None:
        source = _make_source()
        vision = source.get_capabilities(category="vision")
        assert len(vision) == 1
        assert all(c["category"] == "vision" for c in vision)

    def test_get_capabilities_filter_by_provider(self) -> None:
        source = _make_source()
        openai_caps = source.get_capabilities(provider="openai-corp")
        assert len(openai_caps) == 2

    def test_search_capabilities_by_name(self) -> None:
        source = _make_source()
        results = source.search_capabilities("summarizer")
        assert len(results) == 1
        assert "Text Summarizer" in str(results[0]["name"])

    def test_search_capabilities_by_tag(self) -> None:
        source = _make_source()
        results = source.search_capabilities("nlp")
        assert len(results) >= 1

    def test_search_capabilities_no_match(self) -> None:
        source = _make_source()
        results = source.search_capabilities("xyz-no-match-999")
        assert results == []

    def test_get_agents_all(self) -> None:
        source = _make_source()
        agents = source.get_agents()
        assert len(agents) == 1

    def test_get_stats_structure(self) -> None:
        source = _make_source()
        stats = source.get_stats()
        assert stats["total_capabilities"] == 3
        assert stats["total_agents"] == 1
        assert stats["total_usage_events"] == 3
        assert "by_category" in stats
        assert "top_capabilities" in stats

    def test_max_capabilities_eviction(self) -> None:
        source = DashboardDataSource(max_capabilities=2)
        for i in range(5):
            source.register_capability({"name": f"Cap {i}"})
        assert source.capability_count == 2


# ---------------------------------------------------------------------------
# Static files
# ---------------------------------------------------------------------------


class TestStaticFiles:
    def test_root_returns_html(self) -> None:
        raw = _call_get("/")
        assert b"html" in raw.lower()

    def test_styles_css(self) -> None:
        raw = _call_get("/styles.css")
        assert b"text/css" in raw or b"--bg" in raw

    def test_app_js(self) -> None:
        raw = _call_get("/app.js")
        assert b"javascript" in raw or b"function" in raw


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


class TestHealth:
    def test_health_ok(self) -> None:
        data = _parse_json("/health")
        assert data["status"] == "ok"

    def test_health_service_name(self) -> None:
        data = _parse_json("/health")
        assert data["service"] == "agent-marketplace-dashboard"

    def test_health_counts(self) -> None:
        data = _parse_json("/health")
        assert data["capabilities"] == 3
        assert data["agents"] == 1


# ---------------------------------------------------------------------------
# /api/capabilities
# ---------------------------------------------------------------------------


class TestCapabilitiesEndpoint:
    def test_returns_list(self) -> None:
        data = _parse_json("/api/capabilities")
        assert "capabilities" in data
        assert data["count"] == 3

    def test_filter_by_category(self) -> None:
        data = _parse_json("/api/capabilities?category=nlp")
        assert data["count"] == 1

    def test_filter_by_provider(self) -> None:
        data = _parse_json("/api/capabilities?provider=openai-corp")
        assert data["count"] == 2

    def test_total_unfiltered_reported(self) -> None:
        data = _parse_json("/api/capabilities?category=vision")
        assert data["total"] == 3


# ---------------------------------------------------------------------------
# /api/capabilities/search
# ---------------------------------------------------------------------------


class TestSearchEndpoint:
    def test_search_by_name(self) -> None:
        data = _parse_json("/api/capabilities/search?q=classifier")
        assert data["count"] >= 1

    def test_search_no_query_returns_400(self) -> None:
        raw = _call_get("/api/capabilities/search?q=")
        assert b"400" in raw

    def test_search_no_match(self) -> None:
        data = _parse_json("/api/capabilities/search?q=xyzzy-no-match")
        assert data["count"] == 0


# ---------------------------------------------------------------------------
# /api/agents
# ---------------------------------------------------------------------------


class TestAgentsEndpoint:
    def test_agents_returns_list(self) -> None:
        data = _parse_json("/api/agents")
        assert "agents" in data
        assert data["count"] == 1

    def test_agent_has_name(self) -> None:
        data = _parse_json("/api/agents")
        assert data["agents"][0]["name"] == "Research Agent"


# ---------------------------------------------------------------------------
# /api/stats
# ---------------------------------------------------------------------------


class TestStatsEndpoint:
    def test_stats_total_capabilities(self) -> None:
        data = _parse_json("/api/stats")
        assert data["total_capabilities"] == 3

    def test_stats_by_category(self) -> None:
        data = _parse_json("/api/stats")
        by_cat = data["by_category"]
        assert by_cat["nlp"] == 1
        assert by_cat["vision"] == 1

    def test_stats_top_capabilities(self) -> None:
        data = _parse_json("/api/stats")
        top = data["top_capabilities"]
        assert isinstance(top, list)


# ---------------------------------------------------------------------------
# 404
# ---------------------------------------------------------------------------


class TestNotFound:
    def test_unknown_path(self) -> None:
        raw = _call_get("/api/unknown")
        assert b"404" in raw


# ---------------------------------------------------------------------------
# DashboardServer
# ---------------------------------------------------------------------------


class TestDashboardServer:
    def test_instantiation(self) -> None:
        source = DashboardDataSource()
        server = DashboardServer(data_source=source)
        assert server.address == "127.0.0.1:8083"

    def test_build_server_returns_http_server(self) -> None:
        source = DashboardDataSource()
        server = DashboardServer(data_source=source, port=0)
        http_server = server.build_server()
        try:
            assert isinstance(http_server, HTTPServer)
        finally:
            http_server.server_close()

    def test_shutdown_noop(self) -> None:
        source = DashboardDataSource()
        server = DashboardServer(data_source=source)
        server.shutdown()  # must not raise
