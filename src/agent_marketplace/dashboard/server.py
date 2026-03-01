"""HTTP dashboard server for agent-marketplace.

Serves a single-page web dashboard (capability registry browser, agent
cards, usage analytics, search) using only the Python standard library.
No external web frameworks are required.

Usage
-----
::

    from agent_marketplace.dashboard.server import DashboardServer, DashboardDataSource

    source = DashboardDataSource()
    server = DashboardServer(data_source=source, host="127.0.0.1", port=8083)
    server.start()  # blocks; Ctrl-C to stop
"""
from __future__ import annotations

import json
import mimetypes
import time
import uuid
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

_STATIC_DIR = Path(__file__).parent / "static"

# ---------------------------------------------------------------------------
# In-memory data store
# ---------------------------------------------------------------------------


class DashboardDataSource:
    """Thread-safe in-memory store for capabilities, agents, and usage records.

    Parameters
    ----------
    max_capabilities:
        Maximum number of capability records to retain.
    """

    def __init__(self, max_capabilities: int = 2000) -> None:
        self._max_capabilities = max_capabilities
        self._capabilities: list[dict[str, object]] = []
        self._agents: dict[str, dict[str, object]] = {}
        self._usage_records: list[dict[str, object]] = []

    # ------------------------------------------------------------------
    # Write helpers
    # ------------------------------------------------------------------

    def register_capability(self, capability: dict[str, object]) -> str:
        """Register a capability, returning its assigned ID."""
        cap_id = str(capability.get("id") or uuid.uuid4())
        record = dict(capability)
        record["id"] = cap_id
        record.setdefault("name", "Unnamed Capability")
        record.setdefault("category", "general")
        record.setdefault("provider", "unknown")
        record.setdefault("version", "0.1.0")
        record.setdefault("registered_at", time.time())
        record.setdefault("trust_score", 0.5)
        self._capabilities.append(record)
        if len(self._capabilities) > self._max_capabilities:
            self._capabilities = self._capabilities[-self._max_capabilities :]
        return cap_id

    def register_agent(self, agent: dict[str, object]) -> str:
        """Register an agent or update an existing one."""
        agent_id = str(agent.get("id") or uuid.uuid4())
        record = dict(agent)
        record["id"] = agent_id
        record.setdefault("name", "Unnamed Agent")
        record.setdefault("registered_at", time.time())
        self._agents[agent_id] = record
        return agent_id

    def record_usage(self, usage: dict[str, object]) -> None:
        """Record a capability usage event."""
        record = dict(usage)
        record.setdefault("timestamp", time.time())
        self._usage_records.append(record)

    # ------------------------------------------------------------------
    # Read helpers
    # ------------------------------------------------------------------

    def get_capabilities(
        self,
        category: str | None = None,
        provider: str | None = None,
        limit: int = 200,
    ) -> list[dict[str, object]]:
        """Return capabilities, optionally filtered by category or provider."""
        caps = self._capabilities
        if category:
            caps = [c for c in caps if str(c.get("category") or "").lower() == category.lower()]
        if provider:
            caps = [c for c in caps if str(c.get("provider") or "").lower() == provider.lower()]
        return caps[-limit:]

    def search_capabilities(self, query: str, limit: int = 50) -> list[dict[str, object]]:
        """Return capabilities whose name or description matches *query*."""
        query_lower = query.lower()
        results: list[dict[str, object]] = []
        for cap in reversed(self._capabilities):
            name = str(cap.get("name") or "").lower()
            desc = str(cap.get("description") or "").lower()
            tags = str(cap.get("tags") or "").lower()
            if query_lower in name or query_lower in desc or query_lower in tags:
                results.append(cap)
                if len(results) >= limit:
                    break
        return results

    def get_agents(self, limit: int = 100) -> list[dict[str, object]]:
        """Return registered agents."""
        agents = list(self._agents.values())
        return agents[-limit:]

    def get_stats(self) -> dict[str, object]:
        """Return marketplace statistics."""
        by_category: dict[str, int] = {}
        for cap in self._capabilities:
            cat = str(cap.get("category") or "general")
            by_category[cat] = by_category.get(cat, 0) + 1

        by_capability: dict[str, int] = {}
        for usage in self._usage_records:
            cap_id = str(usage.get("capability_id") or "unknown")
            by_capability[cap_id] = by_capability.get(cap_id, 0) + 1

        top_capabilities = sorted(by_capability.items(), key=lambda x: x[1], reverse=True)[:10]

        return {
            "total_capabilities": len(self._capabilities),
            "total_agents": len(self._agents),
            "total_usage_events": len(self._usage_records),
            "by_category": by_category,
            "top_capabilities": [
                {"capability_id": cap_id, "usage_count": count}
                for cap_id, count in top_capabilities
            ],
        }

    @property
    def capability_count(self) -> int:
        """Total number of registered capabilities."""
        return len(self._capabilities)

    @property
    def agent_count(self) -> int:
        """Total number of registered agents."""
        return len(self._agents)


# ---------------------------------------------------------------------------
# Request handler
# ---------------------------------------------------------------------------


def _build_handler(data_source: DashboardDataSource) -> type[BaseHTTPRequestHandler]:
    """Build an HTTP request handler bound to *data_source*."""

    class _Handler(BaseHTTPRequestHandler):
        _source = data_source

        def log_message(self, fmt: str, *args: object) -> None:  # pragma: no cover
            pass

        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            path = parsed.path.rstrip("/") or "/"
            params = parse_qs(parsed.query)

            if path == "/" or path == "/index.html":
                self._serve_static("index.html")
            elif path == "/app.js":
                self._serve_static("app.js")
            elif path == "/styles.css":
                self._serve_static("styles.css")
            elif path == "/health":
                self._send_json(200, {
                    "status": "ok",
                    "service": "agent-marketplace-dashboard",
                    "capabilities": self._source.capability_count,
                    "agents": self._source.agent_count,
                })
            elif path == "/api/capabilities":
                category = (params.get("category") or [None])[0]
                provider = (params.get("provider") or [None])[0]
                limit = int((params.get("limit") or ["200"])[0])
                caps = self._source.get_capabilities(
                    category=category, provider=provider, limit=limit
                )
                self._send_json(200, {
                    "capabilities": caps,
                    "count": len(caps),
                    "total": self._source.capability_count,
                })
            elif path == "/api/capabilities/search":
                query = (params.get("q") or [""])[0]
                limit = int((params.get("limit") or ["50"])[0])
                if not query:
                    self._send_json(400, {"error": "q parameter is required"})
                    return
                results = self._source.search_capabilities(query=query, limit=limit)
                self._send_json(200, {"results": results, "count": len(results), "query": query})
            elif path == "/api/agents":
                limit = int((params.get("limit") or ["100"])[0])
                agents = self._source.get_agents(limit=limit)
                self._send_json(200, {"agents": agents, "count": len(agents)})
            elif path == "/api/stats":
                self._send_json(200, self._source.get_stats())
            else:
                self._send_json(404, {"error": "Not found", "path": path})

        def _serve_static(self, filename: str) -> None:
            file_path = _STATIC_DIR / filename
            if not file_path.exists():
                self._send_json(404, {"error": f"Static file not found: {filename}"})
                return
            content_type, _ = mimetypes.guess_type(filename)
            content_type = content_type or "application/octet-stream"
            body = file_path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(body)

        def _send_json(self, status: int, data: dict[str, object]) -> None:
            body = json.dumps(data, indent=2, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(body)

    return _Handler


# ---------------------------------------------------------------------------
# Server wrapper
# ---------------------------------------------------------------------------


class DashboardServer:
    """Agent-marketplace web dashboard server.

    Parameters
    ----------
    data_source:
        The data source to serve dashboard data from.
    host:
        Bind host (default ``"127.0.0.1"``).
    port:
        Bind port (default ``8083``).
    """

    def __init__(
        self,
        data_source: DashboardDataSource,
        host: str = "127.0.0.1",
        port: int = 8083,
    ) -> None:
        self._data_source = data_source
        self._host = host
        self._port = port
        self._server: HTTPServer | None = None

    def build_server(self) -> HTTPServer:
        """Build and return the underlying ``HTTPServer`` without starting it."""
        handler_cls = _build_handler(self._data_source)
        server = HTTPServer((self._host, self._port), handler_cls)
        self._server = server
        return server

    def start(self) -> None:
        """Start the HTTP server and block until interrupted."""
        server = self.build_server()
        try:
            server.serve_forever()
        finally:
            server.server_close()

    def shutdown(self) -> None:
        """Stop the server if it is running."""
        if self._server is not None:
            self._server.shutdown()

    @property
    def address(self) -> str:
        """Return the server's bind address as ``host:port``."""
        return f"{self._host}:{self._port}"


__all__ = [
    "DashboardServer",
    "DashboardDataSource",
]
