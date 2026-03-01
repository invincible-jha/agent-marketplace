"""agent-marketplace web dashboard subpackage.

Provides a self-contained HTTP dashboard for browsing capability
registrations, viewing agent cards, analytics, and searching
capabilities.  Requires no external dependencies beyond the Python
standard library.

Usage
-----
::

    from agent_marketplace.dashboard import DashboardServer
    from agent_marketplace.dashboard.server import DashboardDataSource

    source = DashboardDataSource()
    server = DashboardServer(data_source=source, host="127.0.0.1", port=8083)
    server.start()
"""
from __future__ import annotations

from agent_marketplace.dashboard.server import DashboardServer, DashboardDataSource

__all__ = [
    "DashboardServer",
    "DashboardDataSource",
]
