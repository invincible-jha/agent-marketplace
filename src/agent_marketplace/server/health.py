"""Health endpoint for agent-marketplace server.

Provides a lightweight ``HealthEndpoint`` class that returns structured
status information about the marketplace service without requiring any
HTTP framework dependency.
"""
from __future__ import annotations

import time
from datetime import datetime, timezone

from agent_marketplace import __version__


class HealthEndpoint:
    """Framework-agnostic health check endpoint.

    Tracks the process start time so it can report uptime, and records
    the result of each ``check()`` invocation for monitoring purposes.

    Usage
    -----
    ::

        health = HealthEndpoint(service_name="agent-marketplace")
        status = health.check()
        # {"status": "ok", "version": "0.1.0", "uptime_seconds": 42, ...}

    Parameters
    ----------
    service_name:
        Name of the service reported in the health payload.
    registry_store:
        Optional ``RegistryStore`` instance.  When provided, the health
        check also reports the number of registered capabilities.
    """

    def __init__(
        self,
        service_name: str = "agent-marketplace",
        registry_store: object | None = None,
    ) -> None:
        self._service_name = service_name
        self._registry_store = registry_store
        self._started_at: float = time.monotonic()
        self._start_timestamp: str = datetime.now(tz=timezone.utc).isoformat()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check(self) -> dict[str, object]:
        """Perform a health check and return a status dictionary.

        Returns
        -------
        dict[str, object]
            A JSON-serializable mapping with at minimum the keys:

            - ``status``          — ``"ok"`` or ``"degraded"``
            - ``service``         — service name
            - ``version``         — package version string
            - ``uptime_seconds``  — seconds since the endpoint was created
            - ``timestamp``       — current UTC ISO-8601 timestamp
            - ``capabilities``    — count of registered capabilities (when a
              registry store was provided)
        """
        uptime = round(time.monotonic() - self._started_at, 3)
        result: dict[str, object] = {
            "status": "ok",
            "service": self._service_name,
            "version": __version__,
            "uptime_seconds": uptime,
            "started_at": self._start_timestamp,
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        }

        if self._registry_store is not None:
            try:
                from agent_marketplace.registry.store import RegistryStore

                if isinstance(self._registry_store, RegistryStore):
                    result["capabilities"] = self._registry_store.count()
                else:
                    result["capabilities"] = "unknown"
            except Exception:  # noqa: BLE001
                result["capabilities"] = "error"
                result["status"] = "degraded"

        return result

    def is_healthy(self) -> bool:
        """Return True if the service reports ``status == "ok"``."""
        return self.check().get("status") == "ok"
