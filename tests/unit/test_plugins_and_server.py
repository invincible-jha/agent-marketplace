"""Unit tests for agent-marketplace plugins and server modules.

Covers PluginRegistry (plugins/registry.py) and HealthEndpoint (server/health.py).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from unittest.mock import MagicMock, patch

import pytest

from agent_marketplace.plugins.registry import (
    PluginAlreadyRegisteredError,
    PluginNotFoundError,
    PluginRegistry,
)
from agent_marketplace.server.health import HealthEndpoint


# ---------------------------------------------------------------------------
# Minimal base class for testing PluginRegistry
# ---------------------------------------------------------------------------


class BaseProcessor(ABC):
    @abstractmethod
    def process(self, data: str) -> str: ...


class ConcreteProcessorA(BaseProcessor):
    def process(self, data: str) -> str:
        return data.upper()


class ConcreteProcessorB(BaseProcessor):
    def process(self, data: str) -> str:
        return data.lower()


class NotAProcessor:
    """Does NOT subclass BaseProcessor."""

    def process(self, data: str) -> str:
        return data


# ---------------------------------------------------------------------------
# PluginRegistry — construction
# ---------------------------------------------------------------------------


class TestPluginRegistryInit:
    def test_empty_registry(self) -> None:
        registry: PluginRegistry[BaseProcessor] = PluginRegistry(
            BaseProcessor, "processors"
        )
        assert len(registry) == 0

    def test_repr_includes_name(self) -> None:
        registry: PluginRegistry[BaseProcessor] = PluginRegistry(
            BaseProcessor, "processors"
        )
        assert "processors" in repr(registry)

    def test_repr_includes_base_class_name(self) -> None:
        registry: PluginRegistry[BaseProcessor] = PluginRegistry(
            BaseProcessor, "processors"
        )
        assert "BaseProcessor" in repr(registry)


# ---------------------------------------------------------------------------
# PluginRegistry — register decorator
# ---------------------------------------------------------------------------


class TestPluginRegistryRegisterDecorator:
    def setup_method(self) -> None:
        self.registry: PluginRegistry[BaseProcessor] = PluginRegistry(
            BaseProcessor, "test-processors"
        )

    def test_register_via_decorator(self) -> None:
        @self.registry.register("proc-a")
        class LocalProcessor(BaseProcessor):
            def process(self, data: str) -> str:
                return data

        assert "proc-a" in self.registry

    def test_decorator_returns_class_unchanged(self) -> None:
        @self.registry.register("proc-b")
        class LocalB(BaseProcessor):
            def process(self, data: str) -> str:
                return data

        # The class is still usable after registration
        instance = LocalB()
        assert instance.process("hello") == "hello"

    def test_duplicate_name_raises(self) -> None:
        @self.registry.register("proc-dup")
        class First(BaseProcessor):
            def process(self, data: str) -> str:
                return data

        with pytest.raises(PluginAlreadyRegisteredError):

            @self.registry.register("proc-dup")
            class Second(BaseProcessor):
                def process(self, data: str) -> str:
                    return data

    def test_non_subclass_raises_type_error(self) -> None:
        with pytest.raises(TypeError):

            @self.registry.register("invalid")
            class NotSubclass:  # type: ignore[misc]
                def process(self, data: str) -> str:
                    return data


# ---------------------------------------------------------------------------
# PluginRegistry — register_class()
# ---------------------------------------------------------------------------


class TestPluginRegistryRegisterClass:
    def setup_method(self) -> None:
        self.registry: PluginRegistry[BaseProcessor] = PluginRegistry(
            BaseProcessor, "test-processors"
        )

    def test_register_class_directly(self) -> None:
        self.registry.register_class("proc-a", ConcreteProcessorA)
        assert "proc-a" in self.registry

    def test_duplicate_class_raises(self) -> None:
        self.registry.register_class("proc-a", ConcreteProcessorA)
        with pytest.raises(PluginAlreadyRegisteredError):
            self.registry.register_class("proc-a", ConcreteProcessorB)

    def test_non_subclass_raises_type_error(self) -> None:
        with pytest.raises(TypeError):
            self.registry.register_class("bad", NotAProcessor)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# PluginRegistry — deregister()
# ---------------------------------------------------------------------------


class TestPluginRegistryDeregister:
    def setup_method(self) -> None:
        self.registry: PluginRegistry[BaseProcessor] = PluginRegistry(
            BaseProcessor, "test-processors"
        )
        self.registry.register_class("proc-a", ConcreteProcessorA)

    def test_deregister_removes_plugin(self) -> None:
        self.registry.deregister("proc-a")
        assert "proc-a" not in self.registry

    def test_deregister_unknown_raises(self) -> None:
        with pytest.raises(PluginNotFoundError):
            self.registry.deregister("nonexistent")

    def test_len_decrements_after_deregister(self) -> None:
        self.registry.deregister("proc-a")
        assert len(self.registry) == 0


# ---------------------------------------------------------------------------
# PluginRegistry — get()
# ---------------------------------------------------------------------------


class TestPluginRegistryGet:
    def setup_method(self) -> None:
        self.registry: PluginRegistry[BaseProcessor] = PluginRegistry(
            BaseProcessor, "test-processors"
        )
        self.registry.register_class("proc-a", ConcreteProcessorA)

    def test_get_registered_returns_class(self) -> None:
        cls = self.registry.get("proc-a")
        assert cls is ConcreteProcessorA

    def test_get_unknown_raises_plugin_not_found(self) -> None:
        with pytest.raises(PluginNotFoundError):
            self.registry.get("nonexistent")

    def test_retrieved_class_is_instantiable(self) -> None:
        cls = self.registry.get("proc-a")
        instance = cls()
        assert instance.process("hello") == "HELLO"


# ---------------------------------------------------------------------------
# PluginRegistry — list_plugins()
# ---------------------------------------------------------------------------


class TestPluginRegistryListPlugins:
    def setup_method(self) -> None:
        self.registry: PluginRegistry[BaseProcessor] = PluginRegistry(
            BaseProcessor, "test-processors"
        )

    def test_empty_registry_returns_empty_list(self) -> None:
        assert self.registry.list_plugins() == []

    def test_returns_sorted_names(self) -> None:
        self.registry.register_class("zebra", ConcreteProcessorA)
        self.registry.register_class("apple", ConcreteProcessorB)
        plugins = self.registry.list_plugins()
        assert plugins == sorted(plugins)

    def test_len_matches_list_plugins(self) -> None:
        self.registry.register_class("proc-1", ConcreteProcessorA)
        self.registry.register_class("proc-2", ConcreteProcessorB)
        assert len(self.registry) == len(self.registry.list_plugins())


# ---------------------------------------------------------------------------
# PluginRegistry — contains
# ---------------------------------------------------------------------------


class TestPluginRegistryContains:
    def setup_method(self) -> None:
        self.registry: PluginRegistry[BaseProcessor] = PluginRegistry(
            BaseProcessor, "test-processors"
        )
        self.registry.register_class("exists", ConcreteProcessorA)

    def test_contains_registered(self) -> None:
        assert "exists" in self.registry

    def test_not_contains_unregistered(self) -> None:
        assert "missing" not in self.registry


# ---------------------------------------------------------------------------
# PluginRegistry — load_entrypoints()
# ---------------------------------------------------------------------------


class TestPluginRegistryLoadEntrypoints:
    def test_load_entrypoints_no_entries(self) -> None:
        registry: PluginRegistry[BaseProcessor] = PluginRegistry(
            BaseProcessor, "test-processors"
        )
        # With an unused group name, no entry-points should be found
        registry.load_entrypoints("nonexistent.group.zzz")
        assert len(registry) == 0

    def test_load_entrypoints_already_registered_skips(self) -> None:
        registry: PluginRegistry[BaseProcessor] = PluginRegistry(
            BaseProcessor, "test-processors"
        )
        registry.register_class("my-proc", ConcreteProcessorA)
        # Simulate an entry-point that would register "my-proc" again
        mock_ep = MagicMock()
        mock_ep.name = "my-proc"
        mock_ep.load.return_value = ConcreteProcessorA

        with patch("importlib.metadata.entry_points", return_value=[mock_ep]):
            registry.load_entrypoints("some.group")

        assert len(registry) == 1  # Still just one

    def test_load_entrypoints_load_failure_skipped(self) -> None:
        registry: PluginRegistry[BaseProcessor] = PluginRegistry(
            BaseProcessor, "test-processors"
        )
        mock_ep = MagicMock()
        mock_ep.name = "bad-ep"
        mock_ep.load.side_effect = ImportError("module not found")

        with patch("importlib.metadata.entry_points", return_value=[mock_ep]):
            registry.load_entrypoints("some.group")

        assert len(registry) == 0

    def test_load_entrypoints_type_error_skipped(self) -> None:
        registry: PluginRegistry[BaseProcessor] = PluginRegistry(
            BaseProcessor, "test-processors"
        )
        mock_ep = MagicMock()
        mock_ep.name = "bad-type-ep"
        mock_ep.load.return_value = NotAProcessor  # Not a valid subclass

        with patch("importlib.metadata.entry_points", return_value=[mock_ep]):
            registry.load_entrypoints("some.group")

        assert len(registry) == 0


# ---------------------------------------------------------------------------
# PluginNotFoundError and PluginAlreadyRegisteredError
# ---------------------------------------------------------------------------


class TestPluginErrorClasses:
    def test_plugin_not_found_error_attributes(self) -> None:
        error = PluginNotFoundError("my-plugin", "my-registry")
        assert error.plugin_name == "my-plugin"
        assert error.registry_name == "my-registry"
        assert isinstance(error, KeyError)

    def test_plugin_already_registered_error_attributes(self) -> None:
        error = PluginAlreadyRegisteredError("my-plugin", "my-registry")
        assert error.plugin_name == "my-plugin"
        assert error.registry_name == "my-registry"
        assert isinstance(error, ValueError)


# ---------------------------------------------------------------------------
# HealthEndpoint — basic checks
# ---------------------------------------------------------------------------


class TestHealthEndpointBasic:
    def test_check_returns_dict(self) -> None:
        endpoint = HealthEndpoint()
        result = endpoint.check()
        assert isinstance(result, dict)

    def test_check_status_is_ok(self) -> None:
        endpoint = HealthEndpoint()
        result = endpoint.check()
        assert result["status"] == "ok"

    def test_check_has_required_keys(self) -> None:
        endpoint = HealthEndpoint()
        result = endpoint.check()
        for key in ("status", "service", "version", "uptime_seconds", "timestamp"):
            assert key in result

    def test_service_name_default(self) -> None:
        endpoint = HealthEndpoint()
        result = endpoint.check()
        assert result["service"] == "agent-marketplace"

    def test_service_name_custom(self) -> None:
        endpoint = HealthEndpoint(service_name="my-service")
        result = endpoint.check()
        assert result["service"] == "my-service"

    def test_uptime_seconds_non_negative(self) -> None:
        endpoint = HealthEndpoint()
        result = endpoint.check()
        assert result["uptime_seconds"] >= 0.0

    def test_version_is_string(self) -> None:
        endpoint = HealthEndpoint()
        result = endpoint.check()
        assert isinstance(result["version"], str)

    def test_timestamp_is_string(self) -> None:
        endpoint = HealthEndpoint()
        result = endpoint.check()
        assert isinstance(result["timestamp"], str)

    def test_is_healthy_returns_true_when_ok(self) -> None:
        endpoint = HealthEndpoint()
        assert endpoint.is_healthy() is True


# ---------------------------------------------------------------------------
# HealthEndpoint — with registry store
# ---------------------------------------------------------------------------


class TestHealthEndpointWithRegistryStore:
    def test_with_registry_store_includes_capabilities_count(self) -> None:
        from agent_marketplace.registry.memory_store import MemoryStore

        store = MemoryStore()
        endpoint = HealthEndpoint(registry_store=store)
        result = endpoint.check()
        assert "capabilities" in result
        assert result["capabilities"] == 0

    def test_with_non_registry_store_returns_unknown(self) -> None:
        fake_store = object()
        endpoint = HealthEndpoint(registry_store=fake_store)
        result = endpoint.check()
        assert result.get("capabilities") == "unknown"

    def test_with_no_store_does_not_include_capabilities(self) -> None:
        endpoint = HealthEndpoint()
        result = endpoint.check()
        assert "capabilities" not in result

    def test_store_raising_exception_sets_degraded(self) -> None:
        from agent_marketplace.registry.memory_store import MemoryStore

        # Subclass MemoryStore so isinstance passes, then break count()
        class BrokenStore(MemoryStore):
            def count(self) -> int:
                raise RuntimeError("DB is down")

        broken_store = BrokenStore()
        endpoint = HealthEndpoint(registry_store=broken_store)
        result = endpoint.check()

        assert result["status"] == "degraded"
        assert result["capabilities"] == "error"
