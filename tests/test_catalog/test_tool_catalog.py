"""Tests for ToolCatalog — tool integration registry.

Covers:
- register_tool() — basic, with category/version, duplicate error
- get() — found, not found
- get_or_raise() — found, raises ToolNotFoundError
- search() — keyword match, category filter, no results, multi-keyword
- list_categories() — empty, populated, sorted
- list_tools() — empty, sorted by name
- export_openapi() — structure, paths, operationId, tags
- import_from_openapi() — basic import, duplicate skipped, no paths
- ToolEntry fields — all attributes, handler is callable
- ToolStorageBackend — save/get/all/contains/extension
- Exception hierarchy and messages
- Edge cases — empty query, special characters, no-handler tools
"""
from __future__ import annotations

import pytest

from agent_marketplace.catalog import (
    ToolAlreadyRegisteredError,
    ToolCatalog,
    ToolCatalogError,
    ToolEntry,
    ToolNotFoundError,
)
from agent_marketplace.catalog.tool_catalog import (
    ToolStorageBackend,
    _path_to_name,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _simple_handler(**kwargs: object) -> dict[str, object]:
    return {"output": "ok"}


def _make_catalog() -> ToolCatalog:
    return ToolCatalog()


def _register_weather(catalog: ToolCatalog) -> ToolEntry:
    return catalog.register_tool(
        name="weather",
        description="Get current weather for a location",
        parameters_schema={
            "type": "object",
            "properties": {"location": {"type": "string"}},
            "required": ["location"],
        },
        handler=_simple_handler,
        category="data",
    )


def _register_search(catalog: ToolCatalog) -> ToolEntry:
    return catalog.register_tool(
        name="web_search",
        description="Search the web for information",
        parameters_schema={
            "type": "object",
            "properties": {"query": {"type": "string"}},
        },
        handler=_simple_handler,
        category="search",
    )


# ---------------------------------------------------------------------------
# _path_to_name
# ---------------------------------------------------------------------------


class TestPathToName:
    def test_simple_path(self) -> None:
        assert _path_to_name("/tools/weather") == "weather"

    def test_hyphen_to_underscore(self) -> None:
        assert _path_to_name("/api/web-search") == "web_search"

    def test_last_segment_taken(self) -> None:
        assert _path_to_name("/a/b/c/my_tool") == "my_tool"

    def test_root_path(self) -> None:
        result = _path_to_name("/")
        assert isinstance(result, str)

    def test_lowercased(self) -> None:
        assert _path_to_name("/TOOLS/Weather") == "weather"


# ---------------------------------------------------------------------------
# ToolStorageBackend
# ---------------------------------------------------------------------------


class TestToolStorageBackend:
    def test_save_and_get(self) -> None:
        backend = ToolStorageBackend()
        entry = ToolEntry(
            name="test_tool",
            description="A test tool",
            category="test",
            parameters_schema={},
            handler=None,
        )
        backend.save(entry)
        assert backend.get("test_tool") is entry

    def test_get_missing_returns_none(self) -> None:
        backend = ToolStorageBackend()
        assert backend.get("missing") is None

    def test_contains_true_after_save(self) -> None:
        backend = ToolStorageBackend()
        entry = ToolEntry("t", "d", "c", {}, None)
        backend.save(entry)
        assert backend.contains("t")

    def test_contains_false_for_missing(self) -> None:
        backend = ToolStorageBackend()
        assert not backend.contains("nope")

    def test_all_returns_all_entries(self) -> None:
        backend = ToolStorageBackend()
        for i in range(3):
            backend.save(ToolEntry(f"tool_{i}", "d", "c", {}, None))
        assert len(backend.all()) == 3

    def test_custom_backend_used_by_catalog(self) -> None:
        class CountingBackend(ToolStorageBackend):
            save_count = 0
            def save(self, entry: ToolEntry) -> None:
                CountingBackend.save_count += 1
                super().save(entry)

        backend = CountingBackend()
        catalog = ToolCatalog(backend=backend)
        catalog.register_tool("t", "d", {})
        assert CountingBackend.save_count == 1


# ---------------------------------------------------------------------------
# register_tool()
# ---------------------------------------------------------------------------


class TestRegisterTool:
    def test_returns_tool_entry(self) -> None:
        catalog = _make_catalog()
        entry = _register_weather(catalog)
        assert isinstance(entry, ToolEntry)
        assert entry.name == "weather"

    def test_all_fields_set(self) -> None:
        catalog = _make_catalog()
        entry = catalog.register_tool(
            name="my_tool",
            description="Does something useful",
            parameters_schema={"type": "object"},
            handler=_simple_handler,
            category="utilities",
            version="2.1.0",
        )
        assert entry.description == "Does something useful"
        assert entry.category == "utilities"
        assert entry.version == "2.1.0"
        assert entry.handler is _simple_handler

    def test_default_category_is_general(self) -> None:
        catalog = _make_catalog()
        entry = catalog.register_tool("t", "d", {})
        assert entry.category == "general"

    def test_default_version_is_1_0_0(self) -> None:
        catalog = _make_catalog()
        entry = catalog.register_tool("t", "d", {})
        assert entry.version == "1.0.0"

    def test_handler_none_allowed(self) -> None:
        catalog = _make_catalog()
        entry = catalog.register_tool("t", "d", {}, handler=None)
        assert entry.handler is None

    def test_duplicate_name_raises(self) -> None:
        catalog = _make_catalog()
        catalog.register_tool("dup", "first", {})
        with pytest.raises(ToolAlreadyRegisteredError) as exc_info:
            catalog.register_tool("dup", "second", {})
        assert "dup" in str(exc_info.value)

    def test_different_names_both_registered(self) -> None:
        catalog = _make_catalog()
        catalog.register_tool("alpha", "d", {})
        catalog.register_tool("beta", "d", {})
        assert catalog.get("alpha") is not None
        assert catalog.get("beta") is not None


# ---------------------------------------------------------------------------
# get()
# ---------------------------------------------------------------------------


class TestGet:
    def test_found_returns_entry(self) -> None:
        catalog = _make_catalog()
        _register_weather(catalog)
        entry = catalog.get("weather")
        assert entry is not None
        assert entry.name == "weather"

    def test_missing_returns_none(self) -> None:
        catalog = _make_catalog()
        assert catalog.get("nonexistent") is None

    def test_returns_correct_entry_among_multiple(self) -> None:
        catalog = _make_catalog()
        _register_weather(catalog)
        _register_search(catalog)
        assert catalog.get("weather").name == "weather"
        assert catalog.get("web_search").name == "web_search"


# ---------------------------------------------------------------------------
# get_or_raise()
# ---------------------------------------------------------------------------


class TestGetOrRaise:
    def test_found_returns_entry(self) -> None:
        catalog = _make_catalog()
        _register_weather(catalog)
        entry = catalog.get_or_raise("weather")
        assert entry.name == "weather"

    def test_missing_raises_tool_not_found(self) -> None:
        catalog = _make_catalog()
        with pytest.raises(ToolNotFoundError) as exc_info:
            catalog.get_or_raise("missing_tool")
        assert "missing_tool" in str(exc_info.value)


# ---------------------------------------------------------------------------
# search()
# ---------------------------------------------------------------------------


class TestSearch:
    def test_keyword_in_name(self) -> None:
        catalog = _make_catalog()
        _register_weather(catalog)
        results = catalog.search("weather")
        assert len(results) == 1
        assert results[0].name == "weather"

    def test_keyword_in_description(self) -> None:
        catalog = _make_catalog()
        _register_weather(catalog)
        results = catalog.search("location")
        assert len(results) == 1

    def test_no_match_returns_empty(self) -> None:
        catalog = _make_catalog()
        _register_weather(catalog)
        results = catalog.search("quantum")
        assert results == []

    def test_case_insensitive(self) -> None:
        catalog = _make_catalog()
        _register_weather(catalog)
        results = catalog.search("WEATHER")
        assert len(results) == 1

    def test_multi_keyword_all_must_match(self) -> None:
        catalog = _make_catalog()
        _register_weather(catalog)
        _register_search(catalog)
        results = catalog.search("weather location")
        assert len(results) == 1
        assert results[0].name == "weather"

    def test_category_filter(self) -> None:
        catalog = _make_catalog()
        _register_weather(catalog)  # category=data
        _register_search(catalog)   # category=search
        results = catalog.search("", category="data")
        assert all(e.category == "data" for e in results)
        assert len(results) == 1

    def test_category_filter_case_insensitive(self) -> None:
        catalog = _make_catalog()
        _register_weather(catalog)
        results = catalog.search("", category="DATA")
        assert len(results) == 1

    def test_empty_query_returns_all(self) -> None:
        catalog = _make_catalog()
        _register_weather(catalog)
        _register_search(catalog)
        results = catalog.search("")
        assert len(results) == 2

    def test_results_sorted_by_name(self) -> None:
        catalog = _make_catalog()
        catalog.register_tool("zoo", "Z tool", {})
        catalog.register_tool("ant", "A tool", {})
        catalog.register_tool("meerkat", "M tool", {})
        results = catalog.search("")
        names = [e.name for e in results]
        assert names == sorted(names)


# ---------------------------------------------------------------------------
# list_categories()
# ---------------------------------------------------------------------------


class TestListCategories:
    def test_empty_catalog(self) -> None:
        catalog = _make_catalog()
        assert catalog.list_categories() == []

    def test_returns_unique_sorted_categories(self) -> None:
        catalog = _make_catalog()
        _register_weather(catalog)  # data
        _register_search(catalog)   # search
        catalog.register_tool("t3", "d", {}, category="analytics")
        cats = catalog.list_categories()
        assert cats == sorted(cats)
        assert len(cats) == 3

    def test_duplicate_categories_deduped(self) -> None:
        catalog = _make_catalog()
        catalog.register_tool("t1", "d", {}, category="data")
        catalog.register_tool("t2", "d", {}, category="data")
        assert catalog.list_categories() == ["data"]


# ---------------------------------------------------------------------------
# list_tools()
# ---------------------------------------------------------------------------


class TestListTools:
    def test_empty_catalog(self) -> None:
        assert _make_catalog().list_tools() == []

    def test_sorted_by_name(self) -> None:
        catalog = _make_catalog()
        catalog.register_tool("charlie", "d", {})
        catalog.register_tool("alpha", "d", {})
        catalog.register_tool("bravo", "d", {})
        names = [e.name for e in catalog.list_tools()]
        assert names == ["alpha", "bravo", "charlie"]

    def test_all_tools_returned(self) -> None:
        catalog = _make_catalog()
        _register_weather(catalog)
        _register_search(catalog)
        assert len(catalog.list_tools()) == 2


# ---------------------------------------------------------------------------
# export_openapi()
# ---------------------------------------------------------------------------


class TestExportOpenapi:
    def test_valid_openapi_structure(self) -> None:
        catalog = _make_catalog()
        spec = catalog.export_openapi()
        assert spec["openapi"] == "3.0.3"
        assert "info" in spec
        assert "paths" in spec

    def test_tool_appears_as_path(self) -> None:
        catalog = _make_catalog()
        _register_weather(catalog)
        spec = catalog.export_openapi()
        assert "/tools/weather" in spec["paths"]

    def test_operation_id_is_tool_name(self) -> None:
        catalog = _make_catalog()
        _register_weather(catalog)
        spec = catalog.export_openapi()
        op = spec["paths"]["/tools/weather"]["post"]
        assert op["operationId"] == "weather"

    def test_description_as_summary(self) -> None:
        catalog = _make_catalog()
        _register_weather(catalog)
        spec = catalog.export_openapi()
        op = spec["paths"]["/tools/weather"]["post"]
        assert "weather" in op["summary"].lower()

    def test_category_as_tag(self) -> None:
        catalog = _make_catalog()
        _register_weather(catalog)  # category=data
        spec = catalog.export_openapi()
        op = spec["paths"]["/tools/weather"]["post"]
        assert "data" in op["tags"]

    def test_version_in_extension(self) -> None:
        catalog = _make_catalog()
        catalog.register_tool("t", "d", {}, version="3.0.0")
        spec = catalog.export_openapi()
        op = spec["paths"]["/tools/t"]["post"]
        assert op["x-tool-version"] == "3.0.0"

    def test_parameters_schema_in_request_body(self) -> None:
        catalog = _make_catalog()
        schema = {"type": "object", "properties": {"x": {"type": "string"}}}
        catalog.register_tool("t", "d", schema)
        spec = catalog.export_openapi()
        rb = spec["paths"]["/tools/t"]["post"]["requestBody"]
        assert rb["content"]["application/json"]["schema"] == schema

    def test_empty_catalog_has_empty_paths(self) -> None:
        catalog = _make_catalog()
        spec = catalog.export_openapi()
        assert spec["paths"] == {}


# ---------------------------------------------------------------------------
# import_from_openapi()
# ---------------------------------------------------------------------------


class TestImportFromOpenapi:
    def _make_spec(
        self,
        name: str,
        description: str = "A tool",
        category: str = "imported",
    ) -> dict[str, object]:
        return {
            "openapi": "3.0.3",
            "info": {"title": "Test", "version": "1.0.0"},
            "paths": {
                f"/tools/{name}": {
                    "post": {
                        "operationId": name,
                        "summary": description,
                        "tags": [category],
                        "requestBody": {
                            "content": {
                                "application/json": {
                                    "schema": {"type": "object"}
                                }
                            }
                        },
                    }
                }
            },
        }

    def test_import_creates_tool_entry(self) -> None:
        catalog = _make_catalog()
        spec = self._make_spec("imported_tool", "An imported tool", "ext")
        entries = catalog.import_from_openapi(spec)
        assert len(entries) == 1
        assert entries[0].name == "imported_tool"

    def test_imported_tool_retrievable(self) -> None:
        catalog = _make_catalog()
        spec = self._make_spec("my_import")
        catalog.import_from_openapi(spec)
        entry = catalog.get("my_import")
        assert entry is not None

    def test_duplicate_import_skipped(self) -> None:
        catalog = _make_catalog()
        spec = self._make_spec("dup_tool")
        first = catalog.import_from_openapi(spec)
        second = catalog.import_from_openapi(spec)
        assert len(first) == 1
        assert len(second) == 0  # already registered

    def test_no_paths_returns_empty(self) -> None:
        catalog = _make_catalog()
        entries = catalog.import_from_openapi({"openapi": "3.0.3"})
        assert entries == []

    def test_handler_is_none_after_import(self) -> None:
        catalog = _make_catalog()
        spec = self._make_spec("no_handler_tool")
        entries = catalog.import_from_openapi(spec)
        assert entries[0].handler is None

    def test_category_from_tags(self) -> None:
        catalog = _make_catalog()
        spec = self._make_spec("tagged_tool", category="analytics")
        entries = catalog.import_from_openapi(spec)
        assert entries[0].category == "analytics"

    def test_roundtrip_export_then_import(self) -> None:
        """Tools exported from one catalog can be imported into another."""
        source = _make_catalog()
        _register_weather(source)
        _register_search(source)
        spec = source.export_openapi()

        dest = _make_catalog()
        imported = dest.import_from_openapi(spec)
        assert len(imported) == 2
        names = {e.name for e in imported}
        assert "weather" in names
        assert "web_search" in names


# ---------------------------------------------------------------------------
# ToolEntry dataclass
# ---------------------------------------------------------------------------


class TestToolEntry:
    def test_construction(self) -> None:
        entry = ToolEntry(
            name="my_tool",
            description="Does stuff",
            category="general",
            parameters_schema={"type": "object"},
            handler=_simple_handler,
            version="1.2.3",
        )
        assert entry.name == "my_tool"
        assert entry.version == "1.2.3"
        assert entry.handler is _simple_handler

    def test_handler_is_callable(self) -> None:
        entry = ToolEntry("t", "d", "c", {}, _simple_handler)
        assert callable(entry.handler)

    def test_default_version(self) -> None:
        entry = ToolEntry("t", "d", "c", {}, None)
        assert entry.version == "1.0.0"


# ---------------------------------------------------------------------------
# Exception hierarchy
# ---------------------------------------------------------------------------


class TestExceptions:
    def test_already_registered_is_catalog_error(self) -> None:
        exc = ToolAlreadyRegisteredError("tool")
        assert isinstance(exc, ToolCatalogError)

    def test_not_found_is_catalog_error(self) -> None:
        exc = ToolNotFoundError("tool")
        assert isinstance(exc, ToolCatalogError)

    def test_already_registered_message_contains_name(self) -> None:
        exc = ToolAlreadyRegisteredError("my_tool")
        assert "my_tool" in str(exc)

    def test_not_found_message_contains_name(self) -> None:
        exc = ToolNotFoundError("missing_tool")
        assert "missing_tool" in str(exc)

    def test_attributes_set(self) -> None:
        exc_dup = ToolAlreadyRegisteredError("dup")
        assert exc_dup.name == "dup"
        exc_nf = ToolNotFoundError("nf")
        assert exc_nf.name == "nf"
