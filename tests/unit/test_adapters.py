"""Unit tests for agent-marketplace adapter modules.

Covers AsyncAPIAdapter, MCPAdapter, and OpenAPIAdapter — all three adapter
classes that import AgentCapability from external spec formats.
"""
from __future__ import annotations

import json

import pytest

from agent_marketplace.adapters.asyncapi import AsyncAPIAdapter
from agent_marketplace.adapters.mcp_adapter import MCPAdapter
from agent_marketplace.adapters.openapi import OpenAPIAdapter
from agent_marketplace.schema.capability import AgentCapability, CapabilityCategory


# ---------------------------------------------------------------------------
# Helpers / shared spec dicts
# ---------------------------------------------------------------------------


def _minimal_asyncapi_spec(title: str = "My Events") -> dict:
    return {
        "asyncapi": "2.6.0",
        "info": {"title": title, "version": "1.0.0"},
    }


def _full_asyncapi_spec() -> dict:
    return {
        "asyncapi": "2.6.0",
        "info": {
            "title": "Order Events",
            "version": "2.0.0",
            "description": "Events for order processing.",
            "contact": {"name": "Order Team", "email": "orders@example.com"},
        },
        "tags": [{"name": "analysis"}, {"name": "event-driven"}],
        "channels": {
            "order/created": {
                "subscribe": {
                    "message": {"contentType": "application/json"}
                }
            },
            "order/processed": {
                "messages": {
                    "OrderProcessed": {"contentType": "text/plain"}
                }
            },
        },
    }


def _minimal_openapi_spec(title: str = "My API") -> dict:
    return {
        "openapi": "3.0.0",
        "info": {"title": title, "version": "1.0.0"},
    }


def _full_openapi_spec() -> dict:
    return {
        "openapi": "3.0.0",
        "info": {
            "title": "Analysis API",
            "version": "2.0.0",
            "description": "Runs statistical analysis.",
            "contact": {"name": "DataCo", "email": "api@dataco.example.com"},
        },
        "servers": [{"url": "https://api.dataco.example.com/v2"}],
        "tags": [{"name": "analysis"}, "data"],
        "paths": {
            "/analyse": {
                "post": {
                    "requestBody": {
                        "content": {"application/json": {}}
                    },
                    "responses": {
                        "200": {
                            "content": {"application/json": {}}
                        }
                    },
                }
            }
        },
    }


def _minimal_mcp_manifest(name: str = "my-tool") -> dict:
    return {
        "name": name,
        "description": "Does something useful.",
        "inputSchema": {"type": "object"},
    }


# ---------------------------------------------------------------------------
# AsyncAPIAdapter
# ---------------------------------------------------------------------------


class TestAsyncAPIAdapterInit:
    def test_default_provider_name(self) -> None:
        adapter = AsyncAPIAdapter()
        assert adapter._provider_name == "Unknown Provider"

    def test_custom_provider_name(self) -> None:
        adapter = AsyncAPIAdapter(provider_name="Acme Events")
        assert adapter._provider_name == "Acme Events"

    def test_default_category(self) -> None:
        adapter = AsyncAPIAdapter()
        assert adapter._default_category == "automation"


class TestAsyncAPIAdapterFromDict:
    def test_minimal_spec_returns_capability(self) -> None:
        adapter = AsyncAPIAdapter()
        cap = adapter.from_dict(_minimal_asyncapi_spec())
        assert isinstance(cap, AgentCapability)
        assert cap.name == "My Events"

    def test_missing_title_raises_value_error(self) -> None:
        adapter = AsyncAPIAdapter()
        with pytest.raises(ValueError, match="info.title"):
            adapter.from_dict({"asyncapi": "2.0.0", "info": {}})

    def test_version_extracted_from_spec(self) -> None:
        adapter = AsyncAPIAdapter()
        cap = adapter.from_dict(_minimal_asyncapi_spec())
        assert cap.version == "1.0.0"

    def test_description_defaults_when_absent(self) -> None:
        adapter = AsyncAPIAdapter()
        cap = adapter.from_dict(_minimal_asyncapi_spec("Fallback"))
        assert "Fallback" in cap.description

    def test_contact_provider_name_used(self) -> None:
        adapter = AsyncAPIAdapter(provider_name="Default Provider")
        cap = adapter.from_dict(_full_asyncapi_spec())
        assert cap.provider.name == "Order Team"

    def test_contact_email_captured(self) -> None:
        adapter = AsyncAPIAdapter()
        cap = adapter.from_dict(_full_asyncapi_spec())
        assert cap.provider.contact_email == "orders@example.com"

    def test_async_tag_always_added(self) -> None:
        adapter = AsyncAPIAdapter()
        cap = adapter.from_dict(_minimal_asyncapi_spec())
        assert "async" in cap.tags

    def test_event_driven_tag_always_added(self) -> None:
        adapter = AsyncAPIAdapter()
        cap = adapter.from_dict(_minimal_asyncapi_spec())
        assert "event-driven" in cap.tags

    def test_spec_tags_included(self) -> None:
        adapter = AsyncAPIAdapter()
        cap = adapter.from_dict(_full_asyncapi_spec())
        assert "analysis" in cap.tags

    def test_category_inferred_from_analysis_tag(self) -> None:
        adapter = AsyncAPIAdapter()
        cap = adapter.from_dict(_full_asyncapi_spec())
        assert cap.category == CapabilityCategory.ANALYSIS

    def test_channel_content_types_extracted(self) -> None:
        adapter = AsyncAPIAdapter()
        cap = adapter.from_dict(_full_asyncapi_spec())
        assert "application/json" in cap.input_types

    def test_no_channels_defaults_to_json(self) -> None:
        adapter = AsyncAPIAdapter()
        cap = adapter.from_dict(_minimal_asyncapi_spec())
        assert "application/json" in cap.input_types
        assert cap.output_type == "application/json"

    def test_invalid_default_category_falls_back_to_automation(self) -> None:
        adapter = AsyncAPIAdapter(default_category="not_a_category")
        cap = adapter.from_dict(_minimal_asyncapi_spec())
        assert cap.category == CapabilityCategory.AUTOMATION

    def test_string_tags_in_spec(self) -> None:
        spec = _minimal_asyncapi_spec()
        spec["tags"] = ["research", "automation"]
        adapter = AsyncAPIAdapter()
        cap = adapter.from_dict(spec)
        assert "research" in cap.tags

    def test_version_fallback_when_empty_string(self) -> None:
        spec = _minimal_asyncapi_spec()
        spec["info"]["version"] = ""
        adapter = AsyncAPIAdapter()
        cap = adapter.from_dict(spec)
        assert cap.version == "1.0.0"


class TestAsyncAPIAdapterFromJson:
    def test_valid_json_returns_capability(self) -> None:
        adapter = AsyncAPIAdapter()
        cap = adapter.from_json(json.dumps(_minimal_asyncapi_spec()))
        assert cap.name == "My Events"


class TestAsyncAPIAdapterFromYaml:
    def test_valid_yaml_returns_capability(self) -> None:
        yaml = pytest.importorskip("yaml")
        adapter = AsyncAPIAdapter()
        yaml_text = yaml.dump(_minimal_asyncapi_spec())
        cap = adapter.from_yaml(yaml_text)
        assert cap.name == "My Events"

    def test_from_file_yaml(self, tmp_path) -> None:
        yaml = pytest.importorskip("yaml")
        adapter = AsyncAPIAdapter()
        spec_file = tmp_path / "spec.yaml"
        spec_file.write_text(yaml.dump(_minimal_asyncapi_spec("File Events")))
        cap = adapter.from_file(str(spec_file))
        assert cap.name == "File Events"


class TestAsyncAPIAdapterFromFile:
    def test_from_file_json(self, tmp_path) -> None:
        adapter = AsyncAPIAdapter()
        spec_file = tmp_path / "spec.json"
        spec_file.write_text(json.dumps(_minimal_asyncapi_spec("JSON Events")))
        cap = adapter.from_file(str(spec_file))
        assert cap.name == "JSON Events"


class TestAsyncAPIExtractMessageTypes:
    def test_channels_non_dict_returns_empty(self) -> None:
        types, output = AsyncAPIAdapter._extract_message_types({"channels": "bad"})
        assert types == []
        assert output == ""

    def test_non_dict_channel_obj_skipped(self) -> None:
        # When all channel objects are non-dict, content_types is empty,
        # so the fallback ["application/json"] is returned.
        types, output = AsyncAPIAdapter._extract_message_types(
            {"channels": {"ch1": "not_a_dict"}}
        )
        assert types == ["application/json"]

    def test_asyncapi2_subscribe_content_type(self) -> None:
        spec = {
            "channels": {
                "events": {
                    "subscribe": {
                        "message": {"contentType": "application/json"}
                    }
                }
            }
        }
        types, output = AsyncAPIAdapter._extract_message_types(spec)
        assert "application/json" in types

    def test_asyncapi3_messages_map(self) -> None:
        spec = {
            "channels": {
                "events": {
                    "messages": {
                        "MyMessage": {"contentType": "text/plain"}
                    }
                }
            }
        }
        types, output = AsyncAPIAdapter._extract_message_types(spec)
        assert "text/plain" in types


# ---------------------------------------------------------------------------
# MCPAdapter
# ---------------------------------------------------------------------------


class TestMCPAdapterInit:
    def test_defaults(self) -> None:
        adapter = MCPAdapter()
        assert adapter._provider_name == "MCP Server"
        assert adapter._server_url == ""
        assert adapter._default_category == "automation"

    def test_custom_values(self) -> None:
        adapter = MCPAdapter(
            provider_name="My Server",
            server_url="https://mcp.example.com",
            default_category="analysis",
        )
        assert adapter._provider_name == "My Server"
        assert adapter._server_url == "https://mcp.example.com"


class TestMCPAdapterFromDict:
    def test_single_tool_manifest(self) -> None:
        adapter = MCPAdapter()
        cap = adapter.from_dict(_minimal_mcp_manifest())
        assert isinstance(cap, AgentCapability)
        assert cap.name == "my-tool"

    def test_multi_tool_manifest_returns_first(self) -> None:
        adapter = MCPAdapter()
        manifest = {
            "tools": [
                _minimal_mcp_manifest("tool-a"),
                _minimal_mcp_manifest("tool-b"),
            ]
        }
        cap = adapter.from_dict(manifest)
        assert cap.name == "tool-a"

    def test_missing_name_raises_value_error(self) -> None:
        adapter = MCPAdapter()
        with pytest.raises(ValueError, match="'name'"):
            adapter.from_dict({"description": "no name here"})

    def test_description_defaults_to_name_based(self) -> None:
        adapter = MCPAdapter()
        cap = adapter.from_dict({"name": "unnamed-tool"})
        assert "unnamed-tool" in cap.description

    def test_version_defaults_to_1_0_0(self) -> None:
        adapter = MCPAdapter()
        cap = adapter.from_dict(_minimal_mcp_manifest())
        assert cap.version == "1.0.0"

    def test_mcp_tag_always_present(self) -> None:
        adapter = MCPAdapter()
        cap = adapter.from_dict(_minimal_mcp_manifest())
        assert "mcp" in cap.tags

    def test_annotations_tags_included(self) -> None:
        adapter = MCPAdapter()
        manifest = dict(_minimal_mcp_manifest())
        manifest["annotations"] = {"tags": ["analysis", "search"]}
        cap = adapter.from_dict(manifest)
        assert "analysis" in cap.tags

    def test_category_inferred_from_name(self) -> None:
        adapter = MCPAdapter()
        manifest = {"name": "data-extraction-tool", "description": "Extracts data"}
        cap = adapter.from_dict(manifest)
        assert cap.category == CapabilityCategory.EXTRACTION

    def test_server_url_stored_in_provider(self) -> None:
        adapter = MCPAdapter(server_url="https://mcp.example.com")
        cap = adapter.from_dict(_minimal_mcp_manifest())
        assert cap.provider.website == "https://mcp.example.com"

    def test_empty_tools_list_raises(self) -> None:
        adapter = MCPAdapter()
        with pytest.raises(ValueError):
            adapter.from_dict({"tools": []})


class TestMCPAdapterFromDictAll:
    def test_multi_tool_manifest_all(self) -> None:
        adapter = MCPAdapter()
        manifest = {
            "tools": [
                _minimal_mcp_manifest("tool-a"),
                _minimal_mcp_manifest("tool-b"),
            ]
        }
        caps = adapter.from_dict_all(manifest)
        assert len(caps) == 2
        names = {c.name for c in caps}
        assert names == {"tool-a", "tool-b"}

    def test_single_tool_dict_all(self) -> None:
        adapter = MCPAdapter()
        caps = adapter.from_dict_all(_minimal_mcp_manifest("solo"))
        assert len(caps) == 1
        assert caps[0].name == "solo"

    def test_empty_manifest_returns_empty(self) -> None:
        adapter = MCPAdapter()
        caps = adapter.from_dict_all({"no_tools": True})
        assert caps == []

    def test_non_dict_tools_entries_skipped(self) -> None:
        adapter = MCPAdapter()
        manifest = {"tools": ["not-a-dict", _minimal_mcp_manifest("valid")]}
        caps = adapter.from_dict_all(manifest)
        assert len(caps) == 1


class TestMCPAdapterFromJson:
    def test_valid_json(self) -> None:
        adapter = MCPAdapter()
        cap = adapter.from_json(json.dumps(_minimal_mcp_manifest()))
        assert cap.name == "my-tool"


class TestMCPAdapterFromFile:
    def test_json_file(self, tmp_path) -> None:
        adapter = MCPAdapter()
        spec_file = tmp_path / "manifest.json"
        spec_file.write_text(json.dumps(_minimal_mcp_manifest("file-tool")))
        cap = adapter.from_file(str(spec_file))
        assert cap.name == "file-tool"

    def test_yaml_file(self, tmp_path) -> None:
        yaml = pytest.importorskip("yaml")
        adapter = MCPAdapter()
        spec_file = tmp_path / "manifest.yaml"
        spec_file.write_text(yaml.dump(_minimal_mcp_manifest("yaml-tool")))
        cap = adapter.from_file(str(spec_file))
        assert cap.name == "yaml-tool"


class TestMCPExtractInputTypes:
    def test_non_dict_schema_returns_json(self) -> None:
        types = MCPAdapter._extract_input_types({"inputSchema": "string"})
        assert types == ["application/json"]

    def test_object_schema_type_returns_json(self) -> None:
        types = MCPAdapter._extract_input_types({"inputSchema": {"type": "object"}})
        assert types == ["application/json"]

    def test_missing_schema_returns_json(self) -> None:
        types = MCPAdapter._extract_input_types({})
        assert types == ["application/json"]


class TestMCPExtractAllTools:
    def test_single_tool_via_name_key(self) -> None:
        manifest = {"name": "solo", "description": "d"}
        tools = MCPAdapter._extract_all_tools(manifest)
        assert len(tools) == 1

    def test_no_tools_no_name_returns_empty(self) -> None:
        tools = MCPAdapter._extract_all_tools({"other": "value"})
        assert tools == []


# ---------------------------------------------------------------------------
# OpenAPIAdapter
# ---------------------------------------------------------------------------


class TestOpenAPIAdapterInit:
    def test_defaults(self) -> None:
        adapter = OpenAPIAdapter()
        assert adapter._provider_name == "Unknown Provider"
        assert adapter._default_category == "automation"


class TestOpenAPIAdapterFromDict:
    def test_minimal_spec_returns_capability(self) -> None:
        adapter = OpenAPIAdapter()
        cap = adapter.from_dict(_minimal_openapi_spec())
        assert isinstance(cap, AgentCapability)
        assert cap.name == "My API"

    def test_missing_title_raises_value_error(self) -> None:
        adapter = OpenAPIAdapter()
        with pytest.raises(ValueError, match="info.title"):
            adapter.from_dict({"openapi": "3.0.0", "info": {}})

    def test_version_extracted(self) -> None:
        adapter = OpenAPIAdapter()
        cap = adapter.from_dict(_minimal_openapi_spec())
        assert cap.version == "1.0.0"

    def test_description_defaults_when_absent(self) -> None:
        adapter = OpenAPIAdapter()
        cap = adapter.from_dict(_minimal_openapi_spec("No Desc"))
        assert "No Desc" in cap.description

    def test_full_spec_contact(self) -> None:
        adapter = OpenAPIAdapter()
        cap = adapter.from_dict(_full_openapi_spec())
        assert cap.provider.name == "DataCo"
        assert cap.provider.contact_email == "api@dataco.example.com"

    def test_server_url_in_provider(self) -> None:
        adapter = OpenAPIAdapter()
        cap = adapter.from_dict(_full_openapi_spec())
        assert cap.provider.website == "https://api.dataco.example.com/v2"

    def test_input_types_from_paths(self) -> None:
        adapter = OpenAPIAdapter()
        cap = adapter.from_dict(_full_openapi_spec())
        assert "application/json" in cap.input_types

    def test_output_type_from_paths(self) -> None:
        adapter = OpenAPIAdapter()
        cap = adapter.from_dict(_full_openapi_spec())
        assert cap.output_type == "application/json"

    def test_tags_extracted_from_dict_entries(self) -> None:
        adapter = OpenAPIAdapter()
        cap = adapter.from_dict(_full_openapi_spec())
        assert "analysis" in cap.tags

    def test_string_tags_also_included(self) -> None:
        adapter = OpenAPIAdapter()
        cap = adapter.from_dict(_full_openapi_spec())
        assert "data" in cap.tags

    def test_category_inferred_from_analysis_tag(self) -> None:
        adapter = OpenAPIAdapter()
        cap = adapter.from_dict(_full_openapi_spec())
        assert cap.category == CapabilityCategory.ANALYSIS

    def test_no_tags_uses_default_category(self) -> None:
        adapter = OpenAPIAdapter()
        cap = adapter.from_dict(_minimal_openapi_spec())
        assert cap.category == CapabilityCategory.AUTOMATION

    def test_invalid_default_category_falls_back(self) -> None:
        adapter = OpenAPIAdapter(default_category="bogus")
        cap = adapter.from_dict(_minimal_openapi_spec())
        assert cap.category == CapabilityCategory.AUTOMATION

    def test_version_empty_falls_back(self) -> None:
        spec = _minimal_openapi_spec()
        spec["info"]["version"] = ""
        adapter = OpenAPIAdapter()
        cap = adapter.from_dict(spec)
        assert cap.version == "1.0.0"


class TestOpenAPIAdapterFromJson:
    def test_valid_json(self) -> None:
        adapter = OpenAPIAdapter()
        cap = adapter.from_json(json.dumps(_minimal_openapi_spec()))
        assert cap.name == "My API"


class TestOpenAPIAdapterFromYaml:
    def test_valid_yaml(self) -> None:
        yaml = pytest.importorskip("yaml")
        adapter = OpenAPIAdapter()
        cap = adapter.from_yaml(yaml.dump(_minimal_openapi_spec("YAML API")))
        assert cap.name == "YAML API"


class TestOpenAPIAdapterFromFile:
    def test_json_file(self, tmp_path) -> None:
        adapter = OpenAPIAdapter()
        path = tmp_path / "api.json"
        path.write_text(json.dumps(_minimal_openapi_spec("File API")))
        cap = adapter.from_file(str(path))
        assert cap.name == "File API"

    def test_yaml_file(self, tmp_path) -> None:
        yaml = pytest.importorskip("yaml")
        adapter = OpenAPIAdapter()
        path = tmp_path / "api.yaml"
        path.write_text(yaml.dump(_minimal_openapi_spec("YAML File API")))
        cap = adapter.from_file(str(path))
        assert cap.name == "YAML File API"

    def test_yml_extension(self, tmp_path) -> None:
        yaml = pytest.importorskip("yaml")
        adapter = OpenAPIAdapter()
        path = tmp_path / "api.yml"
        path.write_text(yaml.dump(_minimal_openapi_spec("YML API")))
        cap = adapter.from_file(str(path))
        assert cap.name == "YML API"


class TestOpenAPIExtractFirstServerUrl:
    def test_first_server_url(self) -> None:
        spec = {"servers": [{"url": "https://api.example.com"}]}
        assert OpenAPIAdapter._extract_first_server_url(spec) == "https://api.example.com"

    def test_non_list_servers(self) -> None:
        assert OpenAPIAdapter._extract_first_server_url({"servers": "bad"}) == ""

    def test_empty_servers_list(self) -> None:
        assert OpenAPIAdapter._extract_first_server_url({"servers": []}) == ""

    def test_non_dict_first_server(self) -> None:
        assert OpenAPIAdapter._extract_first_server_url({"servers": ["not_dict"]}) == ""


class TestOpenAPIExtractIOTypes:
    def test_no_paths_key(self) -> None:
        inputs, output = OpenAPIAdapter._extract_io_types({})
        assert inputs == []
        assert output == ""

    def test_non_dict_paths(self) -> None:
        inputs, output = OpenAPIAdapter._extract_io_types({"paths": "bad"})
        assert inputs == []

    def test_non_dict_path_item_skipped(self) -> None:
        spec = {"paths": {"/test": "not_a_dict"}}
        inputs, output = OpenAPIAdapter._extract_io_types(spec)
        assert inputs == []

    def test_non_dict_operation_skipped(self) -> None:
        spec = {"paths": {"/test": {"get": "not_a_dict"}}}
        inputs, output = OpenAPIAdapter._extract_io_types(spec)
        assert inputs == []

    def test_text_plain_preferred_output(self) -> None:
        spec = {
            "paths": {
                "/test": {
                    "get": {
                        "responses": {
                            "200": {"content": {"text/plain": {}}}
                        }
                    }
                }
            }
        }
        inputs, output = OpenAPIAdapter._extract_io_types(spec)
        assert output == "text/plain"
