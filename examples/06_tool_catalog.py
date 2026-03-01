#!/usr/bin/env python3
"""Example: Tool Catalog

Demonstrates registering and discovering tools using the ToolCatalog,
which provides a simpler interface for tool-centric use cases.

Usage:
    python examples/06_tool_catalog.py

Requirements:
    pip install agent-marketplace
"""
from __future__ import annotations

import agent_marketplace
from agent_marketplace import (
    ToolCatalog,
    ToolEntry,
    ToolNotFoundError,
    ToolAlreadyRegisteredError,
)


def main() -> None:
    print(f"agent-marketplace version: {agent_marketplace.__version__}")

    # Step 1: Create a tool catalog
    catalog = ToolCatalog()

    # Step 2: Register tools
    tools: list[dict[str, object]] = [
        {
            "tool_id": "web-search",
            "name": "Web Search",
            "description": "Search the web for publicly available information.",
            "tags": ["search", "web", "retrieval"],
            "provider": "search-service",
            "version": "2.1.0",
        },
        {
            "tool_id": "doc-parser",
            "name": "Document Parser",
            "description": "Parse PDFs, Word, and Excel files into structured text.",
            "tags": ["parse", "documents", "extraction"],
            "provider": "doc-tools",
            "version": "1.5.2",
        },
        {
            "tool_id": "code-executor",
            "name": "Sandboxed Code Executor",
            "description": "Execute Python code safely in an isolated sandbox.",
            "tags": ["code", "execution", "python"],
            "provider": "sandbox-labs",
            "version": "3.0.1",
        },
        {
            "tool_id": "email-sender",
            "name": "Email Sender",
            "description": "Send emails via SMTP with attachment support.",
            "tags": ["email", "communication", "smtp"],
            "provider": "comm-tools",
            "version": "1.2.0",
        },
    ]

    print("Registering tools:")
    for tool_data in tools:
        entry = ToolEntry(
            tool_id=str(tool_data["tool_id"]),
            name=str(tool_data["name"]),
            description=str(tool_data["description"]),
            tags=list(tool_data["tags"]),  # type: ignore[arg-type]
            provider=str(tool_data["provider"]),
            version=str(tool_data["version"]),
        )
        try:
            catalog.register(entry)
            print(f"  [{entry.tool_id}] {entry.name} v{entry.version}")
        except ToolAlreadyRegisteredError as error:
            print(f"  [SKIP] {error}")

    print(f"\nCatalog: {catalog.count()} tools registered")

    # Step 3: Search by tag
    search_results = catalog.search_by_tag("search")
    print(f"\nTools tagged 'search': {len(search_results)}")
    for tool in search_results:
        print(f"  {tool.name}")

    # Step 4: Retrieve a specific tool
    try:
        tool = catalog.get("web-search")
        print(f"\nRetrieved 'web-search': {tool.name} | provider={tool.provider}")
    except ToolNotFoundError as error:
        print(f"Tool not found: {error}")

    # Step 5: List all tools
    all_tools = catalog.list()
    print(f"\nAll tools:")
    for tool in all_tools:
        print(f"  [{tool.tool_id}] {tool.name} | tags={tool.tags}")


if __name__ == "__main__":
    main()
