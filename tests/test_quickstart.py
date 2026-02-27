"""Test that the 3-line quickstart API works for agent-marketplace."""
from __future__ import annotations


def test_quickstart_import() -> None:
    from agent_marketplace import Marketplace

    mp = Marketplace()
    assert mp is not None


def test_quickstart_register_and_find() -> None:
    from agent_marketplace import Marketplace

    mp = Marketplace()
    mp.register("doc-summarizer", "Summarizes long documents", ["nlp"])
    results = mp.find("document summarization")
    assert isinstance(results, list)


def test_quickstart_find_empty() -> None:
    from agent_marketplace import Marketplace

    mp = Marketplace()
    results = mp.find("nonexistent-capability-xyz")
    assert isinstance(results, list)


def test_quickstart_register_returns_capability() -> None:
    from agent_marketplace import Marketplace
    from agent_marketplace.schema.capability import AgentCapability

    mp = Marketplace()
    cap = mp.register("test-cap", "A test capability", ["testing"])
    assert isinstance(cap, AgentCapability)
    assert cap.name == "test-cap"


def test_quickstart_store_accessible() -> None:
    from agent_marketplace import Marketplace
    from agent_marketplace.registry.memory_store import MemoryStore

    mp = Marketplace()
    assert isinstance(mp.store, MemoryStore)


def test_quickstart_repr() -> None:
    from agent_marketplace import Marketplace

    mp = Marketplace()
    mp.register("cap-1", "First capability")
    text = repr(mp)
    assert "Marketplace" in text
    assert "1" in text
