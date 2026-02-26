"""Unit tests for agent-marketplace registry modules.

Covers MemoryStore, SQLiteStore, Namespace, and NamespaceManager.
RedisStore is skipped because it requires the redis package.
"""
from __future__ import annotations

import pytest

from agent_marketplace.registry.memory_store import MemoryStore
from agent_marketplace.registry.namespace import Namespace, NamespaceManager
from agent_marketplace.registry.sqlite_store import SQLiteStore
from agent_marketplace.registry.store import SearchQuery
from agent_marketplace.schema.capability import (
    AgentCapability,
    CapabilityCategory,
    PricingModel,
)
from agent_marketplace.schema.provider import ProviderInfo


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_capability(
    name: str = "test-cap",
    category: CapabilityCategory = CapabilityCategory.EXTRACTION,
    tags: list[str] | None = None,
    trust_level: float = 0.7,
    cost: float = 0.01,
    pricing_model: PricingModel = PricingModel.PER_CALL,
    provider_name: str = "TestProvider",
    supported_languages: list[str] | None = None,
    supported_frameworks: list[str] | None = None,
) -> AgentCapability:
    return AgentCapability(
        name=name,
        version="1.0.0",
        description="A test capability.",
        category=category,
        tags=tags or ["test", "extraction"],
        input_types=["application/json"],
        output_type="application/json",
        pricing_model=pricing_model,
        cost=cost,
        trust_level=trust_level,
        provider=ProviderInfo(name=provider_name),
        supported_languages=supported_languages or ["en"],
        supported_frameworks=supported_frameworks or ["langchain"],
    )


# ---------------------------------------------------------------------------
# MemoryStore — CRUD
# ---------------------------------------------------------------------------


class TestMemoryStoreCRUD:
    def setup_method(self) -> None:
        self.store = MemoryStore()
        self.cap = _make_capability()

    def test_register_and_get(self) -> None:
        self.store.register(self.cap)
        retrieved = self.store.get(self.cap.capability_id)
        assert retrieved.capability_id == self.cap.capability_id

    def test_register_duplicate_raises(self) -> None:
        self.store.register(self.cap)
        with pytest.raises(ValueError, match="already registered"):
            self.store.register(self.cap)

    def test_get_unknown_raises_key_error(self) -> None:
        with pytest.raises(KeyError):
            self.store.get("non-existent-id")

    def test_update_existing(self) -> None:
        self.store.register(self.cap)
        updated = self.cap.model_copy(update={"trust_level": 0.99})
        self.store.update(updated)
        assert self.store.get(self.cap.capability_id).trust_level == pytest.approx(0.99)

    def test_update_unknown_raises_key_error(self) -> None:
        with pytest.raises(KeyError):
            self.store.update(self.cap)

    def test_delete_existing(self) -> None:
        self.store.register(self.cap)
        self.store.delete(self.cap.capability_id)
        with pytest.raises(KeyError):
            self.store.get(self.cap.capability_id)

    def test_delete_unknown_raises_key_error(self) -> None:
        with pytest.raises(KeyError):
            self.store.delete("no-such-id")

    def test_list_all_empty(self) -> None:
        assert self.store.list_all() == []

    def test_list_all_returns_all(self) -> None:
        cap2 = _make_capability(name="cap-two")
        self.store.register(self.cap)
        self.store.register(cap2)
        assert len(self.store.list_all()) == 2

    def test_count_reflects_registrations(self) -> None:
        self.store.register(self.cap)
        assert self.store.count() == 1

    def test_exists_returns_true_for_registered(self) -> None:
        self.store.register(self.cap)
        assert self.store.exists(self.cap.capability_id) is True

    def test_exists_returns_false_for_unknown(self) -> None:
        assert self.store.exists("no-id") is False


# ---------------------------------------------------------------------------
# MemoryStore — search()
# ---------------------------------------------------------------------------


class TestMemoryStoreSearch:
    def setup_method(self) -> None:
        self.store = MemoryStore()
        self.cap_a = _make_capability(
            name="pdf-extractor",
            category=CapabilityCategory.EXTRACTION,
            tags=["pdf", "ocr"],
            trust_level=0.8,
            cost=0.01,
        )
        self.cap_b = _make_capability(
            name="image-generator",
            category=CapabilityCategory.GENERATION,
            tags=["image", "diffusion"],
            trust_level=0.5,
            cost=0.10,
            pricing_model=PricingModel.FREE,
        )
        self.store.register(self.cap_a)
        self.store.register(self.cap_b)

    def test_empty_query_returns_all(self) -> None:
        results = self.store.search(SearchQuery())
        assert len(results) == 2

    def test_keyword_filter(self) -> None:
        results = self.store.search(SearchQuery(keyword="pdf"))
        assert len(results) == 1
        assert results[0].name == "pdf-extractor"

    def test_category_filter(self) -> None:
        results = self.store.search(
            SearchQuery(category=CapabilityCategory.GENERATION)
        )
        assert len(results) == 1
        assert results[0].name == "image-generator"

    def test_tag_filter_and_semantics(self) -> None:
        results = self.store.search(SearchQuery(tags=["pdf", "ocr"]))
        assert len(results) == 1

    def test_min_trust_filter(self) -> None:
        results = self.store.search(SearchQuery(min_trust=0.7))
        assert len(results) == 1
        assert results[0].name == "pdf-extractor"

    def test_max_cost_filter(self) -> None:
        results = self.store.search(SearchQuery(max_cost=0.05))
        assert len(results) == 1
        assert results[0].name == "pdf-extractor"

    def test_pricing_model_filter(self) -> None:
        results = self.store.search(SearchQuery(pricing_model="free"))
        assert len(results) == 1
        assert results[0].name == "image-generator"

    def test_language_filter(self) -> None:
        cap_de = _make_capability(
            name="german-cap",
            supported_languages=["de", "en"],
        )
        self.store.register(cap_de)
        results = self.store.search(SearchQuery(supported_language="de"))
        assert any(r.name == "german-cap" for r in results)

    def test_framework_filter(self) -> None:
        cap_hugging = _make_capability(
            name="hugging-cap",
            supported_frameworks=["huggingface"],
        )
        self.store.register(cap_hugging)
        results = self.store.search(SearchQuery(supported_framework="huggingface"))
        assert any(r.name == "hugging-cap" for r in results)

    def test_limit_applied(self) -> None:
        results = self.store.search(SearchQuery(limit=1))
        assert len(results) == 1

    def test_offset_applied(self) -> None:
        all_results = self.store.search(SearchQuery(limit=0))
        offset_results = self.store.search(SearchQuery(offset=1, limit=0))
        assert len(offset_results) == len(all_results) - 1

    def test_no_match_returns_empty(self) -> None:
        results = self.store.search(SearchQuery(keyword="zzz-nonexistent"))
        assert results == []


# ---------------------------------------------------------------------------
# SQLiteStore — CRUD
# ---------------------------------------------------------------------------


class TestSQLiteStoreCRUD:
    def setup_method(self) -> None:
        self.store = SQLiteStore(":memory:")
        self.cap = _make_capability()

    def teardown_method(self) -> None:
        self.store.close()

    def test_register_and_get(self) -> None:
        self.store.register(self.cap)
        retrieved = self.store.get(self.cap.capability_id)
        assert retrieved.capability_id == self.cap.capability_id

    def test_register_duplicate_raises(self) -> None:
        self.store.register(self.cap)
        with pytest.raises(ValueError, match="already registered"):
            self.store.register(self.cap)

    def test_get_unknown_raises_key_error(self) -> None:
        with pytest.raises(KeyError):
            self.store.get("no-such-id")

    def test_update_existing(self) -> None:
        self.store.register(self.cap)
        updated = self.cap.model_copy(update={"trust_level": 0.95})
        self.store.update(updated)
        assert self.store.get(self.cap.capability_id).trust_level == pytest.approx(0.95)

    def test_update_unknown_raises_key_error(self) -> None:
        with pytest.raises(KeyError):
            self.store.update(self.cap)

    def test_delete_existing(self) -> None:
        self.store.register(self.cap)
        self.store.delete(self.cap.capability_id)
        with pytest.raises(KeyError):
            self.store.get(self.cap.capability_id)

    def test_delete_unknown_raises_key_error(self) -> None:
        with pytest.raises(KeyError):
            self.store.delete("no-such-id")

    def test_list_all_empty(self) -> None:
        assert self.store.list_all() == []

    def test_list_all_returns_all(self) -> None:
        cap2 = _make_capability(name="second-cap")
        self.store.register(self.cap)
        self.store.register(cap2)
        assert len(self.store.list_all()) == 2

    def test_count_reflects_registrations(self) -> None:
        self.store.register(self.cap)
        assert self.store.count() == 1

    def test_exists_true_for_registered(self) -> None:
        self.store.register(self.cap)
        assert self.store.exists(self.cap.capability_id) is True

    def test_exists_false_for_missing(self) -> None:
        assert self.store.exists("nope") is False

    def test_persists_capability_data_correctly(self) -> None:
        self.store.register(self.cap)
        retrieved = self.store.get(self.cap.capability_id)
        assert retrieved.name == self.cap.name
        assert retrieved.version == self.cap.version
        assert retrieved.trust_level == pytest.approx(self.cap.trust_level)


# ---------------------------------------------------------------------------
# SQLiteStore — search()
# ---------------------------------------------------------------------------


class TestSQLiteStoreSearch:
    def setup_method(self) -> None:
        self.store = SQLiteStore(":memory:")
        self.cap_a = _make_capability(
            name="pdf-extractor",
            category=CapabilityCategory.EXTRACTION,
            tags=["pdf", "ocr"],
            trust_level=0.8,
            cost=0.01,
        )
        self.cap_b = _make_capability(
            name="image-generator",
            category=CapabilityCategory.GENERATION,
            tags=["image", "diffusion"],
            trust_level=0.4,
            cost=0.10,
            pricing_model=PricingModel.FREE,
        )
        self.store.register(self.cap_a)
        self.store.register(self.cap_b)

    def teardown_method(self) -> None:
        self.store.close()

    def test_empty_query_returns_all(self) -> None:
        results = self.store.search(SearchQuery(limit=0))
        assert len(results) == 2

    def test_category_filter(self) -> None:
        results = self.store.search(
            SearchQuery(category=CapabilityCategory.EXTRACTION, limit=0)
        )
        assert len(results) == 1
        assert results[0].name == "pdf-extractor"

    def test_min_trust_filter(self) -> None:
        results = self.store.search(SearchQuery(min_trust=0.7, limit=0))
        assert len(results) == 1
        assert results[0].name == "pdf-extractor"

    def test_max_cost_filter(self) -> None:
        results = self.store.search(SearchQuery(max_cost=0.05, limit=0))
        assert len(results) == 1

    def test_pricing_model_filter(self) -> None:
        results = self.store.search(SearchQuery(pricing_model="free", limit=0))
        assert len(results) == 1
        assert results[0].name == "image-generator"

    def test_keyword_python_filter(self) -> None:
        results = self.store.search(SearchQuery(keyword="pdf", limit=0))
        assert len(results) == 1

    def test_tag_python_filter(self) -> None:
        results = self.store.search(SearchQuery(tags=["pdf"], limit=0))
        assert len(results) == 1

    def test_language_python_filter(self) -> None:
        cap_de = _make_capability(
            name="german-cap",
            supported_languages=["de"],
        )
        self.store.register(cap_de)
        results = self.store.search(SearchQuery(supported_language="de", limit=0))
        assert any(r.name == "german-cap" for r in results)

    def test_framework_python_filter(self) -> None:
        cap_hf = _make_capability(
            name="hf-cap",
            supported_frameworks=["huggingface"],
        )
        self.store.register(cap_hf)
        results = self.store.search(SearchQuery(supported_framework="huggingface", limit=0))
        assert any(r.name == "hf-cap" for r in results)

    def test_limit_and_offset(self) -> None:
        results = self.store.search(SearchQuery(limit=1, offset=0))
        assert len(results) == 1

    def test_no_match_returns_empty(self) -> None:
        results = self.store.search(SearchQuery(keyword="zzz-nonexistent", limit=0))
        assert results == []


# ---------------------------------------------------------------------------
# SQLiteStore — file-based persistence
# ---------------------------------------------------------------------------


class TestSQLiteStoreFilePersistence:
    def test_file_based_store(self, tmp_path) -> None:
        db_file = tmp_path / "test.db"
        store = SQLiteStore(str(db_file))
        cap = _make_capability()
        store.register(cap)
        assert store.count() == 1
        store.close()
        # Re-open and verify data persists
        store2 = SQLiteStore(str(db_file))
        assert store2.count() == 1
        store2.close()

    def test_accepts_path_object(self, tmp_path) -> None:
        db_path = tmp_path / "registry.db"
        store = SQLiteStore(db_path)
        cap = _make_capability()
        store.register(cap)
        assert store.count() == 1
        store.close()


# ---------------------------------------------------------------------------
# Namespace
# ---------------------------------------------------------------------------


class TestNamespace:
    def test_valid_namespace_created(self) -> None:
        ns = Namespace(organization="acme", agent="assistant", capability="search")
        assert ns.organization == "acme"
        assert ns.agent == "assistant"
        assert ns.capability == "search"

    def test_path_property(self) -> None:
        ns = Namespace(organization="acme", agent="assistant", capability="search")
        assert ns.path == "acme/assistant/search"

    def test_str_returns_path(self) -> None:
        ns = Namespace(organization="acme", agent="assistant", capability="search")
        assert str(ns) == "acme/assistant/search"

    def test_from_path_roundtrip(self) -> None:
        original = Namespace(organization="org-1", agent="bot", capability="cap-a")
        restored = Namespace.from_path(original.path)
        assert restored == original

    def test_from_path_invalid_segment_count_raises(self) -> None:
        with pytest.raises(ValueError, match="three segments"):
            Namespace.from_path("only/two")

    def test_from_path_too_many_segments_raises(self) -> None:
        with pytest.raises(ValueError, match="three segments"):
            Namespace.from_path("a/b/c/d")

    def test_invalid_segment_uppercase_raises(self) -> None:
        with pytest.raises(ValueError, match="invalid"):
            Namespace(organization="Acme", agent="bot", capability="search")

    def test_invalid_segment_starts_with_hyphen_raises(self) -> None:
        with pytest.raises(ValueError, match="invalid"):
            Namespace(organization="-acme", agent="bot", capability="search")

    def test_valid_with_hyphens_and_underscores(self) -> None:
        ns = Namespace(
            organization="my-org",
            agent="research_bot",
            capability="web-search",
        )
        assert ns.path == "my-org/research_bot/web-search"

    def test_valid_with_digits(self) -> None:
        ns = Namespace(organization="org2", agent="bot3", capability="cap4")
        assert ns.organization == "org2"

    def test_frozen_immutable(self) -> None:
        ns = Namespace(organization="acme", agent="bot", capability="search")
        with pytest.raises((AttributeError, TypeError)):
            ns.organization = "other"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# NamespaceManager
# ---------------------------------------------------------------------------


class TestNamespaceManager:
    def setup_method(self) -> None:
        self.manager = NamespaceManager()
        self.ns = Namespace(organization="acme", agent="assistant", capability="search")

    def test_register_and_resolve(self) -> None:
        self.manager.register(self.ns, "cap-001")
        assert self.manager.resolve(self.ns) == "cap-001"

    def test_register_duplicate_raises(self) -> None:
        self.manager.register(self.ns, "cap-001")
        with pytest.raises(ValueError, match="already registered"):
            self.manager.register(self.ns, "cap-002")

    def test_resolve_unknown_raises_key_error(self) -> None:
        with pytest.raises(KeyError):
            self.manager.resolve(self.ns)

    def test_reverse_resolve(self) -> None:
        self.manager.register(self.ns, "cap-001")
        reversed_ns = self.manager.reverse_resolve("cap-001")
        assert reversed_ns == self.ns

    def test_reverse_resolve_unknown_raises_key_error(self) -> None:
        with pytest.raises(KeyError):
            self.manager.reverse_resolve("no-such-cap-id")

    def test_deregister_removes_mapping(self) -> None:
        self.manager.register(self.ns, "cap-001")
        self.manager.deregister(self.ns)
        with pytest.raises(KeyError):
            self.manager.resolve(self.ns)

    def test_deregister_unknown_raises_key_error(self) -> None:
        with pytest.raises(KeyError):
            self.manager.deregister(self.ns)

    def test_list_namespaces_returns_sorted(self) -> None:
        ns_b = Namespace(organization="beta", agent="bot", capability="cap")
        ns_a = Namespace(organization="alpha", agent="bot", capability="cap")
        self.manager.register(ns_b, "cap-b")
        self.manager.register(ns_a, "cap-a")
        namespaces = self.manager.list_namespaces()
        assert namespaces[0].organization == "alpha"

    def test_list_by_org_filters(self) -> None:
        ns2 = Namespace(organization="other", agent="bot", capability="cap")
        self.manager.register(self.ns, "cap-001")
        self.manager.register(ns2, "cap-002")
        acme_ns = self.manager.list_by_org("acme")
        assert len(acme_ns) == 1
        assert acme_ns[0].organization == "acme"

    def test_list_by_org_no_match_returns_empty(self) -> None:
        self.manager.register(self.ns, "cap-001")
        assert self.manager.list_by_org("nonexistent-org") == []

    def test_empty_manager_lists_empty(self) -> None:
        assert self.manager.list_namespaces() == []
