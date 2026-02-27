"""Tests for agent_marketplace.semantic.index — CapabilityIndex."""
from __future__ import annotations

import pytest

from agent_marketplace.semantic.index import (
    CapabilityIndex,
    CapabilityIndexConfig,
    IndexedCapability,
    SearchResult,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _cap(
    cap_id: str,
    name: str = "Test Capability",
    description: str = "A test capability description",
    tags: list[str] | None = None,
) -> IndexedCapability:
    return IndexedCapability(
        capability_id=cap_id,
        name=name,
        description=description,
        tags=tags or [],
    )


def _populated_index() -> CapabilityIndex:
    index = CapabilityIndex()
    index.add_many(
        [
            _cap("c1", "PDF Extractor", "Extract tables and text from PDF documents", ["pdf", "ocr"]),
            _cap("c2", "Image Generator", "Generate images using artificial intelligence", ["ai", "image"]),
            _cap("c3", "Translator", "Translate documents between multiple languages", ["translation"]),
            _cap("c4", "Summariser", "Summarise long documents into concise reports", ["nlp", "summary"]),
            _cap("c5", "Sentiment Analyser", "Analyse sentiment in customer reviews", ["nlp", "sentiment"]),
        ]
    )
    return index


# ---------------------------------------------------------------------------
# IndexedCapability
# ---------------------------------------------------------------------------


class TestIndexedCapability:
    def test_construction(self) -> None:
        cap = IndexedCapability(
            capability_id="c1",
            name="PDF Extractor",
            description="Extract data from PDF",
        )
        assert cap.capability_id == "c1"
        assert cap.name == "PDF Extractor"

    def test_frozen(self) -> None:
        cap = _cap("c1")
        with pytest.raises((TypeError, AttributeError)):
            cap.capability_id = "c2"  # type: ignore[misc]

    def test_index_text_combines_fields(self) -> None:
        cap = IndexedCapability(
            capability_id="c1",
            name="PDF Extractor",
            description="Extract data from PDF",
            tags=["ocr", "tables"],
        )
        text = cap.index_text
        assert "PDF Extractor" in text
        assert "Extract data from PDF" in text
        assert "ocr" in text
        assert "tables" in text

    def test_index_text_no_tags(self) -> None:
        cap = IndexedCapability(
            capability_id="c1",
            name="PDF",
            description="Extract",
        )
        text = cap.index_text
        assert text.strip() != ""

    def test_to_dict(self) -> None:
        cap = IndexedCapability(
            capability_id="c1",
            name="PDF Extractor",
            description="Extract data",
            tags=["pdf"],
            metadata={"version": "1"},
        )
        d = cap.to_dict()
        assert d["capability_id"] == "c1"
        assert d["name"] == "PDF Extractor"
        assert d["description"] == "Extract data"
        assert d["tags"] == ["pdf"]
        assert d["metadata"] == {"version": "1"}

    def test_default_tags_empty(self) -> None:
        cap = IndexedCapability(capability_id="c1", name="X", description="Y")
        assert cap.tags == []

    def test_default_metadata_empty(self) -> None:
        cap = IndexedCapability(capability_id="c1", name="X", description="Y")
        assert cap.metadata == {}


# ---------------------------------------------------------------------------
# SearchResult
# ---------------------------------------------------------------------------


class TestSearchResult:
    def test_construction(self) -> None:
        cap = _cap("c1")
        result = SearchResult(capability=cap, similarity=0.85, rank=0)
        assert result.capability is cap
        assert result.similarity == pytest.approx(0.85)
        assert result.rank == 0

    def test_frozen(self) -> None:
        cap = _cap("c1")
        result = SearchResult(capability=cap, similarity=0.5, rank=0)
        with pytest.raises((TypeError, AttributeError)):
            result.rank = 1  # type: ignore[misc]

    def test_to_dict(self) -> None:
        cap = _cap("c1", name="PDF", description="Extract")
        result = SearchResult(capability=cap, similarity=0.75, rank=0)
        d = result.to_dict()
        assert "capability" in d
        assert d["similarity"] == pytest.approx(0.75)
        assert d["rank"] == 0
        assert d["capability"]["capability_id"] == "c1"


# ---------------------------------------------------------------------------
# CapabilityIndex — add and size
# ---------------------------------------------------------------------------


class TestCapabilityIndexAdd:
    def test_empty_initially(self) -> None:
        index = CapabilityIndex()
        assert index.size == 0
        assert index.is_empty is True

    def test_add_increases_size(self) -> None:
        index = CapabilityIndex()
        index.add(_cap("c1"))
        assert index.size == 1
        assert index.is_empty is False

    def test_add_many_increases_size(self) -> None:
        index = CapabilityIndex()
        index.add_many([_cap("c1"), _cap("c2"), _cap("c3")])
        assert index.size == 3

    def test_add_replaces_existing(self) -> None:
        index = CapabilityIndex()
        index.add(_cap("c1", name="Old Name"))
        index.add(_cap("c1", name="New Name"))
        assert index.size == 1
        cap = index.get("c1")
        assert cap is not None
        assert cap.name == "New Name"


# ---------------------------------------------------------------------------
# CapabilityIndex — get
# ---------------------------------------------------------------------------


class TestCapabilityIndexGet:
    def test_get_existing(self) -> None:
        index = CapabilityIndex()
        index.add(_cap("c1", name="PDF"))
        cap = index.get("c1")
        assert cap is not None
        assert cap.capability_id == "c1"

    def test_get_missing_returns_none(self) -> None:
        index = CapabilityIndex()
        assert index.get("nonexistent") is None


# ---------------------------------------------------------------------------
# CapabilityIndex — remove
# ---------------------------------------------------------------------------


class TestCapabilityIndexRemove:
    def test_remove_decreases_size(self) -> None:
        index = CapabilityIndex()
        index.add(_cap("c1"))
        index.add(_cap("c2"))
        index.remove("c1")
        assert index.size == 1
        assert index.get("c1") is None

    def test_remove_missing_raises_key_error(self) -> None:
        index = CapabilityIndex()
        with pytest.raises(KeyError):
            index.remove("nonexistent")

    def test_remove_then_get_returns_none(self) -> None:
        index = CapabilityIndex()
        index.add(_cap("c1"))
        index.remove("c1")
        assert index.get("c1") is None


# ---------------------------------------------------------------------------
# CapabilityIndex — search
# ---------------------------------------------------------------------------


class TestCapabilityIndexSearch:
    def test_search_returns_list(self) -> None:
        index = _populated_index()
        results = index.search("extract data from PDF")
        assert isinstance(results, list)

    def test_search_returns_search_results(self) -> None:
        index = _populated_index()
        results = index.search("pdf extraction")
        assert all(isinstance(r, SearchResult) for r in results)

    def test_pdf_query_returns_c1_first(self) -> None:
        index = _populated_index()
        results = index.search("extract text from PDF documents")
        assert len(results) > 0
        assert results[0].capability.capability_id == "c1"

    def test_results_sorted_descending(self) -> None:
        index = _populated_index()
        results = index.search("document analysis")
        similarities = [r.similarity for r in results]
        assert similarities == sorted(similarities, reverse=True)

    def test_ranks_are_sequential(self) -> None:
        index = _populated_index()
        results = index.search("pdf text")
        for i, r in enumerate(results):
            assert r.rank == i

    def test_top_k_respected(self) -> None:
        index = _populated_index()
        results = index.search("document", top_k=2)
        assert len(results) <= 2

    def test_min_similarity_filter(self) -> None:
        index = _populated_index()
        results = index.search("pdf", min_similarity=0.90)
        for r in results:
            assert r.similarity >= 0.90

    def test_empty_index_returns_empty(self) -> None:
        index = CapabilityIndex()
        results = index.search("anything")
        assert results == []

    def test_search_result_contains_capability(self) -> None:
        index = _populated_index()
        results = index.search("PDF extraction")
        assert len(results) > 0
        top = results[0]
        assert isinstance(top.capability, IndexedCapability)
        assert top.capability.capability_id in {"c1", "c2", "c3", "c4", "c5"}


# ---------------------------------------------------------------------------
# CapabilityIndex — all_capabilities
# ---------------------------------------------------------------------------


class TestCapabilityIndexAllCapabilities:
    def test_all_capabilities_count(self) -> None:
        index = _populated_index()
        caps = index.all_capabilities()
        assert len(caps) == 5

    def test_empty_index_all_capabilities(self) -> None:
        index = CapabilityIndex()
        assert index.all_capabilities() == []

    def test_all_capabilities_type(self) -> None:
        index = _populated_index()
        caps = index.all_capabilities()
        assert all(isinstance(c, IndexedCapability) for c in caps)


# ---------------------------------------------------------------------------
# CapabilityIndex — clear
# ---------------------------------------------------------------------------


class TestCapabilityIndexClear:
    def test_clear_empties_index(self) -> None:
        index = _populated_index()
        index.clear()
        assert index.size == 0
        assert index.is_empty is True

    def test_clear_search_returns_empty(self) -> None:
        index = _populated_index()
        index.clear()
        results = index.search("pdf")
        assert results == []

    def test_clear_then_add(self) -> None:
        index = _populated_index()
        index.clear()
        index.add(_cap("new1", name="New Cap", description="Fresh capability"))
        assert index.size == 1


# ---------------------------------------------------------------------------
# CapabilityIndex — config
# ---------------------------------------------------------------------------


class TestCapabilityIndexConfig:
    def test_config_top_k_applied(self) -> None:
        from agent_marketplace.semantic.matcher import SemanticMatcherConfig

        config = CapabilityIndexConfig(matcher_config=SemanticMatcherConfig(top_k=2))
        index = CapabilityIndex(config)
        index.add_many(
            [
                _cap("c1", description="pdf extraction documents tables"),
                _cap("c2", description="image generation ai graphics"),
                _cap("c3", description="translation language documents"),
                _cap("c4", description="sentiment analysis reviews customers"),
            ]
        )
        results = index.search("document text")
        assert len(results) <= 2

    def test_auto_refit_false_does_not_refit_on_add(self) -> None:
        config = CapabilityIndexConfig(auto_refit=False)
        index = CapabilityIndex(config)
        # With auto_refit=False, add does not trigger fit, search returns empty
        index.add(_cap("c1", description="pdf extraction"))
        results = index.search("pdf")
        # Without refit, embedder is not fitted, so returns empty
        assert results == []
