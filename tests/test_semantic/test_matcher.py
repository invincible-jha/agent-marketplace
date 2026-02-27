"""Tests for agent_marketplace.semantic.matcher — SemanticMatcher."""
from __future__ import annotations

import pytest

from agent_marketplace.semantic.embedder import TFIDFEmbedder
from agent_marketplace.semantic.matcher import (
    MatchResult,
    SemanticMatcher,
    SemanticMatcherConfig,
)


# ---------------------------------------------------------------------------
# MatchResult
# ---------------------------------------------------------------------------


class TestMatchResult:
    def test_construction(self) -> None:
        mr = MatchResult(capability_id="c1", similarity=0.85, rank=0)
        assert mr.capability_id == "c1"
        assert mr.similarity == pytest.approx(0.85)
        assert mr.rank == 0

    def test_frozen(self) -> None:
        mr = MatchResult(capability_id="c1", similarity=0.5, rank=0)
        with pytest.raises((TypeError, AttributeError)):
            mr.similarity = 1.0  # type: ignore[misc]

    def test_to_dict(self) -> None:
        mr = MatchResult(capability_id="c1", similarity=0.70, rank=1)
        d = mr.to_dict()
        assert d["capability_id"] == "c1"
        assert d["similarity"] == pytest.approx(0.70)
        assert d["rank"] == 1


# ---------------------------------------------------------------------------
# SemanticMatcherConfig
# ---------------------------------------------------------------------------


class TestSemanticMatcherConfig:
    def test_defaults(self) -> None:
        config = SemanticMatcherConfig()
        assert config.top_k == 10
        assert config.min_similarity == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# SemanticMatcher — setup
# ---------------------------------------------------------------------------


def _fitted_embedder(corpus: dict[str, str] | None = None) -> TFIDFEmbedder:
    default_corpus = {
        "c1": "extract tables and text from PDF documents",
        "c2": "generate images using artificial intelligence",
        "c3": "translate documents between multiple languages",
        "c4": "summarise long documents into concise reports",
        "c5": "analyse sentiment in customer reviews",
    }
    embedder = TFIDFEmbedder()
    embedder.fit(corpus if corpus is not None else default_corpus)
    return embedder


# ---------------------------------------------------------------------------
# SemanticMatcher — match
# ---------------------------------------------------------------------------


class TestSemanticMatcherMatch:
    def test_returns_match_results(self) -> None:
        embedder = _fitted_embedder()
        matcher = SemanticMatcher(embedder)
        results = matcher.match("extract data from PDF")
        assert isinstance(results, list)
        assert all(isinstance(r, MatchResult) for r in results)

    def test_pdf_query_returns_c1_first(self) -> None:
        embedder = _fitted_embedder()
        matcher = SemanticMatcher(embedder)
        results = matcher.match("extract text from PDF documents")
        assert len(results) > 0
        assert results[0].capability_id == "c1"

    def test_image_query_returns_c2_first(self) -> None:
        embedder = _fitted_embedder()
        matcher = SemanticMatcher(embedder)
        results = matcher.match("generate images AI")
        assert len(results) > 0
        assert results[0].capability_id == "c2"

    def test_results_sorted_by_descending_similarity(self) -> None:
        embedder = _fitted_embedder()
        matcher = SemanticMatcher(embedder)
        results = matcher.match("document analysis")
        similarities = [r.similarity for r in results]
        assert similarities == sorted(similarities, reverse=True)

    def test_ranks_are_sequential(self) -> None:
        embedder = _fitted_embedder()
        matcher = SemanticMatcher(embedder)
        results = matcher.match("pdf text extraction")
        for i, r in enumerate(results):
            assert r.rank == i

    def test_top_k_respected(self) -> None:
        embedder = _fitted_embedder()
        matcher = SemanticMatcher(embedder)
        results = matcher.match("document", top_k=2)
        assert len(results) <= 2

    def test_min_similarity_filter(self) -> None:
        embedder = _fitted_embedder()
        matcher = SemanticMatcher(embedder)
        results = matcher.match("pdf extraction", min_similarity=0.90)
        # All returned results must meet threshold
        for r in results:
            assert r.similarity >= 0.90

    def test_empty_corpus_returns_empty(self) -> None:
        embedder = TFIDFEmbedder()
        embedder.fit({})
        matcher = SemanticMatcher(embedder)
        results = matcher.match("anything")
        assert results == []

    def test_unfitted_embedder_raises(self) -> None:
        embedder = TFIDFEmbedder()
        matcher = SemanticMatcher(embedder)
        with pytest.raises(RuntimeError, match="fitted"):
            matcher.match("query")

    def test_similarity_in_unit_range(self) -> None:
        embedder = _fitted_embedder()
        matcher = SemanticMatcher(embedder)
        results = matcher.match("pdf extraction")
        for r in results:
            assert 0.0 <= r.similarity <= 1.0


# ---------------------------------------------------------------------------
# SemanticMatcher — match_against
# ---------------------------------------------------------------------------


class TestSemanticMatcherMatchAgainst:
    def test_match_against_subset(self) -> None:
        embedder = _fitted_embedder()
        matcher = SemanticMatcher(embedder)
        all_vectors = embedder.all_vectors()
        # Match against only the first two
        subset = all_vectors[:2]
        query_vec = embedder.embed_query("pdf extraction")
        results = matcher.match_against(query_vec, subset)
        assert len(results) <= 2

    def test_match_against_empty(self) -> None:
        embedder = _fitted_embedder()
        matcher = SemanticMatcher(embedder)
        query_vec = embedder.embed_query("pdf")
        results = matcher.match_against(query_vec, [])
        assert results == []

    def test_match_against_top_k(self) -> None:
        embedder = _fitted_embedder()
        matcher = SemanticMatcher(embedder)
        all_vectors = embedder.all_vectors()
        query_vec = embedder.embed_query("pdf extraction")
        results = matcher.match_against(query_vec, all_vectors, top_k=2)
        assert len(results) <= 2


# ---------------------------------------------------------------------------
# SemanticMatcher — config
# ---------------------------------------------------------------------------


class TestSemanticMatcherConfig:
    def test_config_top_k_applied(self) -> None:
        embedder = _fitted_embedder()
        config = SemanticMatcherConfig(top_k=2)
        matcher = SemanticMatcher(embedder, config)
        results = matcher.match("document text")
        assert len(results) <= 2

    def test_config_min_similarity_applied(self) -> None:
        embedder = _fitted_embedder()
        config = SemanticMatcherConfig(min_similarity=0.99)
        matcher = SemanticMatcher(embedder, config)
        results = matcher.match("completely unrelated zephyr")
        # Very high threshold — likely no results
        for r in results:
            assert r.similarity >= 0.99
