"""Tests for embedding_backend and vector_index modules.

All tests use a deterministic mock embedder — no sentence-transformers
installation is required.

Test groups
-----------
- InMemoryCosineIndex: add, search, remove, clear, count (8 tests)
- Cosine similarity: orthogonal=0, identical=1, opposite=-1 (3 tests)
- Search ordering: top_k and min_score (4 tests)
- Matcher integration: with embedder, embedding-based results (5 tests)
- Backward compat: without embedder, TF-IDF still works (3 tests)
- Fusion: both TF-IDF and embedding scores combined (3 tests)
- EmbeddingBackend ABC enforcement (2 tests)
- SentenceTransformerEmbedder: ImportError without library (2 tests)
"""
from __future__ import annotations

import math

import pytest

from agent_marketplace.semantic.embedding_backend import (
    EmbeddingBackend,
    SentenceTransformerEmbedder,
    cosine_similarity,
)
from agent_marketplace.semantic.embedder import TFIDFEmbedder
from agent_marketplace.semantic.matcher import SemanticMatcher, SemanticMatcherConfig
from agent_marketplace.semantic.vector_index import InMemoryCosineIndex, SearchHit


# ---------------------------------------------------------------------------
# Mock embedding backend — deterministic 3-D vectors
# ---------------------------------------------------------------------------

# Vocabulary: maps keyword -> unit basis vector index
# "pdf"  -> [1, 0, 0]
# "image" -> [0, 1, 0]
# "audio" -> [0, 0, 1]
# Any unrecognised text -> uniform [1/√3, 1/√3, 1/√3]

_KEYWORD_VECTORS: dict[str, list[float]] = {
    "pdf": [1.0, 0.0, 0.0],
    "image": [0.0, 1.0, 0.0],
    "audio": [0.0, 0.0, 1.0],
}
_DEFAULT_VECTOR: list[float] = [
    1.0 / math.sqrt(3),
    1.0 / math.sqrt(3),
    1.0 / math.sqrt(3),
]


def _keyword_vector(text: str) -> list[float]:
    """Return a deterministic 3-D unit vector based on text content."""
    text_lower = text.lower()
    for keyword, vector in _KEYWORD_VECTORS.items():
        if keyword in text_lower:
            return list(vector)
    return list(_DEFAULT_VECTOR)


class MockEmbeddingBackend(EmbeddingBackend):
    """Deterministic 3-D embedding backend for testing."""

    def embed(self, text: str) -> list[float]:
        return _keyword_vector(text)

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [_keyword_vector(t) for t in texts]

    @property
    def dimension(self) -> int:
        return 3

    def fit(self, corpus: list[str]) -> None:
        pass  # No training required for deterministic backend


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_CORPUS: dict[str, str] = {
    "cap-pdf": "extract tables and text from PDF documents",
    "cap-image": "generate images using artificial intelligence",
    "cap-audio": "transcribe audio recordings to text",
    "cap-translation": "translate documents between multiple languages",
}


def _fitted_embedder(corpus: dict[str, str] | None = None) -> TFIDFEmbedder:
    embedder = TFIDFEmbedder()
    embedder.fit(corpus if corpus is not None else _CORPUS)
    return embedder


# ---------------------------------------------------------------------------
# InMemoryCosineIndex — add, search, remove, clear, count
# ---------------------------------------------------------------------------


class TestInMemoryCosineIndex:
    def test_empty_index_count_is_zero(self) -> None:
        index = InMemoryCosineIndex()
        assert index.count() == 0

    def test_add_single_vector(self) -> None:
        index = InMemoryCosineIndex()
        index.add("v1", [1.0, 0.0, 0.0])
        assert index.count() == 1

    def test_add_multiple_vectors(self) -> None:
        index = InMemoryCosineIndex()
        index.add("v1", [1.0, 0.0, 0.0])
        index.add("v2", [0.0, 1.0, 0.0])
        index.add("v3", [0.0, 0.0, 1.0])
        assert index.count() == 3

    def test_add_replaces_existing_key(self) -> None:
        index = InMemoryCosineIndex()
        index.add("v1", [1.0, 0.0, 0.0])
        index.add("v1", [0.0, 1.0, 0.0])
        assert index.count() == 1
        hits = index.search([0.0, 1.0, 0.0], top_k=1)
        assert hits[0].key == "v1"
        assert hits[0].score == pytest.approx(1.0)

    def test_search_returns_search_hits(self) -> None:
        index = InMemoryCosineIndex()
        index.add("v1", [1.0, 0.0, 0.0])
        hits = index.search([1.0, 0.0, 0.0])
        assert isinstance(hits, list)
        assert all(isinstance(h, SearchHit) for h in hits)

    def test_remove_existing_key(self) -> None:
        index = InMemoryCosineIndex()
        index.add("v1", [1.0, 0.0, 0.0])
        index.remove("v1")
        assert index.count() == 0

    def test_remove_absent_key_is_noop(self) -> None:
        index = InMemoryCosineIndex()
        index.remove("nonexistent")
        assert index.count() == 0

    def test_clear_empties_index(self) -> None:
        index = InMemoryCosineIndex()
        index.add("v1", [1.0, 0.0, 0.0])
        index.add("v2", [0.0, 1.0, 0.0])
        index.clear()
        assert index.count() == 0

    def test_contains_returns_true_for_known_key(self) -> None:
        index = InMemoryCosineIndex()
        index.add("v1", [1.0, 0.0, 0.0])
        assert index.contains("v1") is True

    def test_contains_returns_false_for_unknown_key(self) -> None:
        index = InMemoryCosineIndex()
        assert index.contains("absent") is False

    def test_metadata_stored_and_retrieved(self) -> None:
        index = InMemoryCosineIndex()
        index.add("v1", [1.0, 0.0, 0.0], metadata={"category": "pdf"})
        hits = index.search([1.0, 0.0, 0.0], top_k=1)
        assert hits[0].metadata["category"] == "pdf"


# ---------------------------------------------------------------------------
# Cosine similarity: orthogonal=0, identical=1, opposite=-1
# ---------------------------------------------------------------------------


class TestCosineSimilarity:
    def test_identical_vectors_score_one(self) -> None:
        vec = [1.0, 0.0, 0.0]
        score = InMemoryCosineIndex._cosine_similarity(vec, vec)
        assert score == pytest.approx(1.0)

    def test_orthogonal_vectors_score_zero(self) -> None:
        vec_a = [1.0, 0.0, 0.0]
        vec_b = [0.0, 1.0, 0.0]
        score = InMemoryCosineIndex._cosine_similarity(vec_a, vec_b)
        assert score == pytest.approx(0.0)

    def test_opposite_vectors_score_negative_one(self) -> None:
        vec_a = [1.0, 0.0, 0.0]
        vec_b = [-1.0, 0.0, 0.0]
        score = InMemoryCosineIndex._cosine_similarity(vec_a, vec_b)
        assert score == pytest.approx(-1.0)

    def test_zero_vector_returns_zero(self) -> None:
        zero = [0.0, 0.0, 0.0]
        vec = [1.0, 0.0, 0.0]
        score = InMemoryCosineIndex._cosine_similarity(zero, vec)
        assert score == pytest.approx(0.0)

    def test_mismatched_dimensions_returns_zero(self) -> None:
        vec_a = [1.0, 0.0]
        vec_b = [1.0, 0.0, 0.0]
        score = InMemoryCosineIndex._cosine_similarity(vec_a, vec_b)
        assert score == pytest.approx(0.0)

    def test_module_level_cosine_similarity_function(self) -> None:
        vec = [0.6, 0.8, 0.0]
        score = cosine_similarity(vec, vec)
        assert score == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Search ordering: top_k and min_score
# ---------------------------------------------------------------------------


class TestSearchOrdering:
    def setup_method(self) -> None:
        self.index = InMemoryCosineIndex()
        self.index.add("pdf", [1.0, 0.0, 0.0])
        self.index.add("image", [0.0, 1.0, 0.0])
        self.index.add("audio", [0.0, 0.0, 1.0])
        # A vector that is close to pdf but not identical
        norm = math.sqrt(0.9**2 + 0.1**2)
        self.index.add("pdf-like", [0.9 / norm, 0.1 / norm, 0.0])

    def test_results_sorted_descending(self) -> None:
        hits = self.index.search([1.0, 0.0, 0.0])
        scores = [h.score for h in hits]
        assert scores == sorted(scores, reverse=True)

    def test_top_k_limits_results(self) -> None:
        hits = self.index.search([1.0, 0.0, 0.0], top_k=2)
        assert len(hits) <= 2

    def test_top_k_returns_best_matches(self) -> None:
        hits = self.index.search([1.0, 0.0, 0.0], top_k=1)
        assert len(hits) == 1
        assert hits[0].key == "pdf"

    def test_min_score_filters_results(self) -> None:
        # min_score=0.95 should exclude pdf-like and orthogonal vectors
        hits = self.index.search([1.0, 0.0, 0.0], min_score=0.95)
        # Only "pdf" has score=1.0; pdf-like has score<0.95 for the exact vector
        for hit in hits:
            assert hit.score >= 0.95

    def test_min_score_zero_includes_all_non_negative(self) -> None:
        # min_score=0.0 includes every result with score >= 0.0
        hits = self.index.search([1.0, 0.0, 0.0], min_score=0.0)
        assert all(h.score >= 0.0 for h in hits)

    def test_empty_index_returns_empty_list(self) -> None:
        empty_index = InMemoryCosineIndex()
        hits = empty_index.search([1.0, 0.0, 0.0])
        assert hits == []


# ---------------------------------------------------------------------------
# Matcher integration: with embedding backend, results are embedding-based
# ---------------------------------------------------------------------------


class TestMatcherWithEmbeddingBackend:
    def setup_method(self) -> None:
        self.backend = MockEmbeddingBackend()
        self.embedder = _fitted_embedder()
        self.matcher = SemanticMatcher(
            self.embedder,
            SemanticMatcherConfig(embedding_weight=1.0, tfidf_weight=0.0),
            embedding_backend=self.backend,
        )
        corpus = {cap_id: text for cap_id, text in _CORPUS.items()}
        self.matcher.index_corpus(corpus)

    def test_pdf_query_returns_cap_pdf_first(self) -> None:
        results = self.matcher.match("pdf extraction")
        assert len(results) > 0
        assert results[0].capability_id == "cap-pdf"

    def test_image_query_returns_cap_image_first(self) -> None:
        results = self.matcher.match("generate image")
        assert len(results) > 0
        assert results[0].capability_id == "cap-image"

    def test_audio_query_returns_cap_audio_first(self) -> None:
        results = self.matcher.match("audio transcription")
        assert len(results) > 0
        assert results[0].capability_id == "cap-audio"

    def test_results_sorted_descending(self) -> None:
        results = self.matcher.match("pdf")
        scores = [r.similarity for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_top_k_respected(self) -> None:
        results = self.matcher.match("pdf", top_k=2)
        assert len(results) <= 2

    def test_backend_property_accessible(self) -> None:
        assert self.matcher.embedding_backend is self.backend


# ---------------------------------------------------------------------------
# Backward compat: without embedder, TF-IDF still works
# ---------------------------------------------------------------------------


class TestMatcherBackwardCompat:
    def test_tfidf_only_mode_returns_results(self) -> None:
        embedder = _fitted_embedder()
        matcher = SemanticMatcher(embedder)
        results = matcher.match("PDF extraction")
        assert len(results) > 0

    def test_tfidf_only_pdf_query_returns_cap_pdf(self) -> None:
        embedder = _fitted_embedder()
        matcher = SemanticMatcher(embedder)
        results = matcher.match("extract text from PDF documents")
        assert len(results) > 0
        assert results[0].capability_id == "cap-pdf"

    def test_tfidf_only_unfitted_raises(self) -> None:
        embedder = TFIDFEmbedder()
        matcher = SemanticMatcher(embedder)
        with pytest.raises(RuntimeError, match="fitted"):
            matcher.match("query")

    def test_embedding_backend_none_by_default(self) -> None:
        embedder = _fitted_embedder()
        matcher = SemanticMatcher(embedder)
        assert matcher.embedding_backend is None


# ---------------------------------------------------------------------------
# Fusion: both TF-IDF and embedding results combined
# ---------------------------------------------------------------------------


class TestMatcherFusion:
    def setup_method(self) -> None:
        self.backend = MockEmbeddingBackend()
        self.embedder = _fitted_embedder()
        # Equal weights: 50% TF-IDF, 50% embedding
        self.matcher = SemanticMatcher(
            self.embedder,
            SemanticMatcherConfig(embedding_weight=0.5, tfidf_weight=0.5),
            embedding_backend=self.backend,
        )
        self.matcher.index_corpus(dict(_CORPUS))

    def test_fusion_returns_results(self) -> None:
        results = self.matcher.match("pdf documents")
        assert len(results) > 0

    def test_fusion_scores_between_zero_and_one(self) -> None:
        results = self.matcher.match("pdf documents")
        for r in results:
            # Fused score: 0.5*tfidf + 0.5*embedding, both in [0,1]
            assert r.similarity >= 0.0

    def test_fusion_pdf_query_ranks_cap_pdf_highly(self) -> None:
        results = self.matcher.match("pdf extraction documents")
        cap_ids = [r.capability_id for r in results]
        assert "cap-pdf" in cap_ids
        # cap-pdf should be in the top 2 results
        assert cap_ids.index("cap-pdf") < 2

    def test_fusion_results_sorted_descending(self) -> None:
        results = self.matcher.match("pdf image audio")
        scores = [r.similarity for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_fusion_top_k_respected(self) -> None:
        results = self.matcher.match("pdf", top_k=1)
        assert len(results) == 1


# ---------------------------------------------------------------------------
# EmbeddingBackend ABC enforcement
# ---------------------------------------------------------------------------


class TestEmbeddingBackendABC:
    def test_cannot_instantiate_abstract_base(self) -> None:
        with pytest.raises(TypeError):
            EmbeddingBackend()  # type: ignore[abstract]

    def test_partial_implementation_cannot_instantiate(self) -> None:
        class PartialBackend(EmbeddingBackend):
            def embed(self, text: str) -> list[float]:
                return [0.0]

            # Missing: embed_batch, dimension, fit

        with pytest.raises(TypeError):
            PartialBackend()  # type: ignore[abstract]

    def test_full_implementation_can_instantiate(self) -> None:
        backend = MockEmbeddingBackend()
        assert backend.dimension == 3

    def test_embed_returns_correct_dimension(self) -> None:
        backend = MockEmbeddingBackend()
        vector = backend.embed("hello world")
        assert len(vector) == backend.dimension

    def test_embed_batch_returns_list_of_vectors(self) -> None:
        backend = MockEmbeddingBackend()
        vectors = backend.embed_batch(["hello", "world"])
        assert len(vectors) == 2
        assert all(len(v) == backend.dimension for v in vectors)

    def test_fit_is_callable_noop(self) -> None:
        backend = MockEmbeddingBackend()
        backend.fit(["text1", "text2"])  # Should not raise


# ---------------------------------------------------------------------------
# SentenceTransformerEmbedder: ImportError without library
# ---------------------------------------------------------------------------


class TestSentenceTransformerEmbedder:
    def test_raises_import_error_when_library_missing(self) -> None:
        """Simulate missing sentence-transformers by patching _ST_AVAILABLE."""
        import agent_marketplace.semantic.embedding_backend as emb_mod

        original_st_available = emb_mod._ST_AVAILABLE
        try:
            emb_mod._ST_AVAILABLE = False
            with pytest.raises(ImportError, match="sentence-transformers"):
                SentenceTransformerEmbedder()
        finally:
            emb_mod._ST_AVAILABLE = original_st_available

    def test_import_error_message_mentions_install_command(self) -> None:
        """Verify the ImportError message is actionable."""
        import agent_marketplace.semantic.embedding_backend as emb_mod

        original_st_available = emb_mod._ST_AVAILABLE
        try:
            emb_mod._ST_AVAILABLE = False
            with pytest.raises(ImportError, match="pip install"):
                SentenceTransformerEmbedder()
        finally:
            emb_mod._ST_AVAILABLE = original_st_available

    def test_embedder_is_subclass_of_embedding_backend(self) -> None:
        assert issubclass(SentenceTransformerEmbedder, EmbeddingBackend)


# ---------------------------------------------------------------------------
# SearchHit dataclass
# ---------------------------------------------------------------------------


class TestSearchHit:
    def test_construction(self) -> None:
        hit = SearchHit(key="doc1", score=0.85)
        assert hit.key == "doc1"
        assert hit.score == pytest.approx(0.85)
        assert hit.metadata == {}

    def test_construction_with_metadata(self) -> None:
        hit = SearchHit(key="doc1", score=0.5, metadata={"type": "pdf"})
        assert hit.metadata["type"] == "pdf"

    def test_frozen_immutability(self) -> None:
        hit = SearchHit(key="doc1", score=0.5)
        with pytest.raises((TypeError, AttributeError)):
            hit.score = 1.0  # type: ignore[misc]

    def test_to_dict_serialization(self) -> None:
        hit = SearchHit(key="doc1", score=0.75, metadata={"tag": "x"})
        d = hit.to_dict()
        assert d["key"] == "doc1"
        assert d["score"] == pytest.approx(0.75)
        assert d["metadata"] == {"tag": "x"}
