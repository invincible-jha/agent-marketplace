"""Tests for agent_marketplace.semantic.embedder — TFIDFEmbedder."""
from __future__ import annotations

import math

import pytest

from agent_marketplace.semantic.embedder import (
    EmbedderConfig,
    TFIDFEmbedder,
    TFIDFVector,
    _STOPWORDS,
)


# ---------------------------------------------------------------------------
# TFIDFVector
# ---------------------------------------------------------------------------


class TestTFIDFVector:
    def test_dot_identical_vectors(self) -> None:
        weights = {"pdf": 0.6, "extract": 0.8}
        norm = math.sqrt(sum(w * w for w in weights.values()))
        nw = {k: v / norm for k, v in weights.items()}
        v = TFIDFVector(text_id="v1", weights=nw)
        # dot with itself should be ~1.0 (L2 normalised)
        assert v.dot(v) == pytest.approx(1.0, abs=0.01)

    def test_dot_disjoint_vectors(self) -> None:
        v1 = TFIDFVector("v1", {"pdf": 1.0})
        v2 = TFIDFVector("v2", {"image": 1.0})
        assert v1.dot(v2) == pytest.approx(0.0)

    def test_norm(self) -> None:
        v = TFIDFVector("v1", {"a": 3.0, "b": 4.0})
        assert v.norm() == pytest.approx(5.0, abs=0.01)

    def test_terms(self) -> None:
        v = TFIDFVector("v1", {"pdf": 0.5, "extract": 0.7, "data": 0.3})
        terms = v.terms()
        assert sorted(terms) == terms  # alphabetically sorted

    def test_to_dict(self) -> None:
        v = TFIDFVector("v1", {"pdf": 0.5})
        assert v.to_dict() == {"pdf": 0.5}

    def test_frozen(self) -> None:
        v = TFIDFVector("v1", {"pdf": 0.5})
        with pytest.raises((TypeError, AttributeError)):
            v.text_id = "v2"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# TFIDFEmbedder — fit
# ---------------------------------------------------------------------------


class TestTFIDFEmbedderFit:
    def test_not_fitted_initially(self) -> None:
        embedder = TFIDFEmbedder()
        assert embedder.is_fitted is False

    def test_fitted_after_fit(self) -> None:
        embedder = TFIDFEmbedder()
        embedder.fit({"c1": "pdf extractor"})
        assert embedder.is_fitted is True

    def test_empty_corpus(self) -> None:
        embedder = TFIDFEmbedder()
        embedder.fit({})
        assert embedder.is_fitted is True
        assert embedder.corpus_size == 0
        assert embedder.vocabulary_size == 0

    def test_corpus_size(self) -> None:
        embedder = TFIDFEmbedder()
        embedder.fit({"c1": "extract pdf", "c2": "generate images"})
        assert embedder.corpus_size == 2

    def test_vocabulary_built(self) -> None:
        embedder = TFIDFEmbedder()
        embedder.fit({"c1": "extract pdf data"})
        assert embedder.vocabulary_size > 0

    def test_refit_replaces_corpus(self) -> None:
        embedder = TFIDFEmbedder()
        embedder.fit({"c1": "old text"})
        embedder.fit({"c2": "new text"})
        assert embedder.embed("c1") is None
        assert embedder.embed("c2") is not None


# ---------------------------------------------------------------------------
# TFIDFEmbedder — embed
# ---------------------------------------------------------------------------


class TestTFIDFEmbedderEmbed:
    def test_embed_returns_vector(self) -> None:
        embedder = TFIDFEmbedder()
        embedder.fit({"c1": "extract data from PDF"})
        vector = embedder.embed("c1")
        assert vector is not None
        assert isinstance(vector, TFIDFVector)

    def test_embed_unknown_id_returns_none(self) -> None:
        embedder = TFIDFEmbedder()
        embedder.fit({"c1": "some text"})
        assert embedder.embed("unknown") is None

    def test_vector_is_l2_normalised(self) -> None:
        embedder = TFIDFEmbedder()
        embedder.fit({"c1": "extract pdf tables"})
        vector = embedder.embed("c1")
        assert vector is not None
        assert vector.norm() == pytest.approx(1.0, abs=0.01)

    def test_identical_docs_have_same_vector(self) -> None:
        embedder = TFIDFEmbedder()
        embedder.fit({"c1": "pdf extractor", "c2": "pdf extractor"})
        v1 = embedder.embed("c1")
        v2 = embedder.embed("c2")
        assert v1 is not None and v2 is not None
        assert v1.dot(v2) == pytest.approx(1.0, abs=0.01)


# ---------------------------------------------------------------------------
# TFIDFEmbedder — embed_query
# ---------------------------------------------------------------------------


class TestTFIDFEmbedderEmbedQuery:
    def test_embed_query_returns_vector(self) -> None:
        embedder = TFIDFEmbedder()
        embedder.fit({"c1": "pdf extractor", "c2": "image generator"})
        qvec = embedder.embed_query("extract pdf tables")
        assert isinstance(qvec, TFIDFVector)

    def test_embed_query_without_fit_raises(self) -> None:
        embedder = TFIDFEmbedder()
        with pytest.raises(RuntimeError, match="fitted"):
            embedder.embed_query("some query")

    def test_query_vector_has_custom_id(self) -> None:
        embedder = TFIDFEmbedder()
        embedder.fit({"c1": "pdf"})
        qvec = embedder.embed_query("pdf", query_id="my_query")
        assert qvec.text_id == "my_query"

    def test_query_similar_to_matching_doc(self) -> None:
        embedder = TFIDFEmbedder()
        embedder.fit(
            {
                "c1": "extract tables from PDF documents",
                "c2": "generate images with AI",
            }
        )
        query_vec = embedder.embed_query("PDF table extraction")
        c1_vec = embedder.embed("c1")
        c2_vec = embedder.embed("c2")
        assert c1_vec is not None and c2_vec is not None
        sim_c1 = query_vec.dot(c1_vec)
        sim_c2 = query_vec.dot(c2_vec)
        assert sim_c1 > sim_c2


# ---------------------------------------------------------------------------
# TFIDFEmbedder — stopwords
# ---------------------------------------------------------------------------


class TestStopwords:
    def test_stopwords_not_in_vector(self) -> None:
        embedder = TFIDFEmbedder()
        embedder.fit({"c1": "extract the data from the file"})
        vector = embedder.embed("c1")
        assert vector is not None
        assert "the" not in vector.weights
        assert "from" not in vector.weights

    def test_stopwords_disabled(self) -> None:
        config = EmbedderConfig(remove_stopwords=False)
        embedder = TFIDFEmbedder(config)
        embedder.fit({"c1": "the file"})
        vector = embedder.embed("c1")
        assert vector is not None
        # "the" may now be in vocabulary (if not filtered)
        assert "the" in vector.weights


# ---------------------------------------------------------------------------
# TFIDFEmbedder — min_token_length
# ---------------------------------------------------------------------------


class TestMinTokenLength:
    def test_short_tokens_excluded(self) -> None:
        config = EmbedderConfig(min_token_length=4, remove_stopwords=False)
        embedder = TFIDFEmbedder(config)
        embedder.fit({"c1": "ok do data extraction"})
        vector = embedder.embed("c1")
        assert vector is not None
        assert "ok" not in vector.weights
        assert "do" not in vector.weights


# ---------------------------------------------------------------------------
# TFIDFEmbedder — max_vocabulary_size
# ---------------------------------------------------------------------------


class TestMaxVocabularySize:
    def test_vocabulary_capped(self) -> None:
        config = EmbedderConfig(max_vocabulary_size=5)
        embedder = TFIDFEmbedder(config)
        corpus = {
            "c1": "extract tables pdf documents reports files data",
            "c2": "generate images illustrations graphics artwork",
        }
        embedder.fit(corpus)
        assert embedder.vocabulary_size <= 5


# ---------------------------------------------------------------------------
# TFIDFEmbedder — all_vectors
# ---------------------------------------------------------------------------


class TestAllVectors:
    def test_all_vectors_count(self) -> None:
        embedder = TFIDFEmbedder()
        embedder.fit({"c1": "pdf", "c2": "image", "c3": "audio"})
        vectors = embedder.all_vectors()
        assert len(vectors) == 3

    def test_empty_corpus_all_vectors(self) -> None:
        embedder = TFIDFEmbedder()
        embedder.fit({})
        assert embedder.all_vectors() == []
