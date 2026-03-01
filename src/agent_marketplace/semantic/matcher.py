"""Semantic matcher — cosine similarity matching between query and capabilities.

SemanticMatcher uses a fitted TFIDFEmbedder to compute cosine similarity
between a query and a collection of capability vectors, returning ranked
match results.

Cosine similarity between two L2-normalised TF-IDF vectors is simply their
dot product, which is a commodity algorithm requiring no external dependencies.

An optional ``embedding_backend`` (EmbeddingBackend) can be supplied to use
dense vector search (e.g. sentence-transformers) instead of or in addition to
TF-IDF.  When both are active, scores are linearly fused using the configurable
``embedding_weight`` and ``tfidf_weight`` parameters.
"""
from __future__ import annotations

from dataclasses import dataclass

from agent_marketplace.semantic.embedder import TFIDFEmbedder, TFIDFVector
from agent_marketplace.semantic.embedding_backend import EmbeddingBackend
from agent_marketplace.semantic.vector_index import InMemoryCosineIndex


# ---------------------------------------------------------------------------
# MatchResult
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MatchResult:
    """A single capability match result.

    Attributes
    ----------
    capability_id:
        The identifier of the matched capability.
    similarity:
        Cosine similarity score in [0.0, 1.0].
    rank:
        Zero-based rank (0 = best match).
    """

    capability_id: str
    similarity: float
    rank: int

    def to_dict(self) -> dict[str, object]:
        """Serialise to a plain dict."""
        return {
            "capability_id": self.capability_id,
            "similarity": self.similarity,
            "rank": self.rank,
        }


# ---------------------------------------------------------------------------
# SemanticMatcherConfig
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SemanticMatcherConfig:
    """Configuration for SemanticMatcher.

    Attributes
    ----------
    top_k:
        Maximum number of results to return.
    min_similarity:
        Minimum cosine similarity threshold.  Results below this are excluded.
    embedding_weight:
        Weight applied to the dense embedding score when fusion is active.
        Must be in [0.0, 1.0].  Default ``0.5``.
    tfidf_weight:
        Weight applied to the TF-IDF score when fusion is active.
        Must be in [0.0, 1.0].  Default ``0.5``.
    """

    top_k: int = 10
    min_similarity: float = 0.0
    embedding_weight: float = 0.5
    tfidf_weight: float = 0.5


# ---------------------------------------------------------------------------
# SemanticMatcher
# ---------------------------------------------------------------------------


class SemanticMatcher:
    """Computes cosine similarity between a query and capability embeddings.

    Supports three operating modes depending on what is provided:

    1. **TF-IDF only** (default): ``embedding_backend=None`` — classic sparse
       vector matching, no external dependencies.
    2. **Embedding only**: ``embedding_backend`` provided and
       ``config.tfidf_weight=0.0`` — dense vector search only.
    3. **Fused**: ``embedding_backend`` provided with non-zero weights for
       both — linear combination of TF-IDF and embedding scores.

    Parameters
    ----------
    embedder:
        A fitted TFIDFEmbedder instance.
    config:
        Matcher configuration.
    embedding_backend:
        Optional dense-vector embedding backend.  When supplied, the matcher
        builds an InMemoryCosineIndex from the corpus and uses it for search.

    Example (TF-IDF only)
    ---------------------
    >>> from agent_marketplace.semantic.embedder import TFIDFEmbedder
    >>> embedder = TFIDFEmbedder()
    >>> embedder.fit({"c1": "extract tables from PDFs", "c2": "generate images"})
    >>> matcher = SemanticMatcher(embedder)
    >>> results = matcher.match("PDF data extraction")
    >>> results[0].capability_id == "c1"
    True
    """

    def __init__(
        self,
        embedder: TFIDFEmbedder,
        config: SemanticMatcherConfig | None = None,
        *,
        embedding_backend: EmbeddingBackend | None = None,
    ) -> None:
        self._embedder = embedder
        self._config = config if config is not None else SemanticMatcherConfig()
        self._embedding_backend: EmbeddingBackend | None = embedding_backend
        self._vector_index: InMemoryCosineIndex | None = None

        # If a backend is provided and the embedder is already fitted, build
        # the vector index immediately.
        if embedding_backend is not None and embedder.is_fitted:
            self._rebuild_vector_index()

    @property
    def config(self) -> SemanticMatcherConfig:
        """The active matcher configuration."""
        return self._config

    @property
    def embedding_backend(self) -> EmbeddingBackend | None:
        """The optional dense-vector embedding backend."""
        return self._embedding_backend

    # ------------------------------------------------------------------
    # Vector index management
    # ------------------------------------------------------------------

    def index_corpus(self, corpus: dict[str, str]) -> None:
        """Build (or rebuild) the dense vector index from a corpus dict.

        This must be called after providing an ``embedding_backend`` whenever
        the corpus changes — analogous to calling ``embedder.fit(corpus)`` for
        the TF-IDF path.

        Parameters
        ----------
        corpus:
            Mapping of capability_id to text content.  The same corpus that
            was passed to :meth:`TFIDFEmbedder.fit`.
        """
        if self._embedding_backend is None:
            return
        self._vector_index = InMemoryCosineIndex()
        texts = list(corpus.values())
        keys = list(corpus.keys())
        if not texts:
            return
        self._embedding_backend.fit(texts)
        vectors = self._embedding_backend.embed_batch(texts)
        for key, vector in zip(keys, vectors):
            self._vector_index.add(key, vector)

    def _rebuild_vector_index(self) -> None:
        """Rebuild the vector index from the fitted embedder's corpus."""
        if self._embedding_backend is None:
            return
        corpus_vectors = self._embedder.all_vectors()
        if not corpus_vectors:
            return
        corpus: dict[str, str] = {
            vec.text_id: vec.original_text for vec in corpus_vectors
        }
        self.index_corpus(corpus)

    def match(
        self,
        query: str,
        *,
        top_k: int | None = None,
        min_similarity: float | None = None,
    ) -> list[MatchResult]:
        """Find capabilities that best match the query.

        When an ``embedding_backend`` is present and a vector index has been
        built (via :meth:`index_corpus`), the final score is a weighted linear
        combination of the TF-IDF score and the embedding cosine score::

            score = tfidf_weight * tfidf_score + embedding_weight * emb_score

        If ``embedding_backend`` is ``None``, pure TF-IDF scoring is used
        (backward-compatible default behaviour).

        Parameters
        ----------
        query:
            Free-text query describing the desired capability.
        top_k:
            Override for the maximum number of results.  Defaults to
            ``config.top_k``.
        min_similarity:
            Override for the minimum similarity threshold.  Defaults to
            ``config.min_similarity``.

        Returns
        -------
        list[MatchResult]
            Ranked match results (best match first).

        Raises
        ------
        RuntimeError
            If the embedder has not been fitted.
        """
        if not self._embedder.is_fitted:
            raise RuntimeError(
                "SemanticMatcher requires a fitted TFIDFEmbedder. "
                "Call embedder.fit() first."
            )

        resolved_top_k = top_k if top_k is not None else self._config.top_k
        resolved_min = (
            min_similarity if min_similarity is not None else self._config.min_similarity
        )

        corpus_vectors = self._embedder.all_vectors()
        if not corpus_vectors:
            return []

        # ---- TF-IDF scores (always computed) --------------------------------
        query_tfidf = self._embedder.embed_query(query)
        tfidf_scores: dict[str, float] = {
            doc_vector.text_id: self._cosine(query_tfidf, doc_vector)
            for doc_vector in corpus_vectors
        }

        # ---- Embedding scores (optional) ------------------------------------
        embedding_scores: dict[str, float] = {}
        if self._embedding_backend is not None and self._vector_index is not None:
            query_dense = self._embedding_backend.embed(query)
            hits = self._vector_index.search(query_dense, top_k=self._vector_index.count())
            embedding_scores = {hit.key: hit.score for hit in hits}

        # ---- Score fusion ---------------------------------------------------
        all_ids = set(tfidf_scores.keys()) | set(embedding_scores.keys())
        tfidf_w = self._config.tfidf_weight
        emb_w = self._config.embedding_weight

        scored: list[tuple[str, float]] = []
        for cap_id in all_ids:
            t_score = tfidf_scores.get(cap_id, 0.0)
            e_score = embedding_scores.get(cap_id, 0.0)

            if embedding_scores:
                # Fusion mode: both backends active
                fused = tfidf_w * t_score + emb_w * e_score
            else:
                # TF-IDF only mode: backward compatible
                fused = t_score

            if fused >= resolved_min:
                scored.append((cap_id, fused))

        scored.sort(key=lambda t: t[1], reverse=True)
        top_scored = scored[:resolved_top_k]

        return [
            MatchResult(
                capability_id=cap_id,
                similarity=round(score, 4),
                rank=rank,
            )
            for rank, (cap_id, score) in enumerate(top_scored)
        ]

    def match_against(
        self,
        query_vector: TFIDFVector,
        candidate_vectors: list[TFIDFVector],
        *,
        top_k: int | None = None,
        min_similarity: float | None = None,
    ) -> list[MatchResult]:
        """Match a pre-computed query vector against a set of candidate vectors.

        This variant is useful when you have already embedded the query
        and want to match it against a subset of candidates.

        Parameters
        ----------
        query_vector:
            Pre-computed query TFIDFVector.
        candidate_vectors:
            List of candidate capability vectors to compare against.
        top_k:
            Maximum number of results.
        min_similarity:
            Minimum similarity threshold.

        Returns
        -------
        list[MatchResult]
            Ranked match results.
        """
        resolved_top_k = top_k if top_k is not None else self._config.top_k
        resolved_min = (
            min_similarity if min_similarity is not None else self._config.min_similarity
        )

        scored: list[tuple[str, float]] = []
        for doc_vector in candidate_vectors:
            similarity = self._cosine(query_vector, doc_vector)
            if similarity >= resolved_min:
                scored.append((doc_vector.text_id, similarity))

        scored.sort(key=lambda t: t[1], reverse=True)
        top_scored = scored[:resolved_top_k]

        return [
            MatchResult(
                capability_id=cap_id,
                similarity=round(score, 4),
                rank=rank,
            )
            for rank, (cap_id, score) in enumerate(top_scored)
        ]

    @staticmethod
    def _cosine(vec_a: TFIDFVector, vec_b: TFIDFVector) -> float:
        """Cosine similarity between two L2-normalised TFIDFVectors.

        Since vectors are pre-normalised, cosine = dot product.

        Parameters
        ----------
        vec_a:
            First vector.
        vec_b:
            Second vector.

        Returns
        -------
        float
            Cosine similarity in [0.0, 1.0].
        """
        return vec_a.dot(vec_b)
