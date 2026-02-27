"""Semantic matcher — cosine similarity matching between query and capabilities.

SemanticMatcher uses a fitted TFIDFEmbedder to compute cosine similarity
between a query and a collection of capability vectors, returning ranked
match results.

Cosine similarity between two L2-normalised TF-IDF vectors is simply their
dot product, which is a commodity algorithm requiring no external dependencies.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from agent_marketplace.semantic.embedder import TFIDFEmbedder, TFIDFVector


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
    """

    top_k: int = 10
    min_similarity: float = 0.0


# ---------------------------------------------------------------------------
# SemanticMatcher
# ---------------------------------------------------------------------------


class SemanticMatcher:
    """Computes cosine similarity between a query and capability embeddings.

    Parameters
    ----------
    embedder:
        A fitted TFIDFEmbedder instance.
    config:
        Matcher configuration.

    Example
    -------
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
    ) -> None:
        self._embedder = embedder
        self._config = config if config is not None else SemanticMatcherConfig()

    @property
    def config(self) -> SemanticMatcherConfig:
        """The active matcher configuration."""
        return self._config

    def match(
        self,
        query: str,
        *,
        top_k: int | None = None,
        min_similarity: float | None = None,
    ) -> list[MatchResult]:
        """Find capabilities that best match the query.

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

        query_vector = self._embedder.embed_query(query)
        scored: list[tuple[str, float]] = []

        for doc_vector in corpus_vectors:
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
