"""In-memory cosine similarity vector index for agent-marketplace.

InMemoryCosineIndex stores dense float vectors keyed by string identifiers
and supports top-k approximate nearest neighbour search via exact cosine
similarity.  No external dependencies are required — the cosine computation
is pure Python.

This is an intentionally simple implementation suitable for catalogues of
up to ~100 k entries.  For larger scales, swap in a purpose-built ANN library
by wrapping it in an EmbeddingBackend.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# SearchHit
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SearchHit:
    """A single result from :meth:`InMemoryCosineIndex.search`.

    Attributes
    ----------
    key:
        The identifier that was provided when the vector was added.
    score:
        Cosine similarity score.  Range is ``[-1.0, 1.0]`` for arbitrary
        vectors; ``[0.0, 1.0]`` for non-negative (e.g. TF-IDF) vectors.
    metadata:
        Arbitrary key-value payload stored alongside the vector.
    """

    key: str
    score: float
    metadata: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        """Serialise to a plain dict."""
        return {
            "key": self.key,
            "score": self.score,
            "metadata": dict(self.metadata),
        }


# ---------------------------------------------------------------------------
# InMemoryCosineIndex
# ---------------------------------------------------------------------------


class InMemoryCosineIndex:
    """In-memory vector index with exact cosine similarity search.

    Vectors are stored as plain Python lists so that no external numeric
    library is required.

    Example
    -------
    >>> index = InMemoryCosineIndex()
    >>> index.add("doc1", [1.0, 0.0, 0.0])
    >>> index.add("doc2", [0.0, 1.0, 0.0])
    >>> hits = index.search([1.0, 0.0, 0.0], top_k=1)
    >>> hits[0].key
    'doc1'
    """

    def __init__(self) -> None:
        self._vectors: dict[str, list[float]] = {}
        self._metadata: dict[str, dict[str, object]] = {}

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def add(
        self,
        key: str,
        vector: list[float],
        metadata: dict[str, object] | None = None,
    ) -> None:
        """Add or replace a vector in the index.

        Parameters
        ----------
        key:
            Unique identifier for this vector.
        vector:
            Dense float embedding vector.
        metadata:
            Optional payload to store alongside the vector.
        """
        self._vectors[key] = list(vector)
        self._metadata[key] = dict(metadata) if metadata is not None else {}

    def remove(self, key: str) -> None:
        """Remove a vector from the index (no-op if the key is absent).

        Parameters
        ----------
        key:
            The identifier of the vector to remove.
        """
        self._vectors.pop(key, None)
        self._metadata.pop(key, None)

    def clear(self) -> None:
        """Remove all vectors and metadata from the index."""
        self._vectors.clear()
        self._metadata.clear()

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def search(
        self,
        query_vector: list[float],
        top_k: int = 10,
        min_score: float = 0.0,
    ) -> list[SearchHit]:
        """Return the top-k most similar vectors to the query.

        Parameters
        ----------
        query_vector:
            Dense float query vector.
        top_k:
            Maximum number of results to return.
        min_score:
            Minimum cosine similarity required to include a result.

        Returns
        -------
        list[SearchHit]
            Results sorted by descending similarity score.
        """
        results: list[SearchHit] = []
        for key, stored_vector in self._vectors.items():
            score = self._cosine_similarity(query_vector, stored_vector)
            if score >= min_score:
                results.append(
                    SearchHit(
                        key=key,
                        score=score,
                        metadata=self._metadata.get(key, {}),
                    )
                )
        results.sort(key=lambda hit: hit.score, reverse=True)
        return results[:top_k]

    # ------------------------------------------------------------------
    # Inspection
    # ------------------------------------------------------------------

    def count(self) -> int:
        """Return the number of vectors currently in the index."""
        return len(self._vectors)

    def contains(self, key: str) -> bool:
        """Return True if a vector with this key exists in the index."""
        return key in self._vectors

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
        """Pure-Python cosine similarity.

        Parameters
        ----------
        vec_a:
            First vector.
        vec_b:
            Second vector.

        Returns
        -------
        float
            Cosine similarity, or ``0.0`` when vectors are zero-length or
            have mismatched dimensions.
        """
        if len(vec_a) != len(vec_b):
            return 0.0
        dot_product = sum(x * y for x, y in zip(vec_a, vec_b))
        norm_a = math.sqrt(sum(x * x for x in vec_a))
        norm_b = math.sqrt(sum(x * x for x in vec_b))
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        return dot_product / (norm_a * norm_b)
