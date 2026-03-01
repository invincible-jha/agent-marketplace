"""Embedding backend abstraction for agent-marketplace semantic search.

EmbeddingBackend is an abstract base class that defines the protocol for any
embedding provider.  The only included concrete implementation is
SentenceTransformerEmbedder, which is guarded by an ImportError so that the
dependency is strictly optional (see the ``embeddings`` extra in pyproject.toml).

This module intentionally contains no proprietary algorithms — cosine
similarity over dense vectors is a commodity operation.
"""
from __future__ import annotations

import math
from abc import ABC, abstractmethod


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class EmbeddingBackend(ABC):
    """Protocol for pluggable embedding backends.

    Subclasses must implement :meth:`embed`, :meth:`embed_batch`,
    :attr:`dimension`, and :meth:`fit`.

    Example
    -------
    >>> class MyBackend(EmbeddingBackend):
    ...     def embed(self, text: str) -> list[float]: return [0.0]
    ...     def embed_batch(self, texts: list[str]) -> list[list[float]]: return [[0.0]] * len(texts)
    ...     @property
    ...     def dimension(self) -> int: return 1
    ...     def fit(self, corpus: list[str]) -> None: pass
    """

    @abstractmethod
    def embed(self, text: str) -> list[float]:
        """Embed a single text into a dense float vector.

        Parameters
        ----------
        text:
            The input text to embed.

        Returns
        -------
        list[float]
            A dense float vector of length :attr:`dimension`.
        """
        ...

    @abstractmethod
    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts.

        Parameters
        ----------
        texts:
            A list of input texts to embed.

        Returns
        -------
        list[list[float]]
            A list of dense float vectors, one per input text.
        """
        ...

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the vector dimensionality produced by this backend."""
        ...

    @abstractmethod
    def fit(self, corpus: list[str]) -> None:
        """Optionally train the backend on a corpus.

        Pre-trained backends (e.g. sentence-transformers) can leave this as
        a no-op.  Corpus-fitted backends (e.g. a custom TF-IDF backend) should
        build their vocabulary here.

        Parameters
        ----------
        corpus:
            List of raw text strings to train on.
        """
        ...


# ---------------------------------------------------------------------------
# Optional sentence-transformers backend
# ---------------------------------------------------------------------------

try:
    from sentence_transformers import SentenceTransformer as _SentenceTransformer

    _ST_AVAILABLE = True
except ImportError:
    _SentenceTransformer = None  # type: ignore[assignment,misc]
    _ST_AVAILABLE = False


class SentenceTransformerEmbedder(EmbeddingBackend):
    """Embedding backend backed by ``sentence-transformers`` (Apache 2.0).

    Parameters
    ----------
    model_name:
        The sentence-transformers model to load.  Defaults to
        ``"all-MiniLM-L6-v2"`` — a lightweight, permissively licensed model.

    Raises
    ------
    ImportError
        If ``sentence-transformers`` is not installed.  Install it with:
        ``pip install agent-marketplace[embeddings]``

    Example
    -------
    >>> # doctest: +SKIP
    >>> embedder = SentenceTransformerEmbedder()
    >>> vec = embedder.embed("extract data from PDFs")
    >>> len(vec) == embedder.dimension
    True
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        if not _ST_AVAILABLE:
            raise ImportError(
                "sentence-transformers is not installed. "
                "Install it with: pip install agent-marketplace[embeddings]"
            )
        self._model = _SentenceTransformer(model_name)
        raw_dim = self._model.get_sentence_embedding_dimension()
        self._dimension: int = int(raw_dim) if raw_dim is not None else 0

    def embed(self, text: str) -> list[float]:
        """Embed a single text string.

        Parameters
        ----------
        text:
            The text to embed.

        Returns
        -------
        list[float]
            Dense float embedding vector.
        """
        return self._model.encode(text).tolist()  # type: ignore[union-attr]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of text strings.

        Parameters
        ----------
        texts:
            List of texts to embed.

        Returns
        -------
        list[list[float]]
            List of dense float embedding vectors.
        """
        return self._model.encode(texts).tolist()  # type: ignore[union-attr]

    @property
    def dimension(self) -> int:
        """Embedding dimensionality."""
        return self._dimension

    def fit(self, corpus: list[str]) -> None:
        """No-op: pre-trained model requires no corpus fitting."""


# ---------------------------------------------------------------------------
# Pure-Python cosine similarity utility (used by vector_index and tests)
# ---------------------------------------------------------------------------


def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """Compute cosine similarity between two dense float vectors.

    Parameters
    ----------
    vec_a:
        First vector.
    vec_b:
        Second vector.

    Returns
    -------
    float
        Cosine similarity.  Returns ``0.0`` when either vector is a zero
        vector or when the lengths differ.
    """
    if len(vec_a) != len(vec_b):
        return 0.0
    dot_product = sum(x * y for x, y in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(x * x for x in vec_a))
    norm_b = math.sqrt(sum(x * x for x in vec_b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot_product / (norm_a * norm_b)
