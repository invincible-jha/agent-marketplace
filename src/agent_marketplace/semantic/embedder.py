"""TF-IDF embedder for capability descriptions.

TFIDFEmbedder converts natural-language capability descriptions into
sparse TF-IDF vectors without requiring any external ML libraries.

Algorithm
---------
1. Tokenise: lowercase, strip punctuation, remove stopwords.
2. Term Frequency (TF): log-normalised count.
   tf(t, d) = 1 + log(count(t, d))   if count > 0 else 0
3. Inverse Document Frequency (IDF): smoothed.
   idf(t, D) = log((|D| + 1) / (df(t) + 1)) + 1
4. TF-IDF vector = TF * IDF for each term in vocabulary.
5. L2 normalisation so cosine similarity = dot product.

The corpus must be fitted (via :meth:`fit`) before vectors can be
produced for individual texts.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable


# ---------------------------------------------------------------------------
# Stopwords (common English function words — commodity list)
# ---------------------------------------------------------------------------

_STOPWORDS: frozenset[str] = frozenset(
    [
        "a", "an", "and", "are", "as", "at", "be", "been", "but", "by",
        "can", "do", "for", "from", "has", "have", "he", "her", "his",
        "how", "in", "is", "it", "its", "if", "me", "my", "no", "not",
        "of", "on", "or", "our", "so", "than", "that", "the", "their",
        "them", "then", "there", "they", "this", "to", "up", "was",
        "we", "were", "what", "when", "which", "who", "will", "with",
        "you", "your",
    ]
)


# ---------------------------------------------------------------------------
# TFIDFVector
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TFIDFVector:
    """A sparse L2-normalised TF-IDF vector.

    Attributes
    ----------
    text_id:
        The identifier of the source text.
    weights:
        Mapping of term to its TF-IDF weight.  The vector is L2-normalised.
    original_text:
        The raw text from which this vector was computed.
    """

    text_id: str
    weights: dict[str, float]
    original_text: str = ""

    def dot(self, other: "TFIDFVector") -> float:
        """Compute dot product with another TFIDFVector.

        Since both vectors are L2-normalised, this equals cosine similarity.

        Parameters
        ----------
        other:
            Another TFIDFVector to compare against.

        Returns
        -------
        float
            Dot product in [0.0, 1.0].
        """
        return sum(
            self.weights[term] * other.weights[term]
            for term in self.weights
            if term in other.weights
        )

    def norm(self) -> float:
        """Return the L2 norm of this vector (should be ~1.0 if normalised)."""
        return math.sqrt(sum(w * w for w in self.weights.values()))

    def terms(self) -> list[str]:
        """Return sorted list of non-zero terms."""
        return sorted(self.weights.keys())

    def to_dict(self) -> dict[str, float]:
        """Return weights as a plain dict."""
        return dict(self.weights)


# ---------------------------------------------------------------------------
# EmbedderConfig
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EmbedderConfig:
    """Configuration for TFIDFEmbedder.

    Attributes
    ----------
    remove_stopwords:
        Whether to remove common English stopwords.
    min_token_length:
        Minimum character length for a token to be included.
    min_document_frequency:
        Minimum number of documents a term must appear in to be
        included in the vocabulary.
    max_vocabulary_size:
        Hard limit on vocabulary size.  The top-N terms by IDF are kept.
        0 means unlimited.
    """

    remove_stopwords: bool = True
    min_token_length: int = 2
    min_document_frequency: int = 1
    max_vocabulary_size: int = 0


# ---------------------------------------------------------------------------
# TFIDFEmbedder
# ---------------------------------------------------------------------------


class TFIDFEmbedder:
    """Bag-of-words TF-IDF embedder for capability descriptions.

    Parameters
    ----------
    config:
        Embedder configuration.  Defaults to standard settings.

    Example
    -------
    >>> embedder = TFIDFEmbedder()
    >>> corpus = {"c1": "extract data from PDF files", "c2": "generate images"}
    >>> embedder.fit(corpus)
    >>> vec = embedder.embed("c1")
    >>> vec is not None
    True
    """

    def __init__(self, config: EmbedderConfig | None = None) -> None:
        self._config = config if config is not None else EmbedderConfig()
        self._corpus: dict[str, str] = {}
        self._idf: dict[str, float] = {}
        self._vocabulary: frozenset[str] = frozenset()
        self._vectors: dict[str, TFIDFVector] = {}
        self._fitted: bool = False

    @property
    def is_fitted(self) -> bool:
        """True when the embedder has been fitted on a corpus."""
        return self._fitted

    @property
    def vocabulary_size(self) -> int:
        """Number of terms in the fitted vocabulary."""
        return len(self._vocabulary)

    @property
    def corpus_size(self) -> int:
        """Number of documents in the corpus."""
        return len(self._corpus)

    def fit(self, corpus: dict[str, str]) -> None:
        """Build TF-IDF index from a corpus of texts.

        Parameters
        ----------
        corpus:
            Mapping of text_id to raw text content.

        Returns
        -------
        None
        """
        self._corpus = dict(corpus)
        if not corpus:
            self._idf = {}
            self._vocabulary = frozenset()
            self._vectors = {}
            self._fitted = True
            return

        tokenized: dict[str, list[str]] = {
            text_id: self._tokenize(text) for text_id, text in corpus.items()
        }

        # Document frequency
        doc_freq: Counter[str] = Counter()
        for tokens in tokenized.values():
            for term in set(tokens):
                doc_freq[term] += 1

        num_docs = len(corpus)
        idf: dict[str, float] = {}
        for term, freq in doc_freq.items():
            if freq >= self._config.min_document_frequency:
                idf[term] = math.log((num_docs + 1) / (freq + 1)) + 1.0

        # Optionally prune vocabulary
        if self._config.max_vocabulary_size > 0 and len(idf) > self._config.max_vocabulary_size:
            # Keep terms with highest IDF (rarest terms — most discriminative)
            sorted_terms = sorted(idf.items(), key=lambda t: t[1], reverse=True)
            idf = dict(sorted_terms[: self._config.max_vocabulary_size])

        self._idf = idf
        self._vocabulary = frozenset(idf.keys())

        # Build vectors
        self._vectors = {}
        for text_id, tokens in tokenized.items():
            vector = self._build_vector(text_id, tokens, corpus[text_id])
            self._vectors[text_id] = vector

        self._fitted = True

    def embed(self, text_id: str) -> TFIDFVector | None:
        """Return the pre-computed TF-IDF vector for a corpus text.

        Parameters
        ----------
        text_id:
            The identifier of the corpus text.

        Returns
        -------
        TFIDFVector | None
            The pre-computed vector, or None if the text_id is not in the corpus.
        """
        return self._vectors.get(text_id)

    def embed_query(self, query: str, query_id: str = "__query__") -> TFIDFVector:
        """Embed an arbitrary query string using the fitted vocabulary.

        Terms not in the fitted vocabulary are ignored.

        Parameters
        ----------
        query:
            The query text to embed.
        query_id:
            An identifier for the resulting vector.

        Returns
        -------
        TFIDFVector
            The query TF-IDF vector.

        Raises
        ------
        RuntimeError
            If fit() has not been called.
        """
        if not self._fitted:
            raise RuntimeError(
                "TFIDFEmbedder must be fitted before embedding a query."
            )
        tokens = self._tokenize(query)
        return self._build_vector(query_id, tokens, query)

    def all_vectors(self) -> list[TFIDFVector]:
        """Return all pre-computed corpus vectors.

        Returns
        -------
        list[TFIDFVector]
            All fitted corpus vectors.
        """
        return list(self._vectors.values())

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _tokenize(self, text: str) -> list[str]:
        """Lowercase, strip punctuation, optionally remove stopwords."""
        text = text.lower()
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        tokens = text.split()
        tokens = [t for t in tokens if len(t) >= self._config.min_token_length]
        if self._config.remove_stopwords:
            tokens = [t for t in tokens if t not in _STOPWORDS]
        return tokens

    def _compute_tf(self, tokens: list[str]) -> dict[str, float]:
        """Log-normalised term frequency."""
        if not tokens:
            return {}
        counts: Counter[str] = Counter(tokens)
        return {term: 1.0 + math.log(count) for term, count in counts.items()}

    def _build_vector(
        self, text_id: str, tokens: list[str], original_text: str
    ) -> TFIDFVector:
        """Build and L2-normalise a TF-IDF vector for given tokens."""
        tf = self._compute_tf(tokens)
        weights: dict[str, float] = {}
        for term, tf_val in tf.items():
            if term in self._vocabulary:
                weights[term] = tf_val * self._idf[term]

        norm = math.sqrt(sum(w * w for w in weights.values())) or 1.0
        normalised = {t: w / norm for t, w in weights.items()}

        return TFIDFVector(
            text_id=text_id,
            weights=normalised,
            original_text=original_text,
        )
