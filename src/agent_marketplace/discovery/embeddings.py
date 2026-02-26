"""TF-IDF embedding search for agent-marketplace.

Provides semantic similarity matching using a pure-Python TF-IDF
vectorizer and cosine similarity computation.  No external ML
dependencies are required.
"""
from __future__ import annotations

import math
import re
from collections import Counter


class EmbeddingSearch:
    """TF-IDF based embedding search for agent capabilities.

    Builds a term-frequency inverse-document-frequency (TF-IDF) vector
    space from a corpus of documents (capability text fields), then
    computes cosine similarity between a query and each document to
    return the most similar items.

    Usage
    -----
    ::

        search = EmbeddingSearch()
        search.fit({"cap1": "analyse PDFs and extract tables", "cap2": "generate images"})
        results = search.query("extract data from PDF documents", top_k=5)
        # results: list of (doc_id, similarity_score) tuples
    """

    # Minimum document frequency for a term to be included in the vocabulary
    _MIN_DOCUMENT_FREQUENCY: int = 1

    def __init__(self) -> None:
        self._documents: dict[str, str] = {}
        self._tfidf_matrix: dict[str, dict[str, float]] = {}
        self._idf: dict[str, float] = {}
        self._vocabulary: set[str] = set()
        self._is_fitted: bool = False

    def fit(self, documents: dict[str, str]) -> None:
        """Build the TF-IDF index from a mapping of id to text.

        Parameters
        ----------
        documents:
            Mapping of ``document_id`` to raw text content.
            Replaces any previously fitted index.
        """
        self._documents = dict(documents)
        if not documents:
            self._is_fitted = True
            return

        tokenized: dict[str, list[str]] = {
            doc_id: self._tokenize(text) for doc_id, text in documents.items()
        }

        # Build vocabulary and document frequency counts
        doc_freq: Counter[str] = Counter()
        for tokens in tokenized.values():
            for term in set(tokens):
                doc_freq[term] += 1

        num_docs = len(documents)
        self._idf = {
            term: math.log((num_docs + 1) / (freq + 1)) + 1.0
            for term, freq in doc_freq.items()
            if freq >= self._MIN_DOCUMENT_FREQUENCY
        }
        self._vocabulary = set(self._idf.keys())

        # Build TF-IDF vectors
        self._tfidf_matrix = {}
        for doc_id, tokens in tokenized.items():
            tf = self._compute_tf(tokens)
            tfidf = {
                term: tf.get(term, 0.0) * self._idf[term]
                for term in self._vocabulary
                if term in tf
            }
            # L2 normalize
            norm = math.sqrt(sum(v * v for v in tfidf.values())) or 1.0
            self._tfidf_matrix[doc_id] = {t: v / norm for t, v in tfidf.items()}

        self._is_fitted = True

    def query(
        self,
        query_text: str,
        top_k: int = 10,
        min_similarity: float = 0.0,
    ) -> list[tuple[str, float]]:
        """Return the top-k most similar documents to *query_text*.

        Parameters
        ----------
        query_text:
            Free-text search query.
        top_k:
            Maximum number of results to return.
        min_similarity:
            Minimum cosine similarity threshold (0.0–1.0).

        Returns
        -------
        list[tuple[str, float]]
            Pairs of ``(document_id, similarity_score)`` sorted by
            descending similarity.

        Raises
        ------
        RuntimeError
            If ``fit()`` has not been called yet.
        """
        if not self._is_fitted:
            raise RuntimeError(
                "EmbeddingSearch.fit() must be called before querying."
            )
        if not self._documents:
            return []

        query_tokens = self._tokenize(query_text)
        query_tf = self._compute_tf(query_tokens)
        query_vector: dict[str, float] = {
            term: query_tf.get(term, 0.0) * self._idf.get(term, 0.0)
            for term in query_tf
            if term in self._vocabulary
        }
        query_norm = math.sqrt(sum(v * v for v in query_vector.values())) or 1.0
        normalized_query = {t: v / query_norm for t, v in query_vector.items()}

        scores: list[tuple[str, float]] = []
        for doc_id, doc_vector in self._tfidf_matrix.items():
            similarity = self._cosine_similarity(normalized_query, doc_vector)
            if similarity >= min_similarity:
                scores.append((doc_id, similarity))

        scores.sort(key=lambda pair: pair[1], reverse=True)
        return scores[:top_k]

    def add_document(self, document_id: str, text: str) -> None:
        """Add or update a single document and re-fit the index.

        Parameters
        ----------
        document_id:
            Unique identifier for the document.
        text:
            Text content to index.
        """
        self._documents[document_id] = text
        self.fit(self._documents)

    def remove_document(self, document_id: str) -> None:
        """Remove a document from the index and re-fit.

        Parameters
        ----------
        document_id:
            The identifier to remove.

        Raises
        ------
        KeyError
            If the document_id is not in the index.
        """
        if document_id not in self._documents:
            raise KeyError(f"Document {document_id!r} not found in index.")
        del self._documents[document_id]
        self.fit(self._documents)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Lowercase, strip punctuation, and split into tokens."""
        text = text.lower()
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        tokens = text.split()
        # Remove single-character tokens
        return [t for t in tokens if len(t) > 1]

    @staticmethod
    def _compute_tf(tokens: list[str]) -> dict[str, float]:
        """Compute log-normalized term frequency."""
        if not tokens:
            return {}
        counts: Counter[str] = Counter(tokens)
        return {term: 1.0 + math.log(count) for term, count in counts.items()}

    @staticmethod
    def _cosine_similarity(vec_a: dict[str, float], vec_b: dict[str, float]) -> float:
        """Compute cosine similarity between two sparse vectors."""
        dot_product = sum(
            vec_a[term] * vec_b[term] for term in vec_a if term in vec_b
        )
        return dot_product
