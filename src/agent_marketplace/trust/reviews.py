"""Review storage for agent-marketplace trust layer.

Defines the ``Review`` dataclass and the ``ReviewStore`` in-memory CRUD
store used to accumulate user reviews for registered providers.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class Review:
    """A single user review of a capability provider.

    Attributes
    ----------
    reviewer_id:
        Unique identifier of the reviewer (e.g. user ID or email hash).
    provider_id:
        The provider name or identifier being reviewed.
    rating:
        Integer rating from 1 (worst) to 5 (best).
    text:
        Optional free-text comment from the reviewer.
    timestamp:
        UTC datetime when the review was submitted.
    review_id:
        Stable unique identifier for this review record (auto-generated).
    """

    reviewer_id: str
    provider_id: str
    rating: int
    text: str = ""
    timestamp: datetime = field(default_factory=lambda: datetime.now(tz=timezone.utc))
    review_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def __post_init__(self) -> None:
        if not self.reviewer_id.strip():
            raise ValueError("reviewer_id must not be empty.")
        if not self.provider_id.strip():
            raise ValueError("provider_id must not be empty.")
        if not (1 <= self.rating <= 5):
            raise ValueError(
                f"rating must be between 1 and 5 inclusive, got {self.rating!r}."
            )


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------


class ReviewStore:
    """In-memory CRUD store for provider reviews.

    All reviews are keyed by their ``review_id``.  Provider-scoped
    queries are performed by linear scan; for production workloads
    consider wrapping with a persistent backend.

    Usage
    -----
    ::

        store = ReviewStore()
        review = Review(reviewer_id="user-1", provider_id="acme", rating=4)
        store.add(review)
        avg = store.average_rating("acme")
    """

    def __init__(self) -> None:
        self._reviews: dict[str, Review] = {}

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def add(self, review: Review) -> None:
        """Persist a new review.

        Parameters
        ----------
        review:
            The review to store.

        Raises
        ------
        ValueError
            If a review with the same ``review_id`` already exists.
        """
        if review.review_id in self._reviews:
            raise ValueError(
                f"Review {review.review_id!r} already exists. "
                "Each review must have a unique review_id."
            )
        self._reviews[review.review_id] = review

    def get(self, review_id: str) -> Review:
        """Retrieve a review by its unique identifier.

        Parameters
        ----------
        review_id:
            The auto-generated review identifier.

        Raises
        ------
        KeyError
            If no review with this identifier exists.
        """
        try:
            return self._reviews[review_id]
        except KeyError:
            raise KeyError(f"Review {review_id!r} not found.") from None

    def update(self, review: Review) -> None:
        """Replace an existing review record.

        Parameters
        ----------
        review:
            Updated review.  ``review.review_id`` must already exist.

        Raises
        ------
        KeyError
            If the review does not exist.
        """
        if review.review_id not in self._reviews:
            raise KeyError(
                f"Review {review.review_id!r} not found. Use add() to create it."
            )
        self._reviews[review.review_id] = review

    def delete(self, review_id: str) -> None:
        """Remove a review from the store.

        Parameters
        ----------
        review_id:
            Identifier of the review to remove.

        Raises
        ------
        KeyError
            If the review does not exist.
        """
        if review_id not in self._reviews:
            raise KeyError(f"Review {review_id!r} not found.")
        del self._reviews[review_id]

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def list_for_provider(self, provider_id: str) -> list[Review]:
        """Return all reviews for a given provider in chronological order.

        Parameters
        ----------
        provider_id:
            The provider whose reviews to retrieve.

        Returns
        -------
        list[Review]
            Reviews sorted oldest-first.
        """
        matching = [r for r in self._reviews.values() if r.provider_id == provider_id]
        matching.sort(key=lambda r: r.timestamp)
        return matching

    def average_rating(self, provider_id: str) -> float:
        """Compute the mean rating for a provider.

        Parameters
        ----------
        provider_id:
            The provider to compute the average for.

        Returns
        -------
        float
            Mean rating in [1.0, 5.0], or 0.0 if no reviews exist.
        """
        reviews = self.list_for_provider(provider_id)
        if not reviews:
            return 0.0
        return sum(r.rating for r in reviews) / len(reviews)

    def count_for_provider(self, provider_id: str) -> int:
        """Return the number of reviews for a provider."""
        return sum(1 for r in self._reviews.values() if r.provider_id == provider_id)

    def list_all(self) -> list[Review]:
        """Return all reviews in the store, sorted oldest-first."""
        return sorted(self._reviews.values(), key=lambda r: r.timestamp)

    def __len__(self) -> int:
        return len(self._reviews)
