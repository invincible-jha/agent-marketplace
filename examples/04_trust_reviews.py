#!/usr/bin/env python3
"""Example: Trust and Reviews

Demonstrates the reputation system — submitting reviews, computing
trust scores, and tracking reputation over time.

Usage:
    python examples/04_trust_reviews.py

Requirements:
    pip install agent-marketplace
"""
from __future__ import annotations

import agent_marketplace
from agent_marketplace import (
    Review,
    ReviewStore,
    TrustScorer,
    ProviderTrustData,
    ReputationTracker,
)


def main() -> None:
    print(f"agent-marketplace version: {agent_marketplace.__version__}")

    provider_id = "nlp-labs-v2"

    # Step 1: Create review store and submit reviews
    review_store = ReviewStore()
    reviews = [
        Review(provider_id=provider_id, reviewer_id="agent-a", rating=5, comment="Excellent accuracy and speed."),
        Review(provider_id=provider_id, reviewer_id="agent-b", rating=4, comment="Good results, occasional latency spikes."),
        Review(provider_id=provider_id, reviewer_id="agent-c", rating=5, comment="Reliable and well-documented API."),
        Review(provider_id=provider_id, reviewer_id="agent-d", rating=3, comment="Acceptable but pricing is high."),
        Review(provider_id=provider_id, reviewer_id="agent-e", rating=4, comment="Solid performance for our use case."),
    ]

    for review in reviews:
        review_store.submit(review)

    print(f"Submitted {len(reviews)} reviews for '{provider_id}'.")

    # Step 2: Retrieve reviews
    provider_reviews = review_store.get_reviews(provider_id)
    avg_rating = sum(r.rating for r in provider_reviews) / len(provider_reviews)
    print(f"Average rating: {avg_rating:.2f}/5 ({len(provider_reviews)} reviews)")

    # Step 3: Compute trust score
    trust_data = ProviderTrustData(
        provider_id=provider_id,
        total_reviews=len(provider_reviews),
        average_rating=avg_rating,
        positive_count=sum(1 for r in provider_reviews if r.rating >= 4),
        negative_count=sum(1 for r in provider_reviews if r.rating <= 2),
    )
    scorer = TrustScorer()
    trust_score = scorer.score(trust_data)
    print(f"Trust score: {trust_score.value:.3f} (level={trust_score.level.value})")

    # Step 4: Reputation tracker
    tracker = ReputationTracker()
    tracker.update(provider_id=provider_id, trust_score=trust_score)
    reputation = tracker.get(provider_id)
    print(f"\nReputation tracker:")
    print(f"  Provider: {reputation.provider_id}")
    print(f"  Current score: {reputation.current_score:.3f}")
    print(f"  Trend: {reputation.trend}")

    # Step 5: Display review summaries
    print(f"\nRecent reviews:")
    for review in provider_reviews[-3:]:
        print(f"  [{review.rating}/5] {review.reviewer_id}: {review.comment[:60]}")


if __name__ == "__main__":
    main()
