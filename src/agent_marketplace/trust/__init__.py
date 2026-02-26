"""Trust sub-package for agent-marketplace.

Provides trust scoring, review storage, and reputation tracking for
registered providers and capabilities.

Public API
----------
- ``TrustScorer`` — computes a composite trust score for a provider.
- ``ReviewStore`` — CRUD store for user reviews.
- ``Review`` — dataclass representing a single review.
- ``ReputationTracker`` — sliding-window reputation tracker.
"""
from __future__ import annotations

from agent_marketplace.trust.reputation import ReputationTracker
from agent_marketplace.trust.reviews import Review, ReviewStore
from agent_marketplace.trust.scorer import TrustScorer

__all__ = [
    "ReputationTracker",
    "Review",
    "ReviewStore",
    "TrustScorer",
]
