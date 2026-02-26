"""Matching sub-package for agent-marketplace.

Provides request-to-provider matching, price negotiation, and the
supporting data models.

Public API
----------
- ``MatchingEngine``  — matches requests to provider capabilities.
- ``MatchResult``     — a capability with its composite match score.
- ``CapabilityRequest`` — structured capability request descriptor.
- ``PriceNegotiator`` — selects the best-value offer within budget.
- ``PriceOffer``      — a cost offer from a specific provider.
- ``NegotiationResult`` — outcome of a price negotiation.
"""
from __future__ import annotations

from agent_marketplace.matching.engine import MatchResult, MatchingEngine
from agent_marketplace.matching.negotiator import (
    NegotiationResult,
    PriceNegotiator,
    PriceOffer,
)
from agent_marketplace.matching.request import CapabilityRequest

__all__ = [
    "CapabilityRequest",
    "MatchResult",
    "MatchingEngine",
    "NegotiationResult",
    "PriceNegotiator",
    "PriceOffer",
]
