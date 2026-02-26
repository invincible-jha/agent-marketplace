"""Analytics sub-package for agent-marketplace.

Provides usage tracking and marketplace reporting.

Public API
----------
- ``UsageTracker``       — records invocations and computes popularity/trends.
- ``UsageRecord``        — dataclass for a single usage event.
- ``MarketplaceReporter`` — generates summary and per-entity analytics reports.
"""
from __future__ import annotations

from agent_marketplace.analytics.reporter import MarketplaceReporter
from agent_marketplace.analytics.usage import UsageRecord, UsageTracker

__all__ = [
    "MarketplaceReporter",
    "UsageRecord",
    "UsageTracker",
]
