"""Capability request model for agent-marketplace matching.

A ``CapabilityRequest`` describes what an agent needs: the required
capability types, acceptable latency, cost budget, minimum trust, and
any certification requirements.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class CapabilityRequest:
    """A structured request for one or more capabilities.

    Attributes
    ----------
    required_capabilities:
        List of capability names, tags, or category strings that the
        fulfilling provider must offer.  At least one entry is required.
    preferred_latency_ms:
        Soft preference for median (p50) latency in milliseconds.
        Providers meeting this threshold receive a scoring boost.
        Use 0.0 to indicate no preference.
    max_cost:
        Maximum acceptable cost per call in USD.  Providers with
        ``cost > max_cost`` are excluded.
    min_trust:
        Minimum acceptable trust level [0.0, 1.0].  Providers with
        ``trust_level < min_trust`` are excluded.
    required_certifications:
        Optional list of certification strings the provider must declare
        (e.g. ``["SOC2", "GDPR"]``).
    request_id:
        Optional caller-supplied identifier for correlation and logging.
    """

    required_capabilities: list[str]
    preferred_latency_ms: float = 0.0
    max_cost: float = float("inf")
    min_trust: float = 0.0
    required_certifications: list[str] = field(default_factory=list)
    request_id: str = ""

    def __post_init__(self) -> None:
        if not self.required_capabilities:
            raise ValueError(
                "required_capabilities must contain at least one entry."
            )
        if self.preferred_latency_ms < 0:
            raise ValueError(
                f"preferred_latency_ms must be non-negative, "
                f"got {self.preferred_latency_ms!r}."
            )
        if self.max_cost < 0:
            raise ValueError(
                f"max_cost must be non-negative, got {self.max_cost!r}."
            )
        if not (0.0 <= self.min_trust <= 1.0):
            raise ValueError(
                f"min_trust must be in [0.0, 1.0], got {self.min_trust!r}."
            )
