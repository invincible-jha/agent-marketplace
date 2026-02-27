"""Pydantic request/response models for the agent-marketplace HTTP server."""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class RegisterCapabilityRequest(BaseModel):
    """Request body for POST /register."""

    name: str
    version: str
    description: str
    category: str
    tags: list[str] = Field(default_factory=list)
    input_types: list[str] = Field(default_factory=list)
    output_type: str = ""
    pricing_model: str = "free"
    cost: float = 0.0
    supported_languages: list[str] = Field(default_factory=list)
    supported_frameworks: list[str] = Field(default_factory=list)
    provider: dict[str, object] = Field(default_factory=dict)
    quality_metrics: dict[str, object] = Field(default_factory=dict)


class CapabilityResponse(BaseModel):
    """Response body representing a single marketplace capability."""

    capability_id: str
    name: str
    version: str
    description: str
    category: str
    tags: list[str] = Field(default_factory=list)
    input_types: list[str] = Field(default_factory=list)
    output_type: str
    pricing_model: str
    cost: float
    trust_level: float
    supported_languages: list[str] = Field(default_factory=list)
    supported_frameworks: list[str] = Field(default_factory=list)


class SearchResponse(BaseModel):
    """Response body for GET /search."""

    query: str
    results: list[CapabilityResponse] = Field(default_factory=list)
    total: int = 0
    limit: int = 20
    offset: int = 0


class RegisterResponse(BaseModel):
    """Response body for POST /register."""

    capability_id: str
    registered: bool = True
    warnings: list[str] = Field(default_factory=list)


class HealthResponse(BaseModel):
    """Response body for GET /health."""

    status: str = "ok"
    service: str = "agent-marketplace"
    version: str = "0.1.0"
    capability_count: int = 0


class ErrorResponse(BaseModel):
    """Standard error response body."""

    error: str
    detail: str = ""


__all__ = [
    "RegisterCapabilityRequest",
    "CapabilityResponse",
    "SearchResponse",
    "RegisterResponse",
    "HealthResponse",
    "ErrorResponse",
]
