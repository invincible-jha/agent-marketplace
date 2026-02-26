"""Provider information schema for agent-marketplace.

Defines the ProviderInfo dataclass that identifies the organization or
individual who publishes an agent capability.
"""
from __future__ import annotations

from pydantic import BaseModel, HttpUrl, field_validator


class ProviderInfo(BaseModel):
    """Identity and contact information for a capability provider.

    Attributes
    ----------
    name:
        Human-readable name of the provider (person or organization).
    organization:
        Optional formal organization name.
    contact_email:
        Email address for support or inquiries.
    website:
        Optional provider website URL.
    github_handle:
        Optional GitHub username or org handle (used for identity verification).
    huggingface_handle:
        Optional Hugging Face username or org handle.
    verified:
        Whether the provider identity has been externally verified.
    """

    name: str
    organization: str = ""
    contact_email: str = ""
    website: str = ""
    github_handle: str = ""
    huggingface_handle: str = ""
    verified: bool = False

    @field_validator("name")
    @classmethod
    def name_must_not_be_empty(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("Provider name must not be empty or whitespace.")
        return stripped

    @field_validator("contact_email")
    @classmethod
    def validate_email_format(cls, value: str) -> str:
        if value and "@" not in value:
            raise ValueError(f"contact_email {value!r} does not appear to be a valid email address.")
        return value

    def display_name(self) -> str:
        """Return a display-friendly name combining name and organization."""
        if self.organization:
            return f"{self.name} ({self.organization})"
        return self.name

    def __str__(self) -> str:
        return self.display_name()
