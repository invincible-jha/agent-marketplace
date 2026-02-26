"""Unit tests for agent_marketplace.schema.provider.ProviderInfo."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from agent_marketplace.schema.provider import ProviderInfo


# ---------------------------------------------------------------------------
# Construction — happy path
# ---------------------------------------------------------------------------


class TestProviderInfoConstruction:
    def test_minimal_creation_with_name_only(self) -> None:
        provider = ProviderInfo(name="Acme Corp")
        assert provider.name == "Acme Corp"
        assert provider.organization == ""
        assert provider.contact_email == ""
        assert provider.website == ""
        assert provider.github_handle == ""
        assert provider.huggingface_handle == ""
        assert provider.verified is False

    def test_full_creation_with_all_fields(self) -> None:
        provider = ProviderInfo(
            name="DataCo",
            organization="Data Corporation",
            contact_email="hello@dataco.example.com",
            website="https://dataco.example.com",
            github_handle="dataco-ai",
            huggingface_handle="dataco",
            verified=True,
        )
        assert provider.name == "DataCo"
        assert provider.organization == "Data Corporation"
        assert provider.contact_email == "hello@dataco.example.com"
        assert provider.verified is True

    def test_whitespace_in_name_is_stripped(self) -> None:
        provider = ProviderInfo(name="  Trimmed Name  ")
        assert provider.name == "Trimmed Name"

    def test_verified_defaults_to_false(self) -> None:
        provider = ProviderInfo(name="Some Provider")
        assert provider.verified is False


# ---------------------------------------------------------------------------
# Construction — validation errors
# ---------------------------------------------------------------------------


class TestProviderInfoValidation:
    def test_empty_name_raises_validation_error(self) -> None:
        with pytest.raises(ValidationError, match="Provider name must not be empty"):
            ProviderInfo(name="")

    def test_whitespace_only_name_raises_validation_error(self) -> None:
        with pytest.raises(ValidationError, match="Provider name must not be empty"):
            ProviderInfo(name="   ")

    def test_email_without_at_symbol_raises_validation_error(self) -> None:
        with pytest.raises(ValidationError, match="does not appear to be a valid email"):
            ProviderInfo(name="X", contact_email="not-an-email")

    def test_empty_email_is_allowed(self) -> None:
        provider = ProviderInfo(name="X", contact_email="")
        assert provider.contact_email == ""

    def test_valid_email_passes(self) -> None:
        provider = ProviderInfo(name="X", contact_email="user@example.com")
        assert provider.contact_email == "user@example.com"


# ---------------------------------------------------------------------------
# display_name method
# ---------------------------------------------------------------------------


class TestProviderInfoDisplayName:
    def test_display_name_without_organization(self) -> None:
        provider = ProviderInfo(name="Alice")
        assert provider.display_name() == "Alice"

    def test_display_name_with_organization(self) -> None:
        provider = ProviderInfo(name="Alice", organization="TechCorp")
        assert provider.display_name() == "Alice (TechCorp)"

    def test_str_returns_display_name(self) -> None:
        provider = ProviderInfo(name="Bob", organization="BuildCo")
        assert str(provider) == "Bob (BuildCo)"
