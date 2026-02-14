import pytest

from proxy_app.security_config import (
    SecurityValidationError,
    get_cors_settings,
    validate_secret_settings,
)


def test_cors_defaults_are_safe(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CORS_ALLOW_ORIGINS", raising=False)
    monkeypatch.delenv("CORS_ALLOW_CREDENTIALS", raising=False)

    settings = get_cors_settings()

    assert settings.allow_origins == []
    assert settings.allow_credentials is False


def test_cors_rejects_wildcard_with_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CORS_ALLOW_ORIGINS", "*")
    monkeypatch.setenv("CORS_ALLOW_CREDENTIALS", "true")

    with pytest.raises(SecurityValidationError):
        get_cors_settings()


def test_prod_fails_with_default_secrets(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("APP_ENV", "prod")
    monkeypatch.setenv("ALLOW_INSECURE_DEFAULTS", "false")
    monkeypatch.delenv("SESSION_SECRET", raising=False)
    monkeypatch.delenv("API_TOKEN_PEPPER", raising=False)

    with pytest.raises(SecurityValidationError):
        validate_secret_settings()


def test_prod_explicit_override_allows_insecure_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("APP_ENV", "prod")
    monkeypatch.setenv("ALLOW_INSECURE_DEFAULTS", "true")
    monkeypatch.delenv("SESSION_SECRET", raising=False)
    monkeypatch.delenv("API_TOKEN_PEPPER", raising=False)

    validate_secret_settings()
