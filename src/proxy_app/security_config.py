import logging
import os
from dataclasses import dataclass


DEFAULT_SESSION_SECRET = "change-me-session-secret"
DEFAULT_API_TOKEN_PEPPER = "change-me-token-pepper"


class SecurityValidationError(RuntimeError):
    pass


def parse_bool_env(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def get_app_env() -> str:
    value = (os.getenv("APP_ENV") or "dev").strip().lower()
    if value in {"prod", "production"}:
        return "prod"
    return "dev"


def is_prod() -> bool:
    return get_app_env() == "prod"


def allow_insecure_defaults() -> bool:
    default = not is_prod()
    return parse_bool_env("ALLOW_INSECURE_DEFAULTS", default)


def get_session_secret() -> str:
    return (os.getenv("SESSION_SECRET") or "").strip() or DEFAULT_SESSION_SECRET


def get_api_token_pepper() -> str:
    return (
        (os.getenv("API_TOKEN_PEPPER") or "").strip() or DEFAULT_API_TOKEN_PEPPER
    )


def validate_secret_settings() -> None:
    if allow_insecure_defaults():
        if get_session_secret() == DEFAULT_SESSION_SECRET:
            logging.warning(
                "SECURITY WARNING: SESSION_SECRET is using default value. "
                "Set SESSION_SECRET for non-local usage."
            )
        if get_api_token_pepper() == DEFAULT_API_TOKEN_PEPPER:
            logging.warning(
                "SECURITY WARNING: API_TOKEN_PEPPER is using default value. "
                "Set API_TOKEN_PEPPER for non-local usage."
            )
        return

    invalid_reasons: list[str] = []
    if get_session_secret() == DEFAULT_SESSION_SECRET:
        invalid_reasons.append("SESSION_SECRET is missing or default")
    if get_api_token_pepper() == DEFAULT_API_TOKEN_PEPPER:
        invalid_reasons.append("API_TOKEN_PEPPER is missing or default")

    if invalid_reasons:
        raise SecurityValidationError(
            "Refusing startup due to insecure defaults: "
            + "; ".join(invalid_reasons)
            + ". Set secure secrets or ALLOW_INSECURE_DEFAULTS=true explicitly."
        )


@dataclass(frozen=True)
class CORSSettings:
    allow_origins: list[str]
    allow_credentials: bool
    allow_methods: list[str]
    allow_headers: list[str]


def _split_csv_env(name: str, default: str) -> list[str]:
    raw = os.getenv(name, default)
    values = [value.strip() for value in raw.split(",")]
    return [value for value in values if value]


def get_cors_settings() -> CORSSettings:
    origins = _split_csv_env("CORS_ALLOW_ORIGINS", "")
    credentials_default = bool(origins)
    allow_credentials = parse_bool_env("CORS_ALLOW_CREDENTIALS", credentials_default)

    if not origins:
        allow_credentials = False

    if allow_credentials and "*" in origins:
        raise SecurityValidationError(
            "Invalid CORS config: CORS_ALLOW_ORIGINS cannot include '*' when "
            "CORS_ALLOW_CREDENTIALS=true."
        )

    return CORSSettings(
        allow_origins=origins,
        allow_credentials=allow_credentials,
        allow_methods=_split_csv_env(
            "CORS_ALLOW_METHODS", "GET,POST,PUT,PATCH,DELETE,OPTIONS"
        ),
        allow_headers=_split_csv_env(
            "CORS_ALLOW_HEADERS",
            "Authorization,Content-Type,X-API-Key,X-Requested-With,X-CSRF-Token",
        ),
    )
