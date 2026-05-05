# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Mirrowel

from datetime import datetime
import logging
import re
from typing import Any

from .provider_urls import get_provider_endpoint

REDACTED = "[REDACTED]"
_SENSITIVE_HEADER_NAMES = {
    "authorization",
    "proxy-authorization",
    "x-api-key",
    "api-key",
    "cookie",
    "set-cookie",
}
_SENSITIVE_KEY_FRAGMENTS = {
    "authorization",
    "api_key",
    "apikey",
    "access_token",
    "refresh_token",
    "id_token",
    "session",
    "cookie",
    "secret",
    "password",
    "token",
}
_TOKEN_PATTERNS = [
    re.compile(r"(?i)^bearer\s+.+$"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{16,}\b"),
    re.compile(r"\bpk_[A-Za-z0-9_-]{16,}\b"),
    re.compile(r"\b[A-Za-z0-9_-]{20,}\.[A-Za-z0-9_-]{20,}\.[A-Za-z0-9_-]{20,}\b"),
]


def _looks_like_token(value: str) -> bool:
    for pattern in _TOKEN_PATTERNS:
        if pattern.search(value):
            return True
    return False


def _is_sensitive_key(key: str | None) -> bool:
    if not key:
        return False
    normalized = key.lower().replace("-", "_")
    return any(fragment in normalized for fragment in _SENSITIVE_KEY_FRAGMENTS)


def redact_sensitive_value(value: Any, *, key_hint: str | None = None) -> Any:
    if value is None:
        return None
    if isinstance(value, str):
        if _is_sensitive_key(key_hint) or _looks_like_token(value):
            return REDACTED
        return value
    if isinstance(value, bytes):
        return REDACTED
    return value


def redact_sensitive_headers(headers: dict[str, Any] | None) -> dict[str, Any] | None:
    if headers is None:
        return None
    redacted: dict[str, Any] = {}
    for key, value in headers.items():
        if key.lower() in _SENSITIVE_HEADER_NAMES:
            redacted[key] = REDACTED
        else:
            redacted[key] = redact_sensitive_value(value, key_hint=key)
    return redacted


def redact_sensitive_data(value: Any, *, key_hint: str | None = None) -> Any:
    if isinstance(value, dict):
        redacted_dict = {}
        for key, nested_value in value.items():
            if _is_sensitive_key(key):
                redacted_dict[key] = REDACTED
            else:
                redacted_dict[key] = redact_sensitive_data(nested_value, key_hint=key)
        return redacted_dict
    if isinstance(value, list):
        return [redact_sensitive_data(item, key_hint=key_hint) for item in value]
    return redact_sensitive_value(value, key_hint=key_hint)

def log_request_to_console(url: str, headers: dict, client_info: tuple, request_data: dict):
    """
    Logs a concise, single-line summary of an incoming request to the console.
    """
    time_str = datetime.now().strftime("%H:%M")
    model_full = request_data.get("model", "N/A")
    
    provider = "N/A"
    model_name = model_full
    endpoint_url = "N/A"

    if '/' in model_full:
        parts = model_full.split('/', 1)
        provider = parts[0]
        model_name = parts[1]
        # Use the helper function to get the full endpoint URL
        endpoint_url = get_provider_endpoint(provider, model_name, url) or "N/A"

    log_message = f"{time_str} - {client_info[0]}:{client_info[1]} - provider: {provider}, model: {model_name} - {endpoint_url}"
    logging.info(log_message)
