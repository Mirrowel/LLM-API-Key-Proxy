# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""Optional structured configuration for experimental native features.

The proxy remains environment-first: existing `.env` variables keep working and
environment variables override this JSON layer. This module intentionally avoids
secrets. API keys, OAuth tokens, bearer headers, and similar values must remain
in environment variables or provider-managed credential files.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional

from ..field_cache import FieldCacheInjection, FieldCacheRule
from ..usage.costs import ModelPricing

_CONFIG_ENV_KEYS = ("LLM_PROXY_CONFIG_FILE", "PROXY_CONFIG_FILE")
_KNOWN_SECTIONS = {"routing", "pricing", "streaming", "field_cache", "providers", "retry", "responses"}
_SECRET_KEY_PARTS = ("api_key", "authorization", "access_token", "refresh_token", "client_secret", "bearer_token", "password")


class ExperimentalConfigError(ValueError):
    """Raised when optional structured config is malformed or unsafe."""


@dataclass(frozen=True)
class ExperimentalConfig:
    """Parsed optional JSON config.

    Sections are stored as dictionaries rather than deep custom classes so Phase
    10 can layer config onto existing feature-specific parsers without creating
    a second full application configuration system.
    """

    routing: dict[str, Any] = field(default_factory=dict)
    pricing: dict[str, Any] = field(default_factory=dict)
    streaming: dict[str, Any] = field(default_factory=dict)
    field_cache: dict[str, Any] = field(default_factory=dict)
    providers: dict[str, Any] = field(default_factory=dict)
    retry: dict[str, Any] = field(default_factory=dict)
    responses: dict[str, Any] = field(default_factory=dict)
    unknown_sections: dict[str, Any] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()
    path: Optional[str] = None

    @property
    def is_empty(self) -> bool:
        return not (self.routing or self.pricing or self.streaming or self.field_cache or self.providers or self.retry or self.responses or self.unknown_sections)


@dataclass(frozen=True)
class StreamRuntimeSettings:
    """Runtime stream observability settings.

    Timeout and heartbeat values default to disabled so existing long-running
    reasoning streams keep working. Operators can opt into active stream
    hardening through env or JSON config without changing provider code.
    """

    ttfb_timeout_seconds: Optional[float] = None
    stall_timeout_seconds: Optional[float] = None
    heartbeat_seconds: Optional[float] = None
    cancel_upstream_on_disconnect: bool = True
    trace_metrics: bool = True


@dataclass(frozen=True)
class RetryRuntimeSettings:
    """Runtime retry/cooldown settings layered from JSON and env."""

    provider_cooldown_min_seconds: int = 10
    provider_cooldown_default_seconds: int = 30
    provider_cooldown_on_quota: bool = False
    provider_backoff_window_seconds: int = 60
    provider_backoff_threshold: int = 3
    provider_backoff_base_seconds: Optional[int] = None
    provider_backoff_max_seconds: int = 300
    failure_history_max_entries: int = 200


def load_experimental_config(path: str | os.PathLike[str] | None = None, env: Mapping[str, str] | None = None) -> ExperimentalConfig:
    """Load optional JSON config from an explicit path or config env var."""

    source = env if env is not None else os.environ
    resolved = Path(path) if path is not None else _path_from_env(source)
    if resolved is None or not resolved.exists():
        return ExperimentalConfig(path=str(resolved) if resolved is not None else None)
    try:
        data = json.loads(resolved.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ExperimentalConfigError(f"Invalid JSON config at {resolved}: {exc.msg}") from exc
    if not isinstance(data, dict):
        raise ExperimentalConfigError("JSON config root must be an object")
    _reject_secret_keys(data)
    warnings = tuple(f"Unknown config section '{key}' ignored by current runtime" for key in data if key not in _KNOWN_SECTIONS)
    unknown = {key: value for key, value in data.items() if key not in _KNOWN_SECTIONS}
    return ExperimentalConfig(
        routing=_dict_section(data, "routing"),
        pricing=_dict_section(data, "pricing"),
        streaming=_dict_section(data, "streaming"),
        field_cache=_dict_section(data, "field_cache"),
        providers=_dict_section(data, "providers"),
        retry=_dict_section(data, "retry"),
        responses=_dict_section(data, "responses"),
        unknown_sections=unknown,
        warnings=warnings,
        path=str(resolved),
    )


def load_config_from_mapping(data: Mapping[str, Any]) -> ExperimentalConfig:
    """Build config from an in-memory mapping for tests and provider helpers."""

    _reject_secret_keys(data)
    warnings = tuple(f"Unknown config section '{key}' ignored by current runtime" for key in data if key not in _KNOWN_SECTIONS)
    return ExperimentalConfig(
        routing=_dict_section(data, "routing"),
        pricing=_dict_section(data, "pricing"),
        streaming=_dict_section(data, "streaming"),
        field_cache=_dict_section(data, "field_cache"),
        providers=_dict_section(data, "providers"),
        retry=_dict_section(data, "retry"),
        responses=_dict_section(data, "responses"),
        unknown_sections={key: value for key, value in data.items() if key not in _KNOWN_SECTIONS},
        warnings=warnings,
    )


def get_configured_model_pricing(
    provider: str,
    model: str,
    *,
    config: ExperimentalConfig | None = None,
    env: Mapping[str, str] | None = None,
) -> Optional[ModelPricing]:
    """Return JSON/env pricing for a provider/model, with env taking priority."""

    source = env if env is not None else os.environ
    env_pricing = _pricing_from_env(provider, model, source)
    if env_pricing:
        return env_pricing
    active = config if config is not None else load_experimental_config(env=source)
    pricing_section = active.pricing.get(provider, {}) if isinstance(active.pricing, dict) else {}
    raw = pricing_section.get(model) if isinstance(pricing_section, dict) else None
    if not isinstance(raw, dict):
        return None
    return ModelPricing(
        input_cost_per_token=as_float(raw.get("input", raw.get("input_cost_per_token", 0.0)), name="pricing.input"),
        output_cost_per_token=as_float(raw.get("output", raw.get("output_cost_per_token", 0.0)), name="pricing.output"),
        cache_read_cost_per_token=as_float(raw.get("cache_read", raw.get("cache_read_cost_per_token", 0.0)), name="pricing.cache_read"),
        cache_write_cost_per_token=as_float(raw.get("cache_write", raw.get("cache_write_cost_per_token", 0.0)), name="pricing.cache_write"),
        reasoning_cost_per_token=as_float(raw.get("reasoning", raw.get("reasoning_cost_per_token", 0.0)), name="pricing.reasoning"),
        currency=str(raw.get("currency", "USD")),
        source="json_config",
    )


def get_stream_runtime_settings(
    *,
    config: ExperimentalConfig | None = None,
    env: Mapping[str, str] | None = None,
) -> StreamRuntimeSettings:
    """Return stream runtime settings with environment overriding JSON."""

    source = env if env is not None else os.environ
    active = config if config is not None else load_experimental_config(env=source)
    streaming = active.streaming if isinstance(active.streaming, dict) else {}

    return StreamRuntimeSettings(
        ttfb_timeout_seconds=_optional_positive_float(_env_or_json(source, "STREAM_TTFB_TIMEOUT_SECONDS", streaming, "ttfb_timeout_seconds"), "STREAM_TTFB_TIMEOUT_SECONDS"),
        stall_timeout_seconds=_optional_positive_float(_env_or_json(source, "STREAM_STALL_TIMEOUT_SECONDS", streaming, "stall_timeout_seconds"), "STREAM_STALL_TIMEOUT_SECONDS"),
        heartbeat_seconds=_optional_positive_float(_env_or_json(source, "STREAM_HEARTBEAT_INTERVAL_SECONDS", streaming, "heartbeat_interval_seconds", default=_env_or_json(source, "STREAM_HEARTBEAT_SECONDS", streaming, "heartbeat_seconds")), "STREAM_HEARTBEAT_INTERVAL_SECONDS"),
        cancel_upstream_on_disconnect=as_bool(_env_or_json(source, "STREAM_CANCEL_UPSTREAM_ON_DISCONNECT", streaming, "cancel_upstream_on_disconnect", default=True), name="STREAM_CANCEL_UPSTREAM_ON_DISCONNECT"),
        trace_metrics=as_bool(_env_or_json(source, "STREAM_TRACE_METRICS", streaming, "trace_metrics", default=True), name="STREAM_TRACE_METRICS"),
    )


def get_retry_runtime_settings(
    *,
    config: ExperimentalConfig | None = None,
    env: Mapping[str, str] | None = None,
) -> RetryRuntimeSettings:
    """Return retry/cooldown settings with environment overriding JSON."""

    source = env if env is not None else os.environ
    active = config if config is not None else load_experimental_config(env=source)
    retry = active.retry if isinstance(active.retry, dict) else {}
    cooldown = retry.get("provider_cooldown", {}) if isinstance(retry.get("provider_cooldown"), dict) else retry
    backoff = retry.get("backoff", {}) if isinstance(retry.get("backoff"), dict) else retry
    return RetryRuntimeSettings(
        provider_cooldown_min_seconds=max(0, as_int(_env_or_json(source, "PROVIDER_COOLDOWN_MIN_SECONDS", cooldown, "provider_cooldown_min_seconds", default=10), name="PROVIDER_COOLDOWN_MIN_SECONDS")),
        provider_cooldown_default_seconds=max(0, as_int(_env_or_json(source, "PROVIDER_COOLDOWN_DEFAULT_SECONDS", cooldown, "provider_cooldown_default_seconds", default=30), name="PROVIDER_COOLDOWN_DEFAULT_SECONDS")),
        provider_cooldown_on_quota=as_bool(_env_or_json(source, "PROVIDER_COOLDOWN_ON_QUOTA", cooldown, "provider_cooldown_on_quota", default=False), name="PROVIDER_COOLDOWN_ON_QUOTA"),
        provider_backoff_window_seconds=max(0, as_int(_env_or_json(source, "PROVIDER_BACKOFF_WINDOW_SECONDS", backoff, "provider_backoff_window_seconds", default=60), name="PROVIDER_BACKOFF_WINDOW_SECONDS")),
        provider_backoff_threshold=max(1, as_int(_env_or_json(source, "PROVIDER_BACKOFF_THRESHOLD", backoff, "provider_backoff_threshold", default=3), name="PROVIDER_BACKOFF_THRESHOLD")),
        provider_backoff_base_seconds=_optional_positive_int(_env_or_json(source, "PROVIDER_BACKOFF_BASE_SECONDS", backoff, "provider_backoff_base_seconds"), "PROVIDER_BACKOFF_BASE_SECONDS"),
        provider_backoff_max_seconds=max(1, as_int(_env_or_json(source, "PROVIDER_BACKOFF_MAX_SECONDS", backoff, "provider_backoff_max_seconds", default=300), name="PROVIDER_BACKOFF_MAX_SECONDS")),
        failure_history_max_entries=max(1, as_int(_env_or_json(source, "FAILURE_HISTORY_MAX_ENTRIES", backoff, "failure_history_max_entries", default=200), name="FAILURE_HISTORY_MAX_ENTRIES")),
    )


def get_responses_store_settings(
    *,
    config: ExperimentalConfig | None = None,
    env: Mapping[str, str] | None = None,
) -> Any:
    """Return Responses store settings with environment overriding JSON."""

    from ..responses import ResponsesStoreSettings

    source = env if env is not None else os.environ
    active = config if config is not None else load_experimental_config(env=source)
    responses = active.responses if isinstance(active.responses, dict) else {}
    store = responses.get("store", {}) if isinstance(responses.get("store"), dict) else responses
    ttl_seconds = _optional_positive_int(_env_or_json(source, "RESPONSES_STORE_TTL_SECONDS", store, "ttl_seconds"), "RESPONSES_STORE_TTL_SECONDS")
    max_items = _optional_positive_int(_env_or_json(source, "RESPONSES_STORE_MAX_ITEMS", store, "max_items"), "RESPONSES_STORE_MAX_ITEMS")
    return ResponsesStoreSettings(
        ttl_seconds=ttl_seconds,
        max_items=max_items,
        store_failed=as_bool(_env_or_json(source, "RESPONSES_STORE_FAILED", store, "store_failed", default=True), name="RESPONSES_STORE_FAILED"),
        store_in_progress=as_bool(_env_or_json(source, "RESPONSES_STORE_IN_PROGRESS", store, "store_in_progress", default=False), name="RESPONSES_STORE_IN_PROGRESS"),
    )


def parse_field_cache_rules(config: ExperimentalConfig, provider: str, model: str) -> tuple[FieldCacheRule, ...]:
    """Parse configured field-cache rules for a provider/model.

    Wildcard model rules are returned before exact model rules so providers can
    define general preservation behavior and then append model-specific rules.
    This helper is intentionally not auto-wired into providers; providers decide
    whether external config is appropriate for their protocol state.
    """

    provider_rules = config.field_cache.get(provider, {}) if isinstance(config.field_cache, dict) else {}
    if not isinstance(provider_rules, dict):
        return ()
    raw_rules: list[Any] = []
    keys = ["*"]
    if "/" in model:
        keys.append(model.split("/", 1)[1])
    keys.append(model)
    for key in dict.fromkeys(keys):
        value = provider_rules.get(key, [])
        if isinstance(value, list):
            raw_rules.extend(value)
    return tuple(_field_cache_rule_from_dict(rule) for rule in raw_rules if isinstance(rule, dict))


def as_bool(value: Any, *, name: str) -> bool:
    """Parse a JSON/env boolean value."""

    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"1", "true", "yes", "on"}:
            return True
        if text in {"0", "false", "no", "off"}:
            return False
    raise ExperimentalConfigError(f"Invalid boolean for {name}")


def as_float(value: Any, *, name: str) -> float:
    """Parse a JSON/env float value with redacted errors."""

    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ExperimentalConfigError(f"Invalid number for {name}") from exc


def as_int(value: Any, *, name: str) -> int:
    """Parse a JSON/env integer value with redacted errors."""

    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ExperimentalConfigError(f"Invalid integer for {name}") from exc


def env_price_key(provider: str, model: str, suffix: str) -> str:
    """Return normalized model-price environment variable name."""

    return f"MODEL_PRICE_{_env_part(provider)}_{_env_part(model)}_{_env_part(suffix)}"


def _path_from_env(env: Mapping[str, str]) -> Optional[Path]:
    for key in _CONFIG_ENV_KEYS:
        value = env.get(key)
        if value:
            return Path(value)
    return None


def _dict_section(data: Mapping[str, Any], key: str) -> dict[str, Any]:
    value = data.get(key, {})
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ExperimentalConfigError(f"Config section '{key}' must be an object")
    return dict(value)


def _reject_secret_keys(value: Any, path: str = "config") -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            key_text = str(key).lower()
            if any(part in key_text for part in _SECRET_KEY_PARTS):
                raise ExperimentalConfigError(f"Unsafe secret-like key in JSON config at {path}.{key}")
            _reject_secret_keys(nested, f"{path}.{key}")
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            _reject_secret_keys(nested, f"{path}[{index}]")


def _pricing_from_env(provider: str, model: str, env: Mapping[str, str]) -> Optional[ModelPricing]:
    suffixes = {
        "input": "INPUT",
        "output": "OUTPUT",
        "cache_read": "CACHE_READ",
        "cache_write": "CACHE_WRITE",
        "reasoning": "REASONING",
    }
    values: dict[str, float] = {}
    for field_name, suffix in suffixes.items():
        key = env_price_key(provider, model, suffix)
        raw = env.get(key)
        if raw not in (None, ""):
            values[field_name] = as_float(raw, name=key)
    if not values:
        return None
    return ModelPricing(
        input_cost_per_token=values.get("input", 0.0),
        output_cost_per_token=values.get("output", 0.0),
        cache_read_cost_per_token=values.get("cache_read", 0.0),
        cache_write_cost_per_token=values.get("cache_write", 0.0),
        reasoning_cost_per_token=values.get("reasoning", 0.0),
        source="env",
    )


def _env_or_json(env: Mapping[str, str], env_key: str, data: Mapping[str, Any], json_key: str, default: Any = None) -> Any:
    if env_key in env:
        return env[env_key]
    return data.get(json_key, default)


def _optional_positive_float(value: Any, name: str) -> Optional[float]:
    if value in (None, ""):
        return None
    parsed = as_float(value, name=name)
    # Zero and negative values mean "not configured" for timeout-like knobs.
    # Runtime enforcement is intentionally disabled by default.
    return parsed if parsed > 0 else None


def _optional_positive_int(value: Any, name: str) -> Optional[int]:
    if value in (None, ""):
        return None
    parsed = as_int(value, name=name)
    return parsed if parsed > 0 else None


def _field_cache_rule_from_dict(data: Mapping[str, Any]) -> FieldCacheRule:
    inject_data = data.get("inject")
    inject = None
    if isinstance(inject_data, Mapping):
        inject = FieldCacheInjection(
            target=str(inject_data.get("target", "request")),
            path=str(inject_data.get("path", data.get("target_path", ""))),
            when_missing_only=as_bool(inject_data.get("when_missing_only", False), name="field_cache.inject.when_missing_only"),
            insert=as_bool(inject_data.get("insert", False), name="field_cache.inject.insert"),
            as_list=as_bool(inject_data.get("as_list", False), name="field_cache.inject.as_list"),
        )
    elif data.get("target_path"):
        inject = FieldCacheInjection(target=str(data.get("target", "request")), path=str(data["target_path"]))
    scope = data.get("scope", ("provider", "model", "classifier", "session"))
    if isinstance(scope, str):
        scope_values = tuple(part.strip() for part in scope.split(",") if part.strip())
    elif isinstance(scope, (list, tuple)):
        scope_values = tuple(str(part) for part in scope)
    else:
        raise ExperimentalConfigError("field_cache.scope must be a string or list")
    try:
        return FieldCacheRule(
            name=str(data["name"]),
            source=str(data["source"]),
            path=str(data["path"]),
            mode=str(data.get("mode", "last")),
            scope=scope_values,
            inject=inject,
            enabled=as_bool(data.get("enabled", True), name="field_cache.enabled"),
            ttl_seconds=int(data["ttl_seconds"]) if data.get("ttl_seconds") is not None else None,
            metadata=dict(data.get("metadata", {})) if isinstance(data.get("metadata", {}), dict) else {},
            allow_missing_session=as_bool(data.get("allow_missing_session", False), name="field_cache.allow_missing_session"),
        )
    except KeyError as exc:
        raise ExperimentalConfigError(f"Missing field-cache rule key {exc.args[0]}") from exc
    except ValueError as exc:
        raise ExperimentalConfigError(str(exc)) from exc


def _env_part(value: str) -> str:
    return re.sub(r"[^A-Z0-9]+", "_", value.upper()).strip("_")
