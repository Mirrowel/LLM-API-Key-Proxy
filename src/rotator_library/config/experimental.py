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
import string
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional
from urllib.parse import parse_qsl, urlparse

from ..field_cache import FieldCacheInjection, FieldCacheRule
from ..usage.costs import ModelPricing

_CONFIG_ENV_KEYS = ("LLM_PROXY_CONFIG_FILE", "PROXY_CONFIG_FILE")
_KNOWN_SECTIONS = {"routing", "pricing", "streaming", "field_cache", "providers", "retry", "responses", "hooks"}
_SECRET_KEY_PARTS = ("api_key", "apikey", "authorization", "access_token", "accesstoken", "refresh_token", "refreshtoken", "oauth_token", "oauthtoken", "oauth_token_secret", "oauthtokensecret", "id_token", "idtoken", "token_secret", "tokensecret", "client_secret", "clientsecret", "secret_key", "secretkey", "bearer_token", "bearertoken", "credential", "credentials", "password", "token", "key", "auth")
_BARE_SECRET_KEY_PARTS = frozenset({"token", "key", "auth"})
_PROVIDER_CONFIG_KEYS = {
    "protocol_name",
    "api_base",
    "endpoint_paths",
    "auth_mode",
    "auth_header_name",
    "models",
    "adapter_names",
    "adapter_config",
    "native_streaming_supported",
    "field_cache",
    "model_quota_groups",
    "hooks",
    # D13 transport profiles (fix-pass G7): multi-variant providers become
    # declarable from JSON config.
    "transport_profiles",
    "default_profile",
    "profiles",
    "model_protocols",
    "cache_replay",
    # G8 capability table: ordered per-model rule rows (the class
    # ``model_rules`` declaration, configurable from JSON; rows append
    # after the class rows so config overrides code).
    "model_rules",
}
_HTTP_TOKEN_RE = re.compile(r"^[!#$%&'*+.^_`|~0-9A-Za-z-]+$")
# Parsed-config cache keyed by (path, mtime) — the request path must never
# re-parse JSON from disk (fix-pass G7 config hardening).
_CONFIG_CACHE: dict[tuple[str, int], "ExperimentalConfig"] = {}
_PROVIDER_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")
_ENDPOINT_TEMPLATE_FIELDS = {"model", "operation", "provider"}
_LISTING_OPERATIONS = frozenset({"models"})


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
    hooks: dict[str, Any] = field(default_factory=dict)
    unknown_sections: dict[str, Any] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()
    path: Optional[str] = None

    @property
    def is_empty(self) -> bool:
        return not (self.routing or self.pricing or self.streaming or self.field_cache or self.providers or self.retry or self.responses or self.hooks or self.unknown_sections)


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


@dataclass(frozen=True)
class ResponsesStoreRuntimeSettings:
    """Runtime backend selection for Responses storage."""

    backend: str = "engine"
    cache_name: str = "responses"
    cache_prefix: str = "responses"
    cache_dir: Optional[str] = None
    cache_memory_ttl_seconds: int = 3600
    cache_disk_ttl_seconds: int = 172800


@dataclass(frozen=True)
class ProviderRuntimeConfig:
    """Safe provider metadata loaded from optional JSON config."""

    protocol_name: Optional[str] = None
    api_base: Optional[str] = None
    endpoint_paths: dict[str, str] = field(default_factory=dict)
    auth_mode: str = "bearer"
    auth_header_name: Optional[str] = None
    models: tuple[str, ...] = ()
    adapter_names: Optional[tuple[str, ...]] = None
    adapter_config: dict[str, dict[str, Any]] = field(default_factory=dict)
    native_streaming_supported: Optional[bool] = None
    field_cache_rules: tuple[FieldCacheRule, ...] = ()
    model_quota_groups: Optional[dict[str, list[str]]] = None
    # G2 hooks: JSON provider ``hooks`` entries (names/objects) that add to
    # the provider class declaration. None means "not configured".
    hooks: Optional[tuple[Any, ...]] = None
    # G8 capability table: ordered per-model rule rows appending after the
    # provider class ``model_rules`` declaration (config overrides code).
    model_rules: tuple[dict[str, Any], ...] = ()
    # D13 transport profiles (fix-pass G7): multi-variant providers declare
    # profiles in JSON; dynamic providers bind them onto the instance.
    transport_profiles: Optional[dict[str, dict[str, Any]]] = None
    default_profile: Optional[str] = None


def load_experimental_config(path: str | os.PathLike[str] | None = None, env: Mapping[str, str] | None = None) -> ExperimentalConfig:
    """Load optional JSON config from an explicit path or config env var.

    Fail-loud contract (fix-pass G7): a config path that was explicitly
    configured (env var or argument) but does not exist is an operator
    error, not "no config". Unset means unset. Results are cached per
    (path, env identity) so the request path never re-parses from disk.
    """

    source = env if env is not None else os.environ
    explicit_path = path is not None
    resolved = Path(path) if path is not None else _path_from_env(source)
    if resolved is None:
        return ExperimentalConfig(path=None)
    if not resolved.exists():
        if explicit_path or _path_from_env(source) is not None:
            raise ExperimentalConfigError(
                f"Config file not found: {resolved} — an explicitly configured "
                "path must exist (unset the config env var or fix the path)"
            )
        return ExperimentalConfig(path=str(resolved))
    cache_key = (str(resolved), resolved.stat().st_mtime_ns)
    cached = _CONFIG_CACHE.get(cache_key)
    if cached is not None:
        return cached
    try:
        data = json.loads(resolved.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ExperimentalConfigError(f"Invalid JSON config at {resolved}: {exc.msg}") from exc
    if not isinstance(data, dict):
        raise ExperimentalConfigError("JSON config root must be an object")
    _reject_secret_keys(data)
    _validate_provider_sections(data.get("providers", {}))
    _validate_global_hooks(data.get("hooks"))
    warnings = tuple(f"Unknown config section '{key}' ignored by current runtime" for key in data if key not in _KNOWN_SECTIONS)
    unknown = {key: value for key, value in data.items() if key not in _KNOWN_SECTIONS}
    parsed = ExperimentalConfig(
        routing=_dict_section(data, "routing"),
        pricing=_dict_section(data, "pricing"),
        streaming=_dict_section(data, "streaming"),
        field_cache=_dict_section(data, "field_cache"),
        providers=_dict_section(data, "providers"),
        retry=_dict_section(data, "retry"),
        responses=_dict_section(data, "responses"),
        hooks=_dict_section(data, "hooks"),
        unknown_sections=unknown,
        warnings=warnings,
        path=str(resolved),
    )
    if len(_CONFIG_CACHE) > 8:
        _CONFIG_CACHE.clear()
    _CONFIG_CACHE[cache_key] = parsed
    return parsed


def load_config_from_mapping(data: Mapping[str, Any]) -> ExperimentalConfig:
    """Build config from an in-memory mapping for tests and provider helpers."""

    _reject_secret_keys(data)
    _validate_provider_sections(data.get("providers", {}))
    _validate_global_hooks(data.get("hooks"))
    warnings = tuple(f"Unknown config section '{key}' ignored by current runtime" for key in data if key not in _KNOWN_SECTIONS)
    return ExperimentalConfig(
        routing=_dict_section(data, "routing"),
        pricing=_dict_section(data, "pricing"),
        streaming=_dict_section(data, "streaming"),
        field_cache=_dict_section(data, "field_cache"),
        providers=_dict_section(data, "providers"),
        retry=_dict_section(data, "retry"),
        responses=_dict_section(data, "responses"),
        hooks=_dict_section(data, "hooks"),
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
        provider_cooldown_min_seconds=max(0, _int_setting(source, "PROVIDER_COOLDOWN_MIN_SECONDS", cooldown, "provider_cooldown_min_seconds", 10)),
        provider_cooldown_default_seconds=max(0, _int_setting(source, "PROVIDER_COOLDOWN_DEFAULT_SECONDS", cooldown, "provider_cooldown_default_seconds", 30)),
        provider_cooldown_on_quota=_bool_setting(source, "PROVIDER_COOLDOWN_ON_QUOTA", cooldown, "provider_cooldown_on_quota", False),
        provider_backoff_window_seconds=max(0, _int_setting(source, "PROVIDER_BACKOFF_WINDOW_SECONDS", backoff, "provider_backoff_window_seconds", 60)),
        provider_backoff_threshold=max(1, _int_setting(source, "PROVIDER_BACKOFF_THRESHOLD", backoff, "provider_backoff_threshold", 3)),
        provider_backoff_base_seconds=_optional_int_setting(source, "PROVIDER_BACKOFF_BASE_SECONDS", backoff, "provider_backoff_base_seconds"),
        provider_backoff_max_seconds=max(1, _int_setting(source, "PROVIDER_BACKOFF_MAX_SECONDS", backoff, "provider_backoff_max_seconds", 300)),
        failure_history_max_entries=max(1, _int_setting(source, "FAILURE_HISTORY_MAX_ENTRIES", backoff, "failure_history_max_entries", 200)),
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
    if max_items is None:
        # Bounded by default even when unset: an unbounded durable store is a
        # disk-growth bug waiting for the first long-running deployment.
        max_items = 10000
    return ResponsesStoreSettings(
        ttl_seconds=ttl_seconds,
        max_items=max_items,
        store_failed=as_bool(_env_or_json(source, "RESPONSES_STORE_FAILED", store, "store_failed", default=True), name="RESPONSES_STORE_FAILED"),
        store_in_progress=as_bool(_env_or_json(source, "RESPONSES_STORE_IN_PROGRESS", store, "store_in_progress", default=False), name="RESPONSES_STORE_IN_PROGRESS"),
    )


def get_responses_store_runtime_settings(
    *,
    config: ExperimentalConfig | None = None,
    env: Mapping[str, str] | None = None,
) -> ResponsesStoreRuntimeSettings:
    """Return Responses store backend settings with env overriding JSON."""

    source = env if env is not None else os.environ
    active = config if config is not None else load_experimental_config(env=source)
    responses = active.responses if isinstance(active.responses, dict) else {}
    store = responses.get("store", {}) if isinstance(responses.get("store"), dict) else responses
    backend = str(_env_or_json(source, "RESPONSES_STORE_BACKEND", store, "backend", default="engine")).strip().lower()
    if backend == "provider_cache":
        backend = "engine"
    if backend not in {"memory", "engine"}:
        raise ExperimentalConfigError("RESPONSES_STORE_BACKEND must be 'memory' or 'engine'")
    return ResponsesStoreRuntimeSettings(
        backend=backend,
        cache_name=str(_env_or_json(source, "RESPONSES_STORE_CACHE_NAME", store, "cache_name", default="responses")),
        cache_prefix=str(_env_or_json(source, "RESPONSES_STORE_CACHE_PREFIX", store, "cache_prefix", default="responses")),
        cache_dir=_optional_string(_env_or_json(source, "RESPONSES_STORE_CACHE_DIR", store, "cache_dir")),
        cache_memory_ttl_seconds=max(1, _int_setting(source, "RESPONSES_STORE_CACHE_MEMORY_TTL_SECONDS", store, "cache_memory_ttl_seconds", 3600)),
        cache_disk_ttl_seconds=max(1, _int_setting(source, "RESPONSES_STORE_CACHE_DISK_TTL_SECONDS", store, "cache_disk_ttl_seconds", 172800)),
    )


def get_provider_runtime_config(
    provider: str,
    model: str = "",
    *,
    config: ExperimentalConfig | None = None,
    env: Mapping[str, str] | None = None,
) -> ProviderRuntimeConfig:
    """Return safe JSON-configured provider metadata for one provider/model."""

    source = env if env is not None else os.environ
    active = config if config is not None else load_experimental_config(env=source)
    providers = active.providers if isinstance(active.providers, dict) else {}
    raw = providers.get(str(provider).lower(), {})
    if not isinstance(raw, Mapping) or not raw:
        return ProviderRuntimeConfig()
    _validate_provider_sections({provider: raw})
    protocol_name = _configured_provider_protocol(raw.get("protocol_name"))
    adapter_names = _configured_adapters(raw.get("adapter_names"))
    adapter_config = _configured_adapter_config(raw.get("adapter_config", {}))
    native_streaming_supported = None
    if "native_streaming_supported" in raw:
        native_streaming_supported = as_bool(raw.get("native_streaming_supported"), name="providers.native_streaming_supported")
    field_cache_rules = _configured_provider_field_cache(provider, model, raw.get("field_cache"))
    model_quota_groups = _configured_quota_groups(raw.get("model_quota_groups")) if "model_quota_groups" in raw else None
    hooks = _configured_hooks(raw.get("hooks")) if "hooks" in raw else None
    model_rules = _configured_model_rules(raw.get("model_rules")) if "model_rules" in raw else ()
    transport_profiles = _configured_transport_profiles(_profiles_raw(raw))
    default_profile = _configured_default_profile(raw.get("default_profile"), transport_profiles)
    return ProviderRuntimeConfig(
        protocol_name=protocol_name,
        api_base=_configured_api_base(raw.get("api_base")),
        endpoint_paths=_configured_endpoint_paths(raw.get("endpoint_paths"), protocol_name),
        auth_mode=_configured_auth_mode(raw.get("auth_mode")),
        auth_header_name=_configured_auth_header_name(raw.get("auth_header_name")),
        models=_configured_models(raw.get("models")),
        adapter_names=adapter_names,
        adapter_config=adapter_config,
        native_streaming_supported=native_streaming_supported,
        field_cache_rules=field_cache_rules,
        model_quota_groups=model_quota_groups,
        hooks=hooks,
        model_rules=model_rules,
        transport_profiles=transport_profiles,
        default_profile=default_profile,
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
        raise ExperimentalConfigError("field_cache provider section must be an object")
    raw_rules: list[Any] = []
    keys = ["*"]
    if "/" in model:
        keys.append(model.split("/", 1)[1])
    keys.append(model)
    for key in dict.fromkeys(keys):
        value = provider_rules.get(key, [])
        if isinstance(value, list):
            raw_rules.extend(value)
        elif value not in (None, []):
            raise ExperimentalConfigError("field_cache model rules must be a list")
    parsed_rules = []
    for rule in raw_rules:
        if not isinstance(rule, dict):
            raise ExperimentalConfigError("field_cache rule entries must be objects")
        parsed_rules.append(_field_cache_rule_from_dict(rule))
    return tuple(parsed_rules)


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
            if _is_secret_like_key(key):
                raise ExperimentalConfigError(f"Unsafe secret-like key in JSON config at {path}.{key}")
            _reject_secret_keys(nested, f"{path}.{key}")
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            _reject_secret_keys(nested, f"{path}[{index}]")


def _validate_provider_sections(value: Any) -> None:
    if value in (None, {}):
        return
    if not isinstance(value, Mapping):
        raise ExperimentalConfigError("providers config section must be an object")
    for provider, raw in value.items():
        if not _PROVIDER_NAME_RE.fullmatch(str(provider)):
            raise ExperimentalConfigError(
                f"providers key {provider!r} must contain only letters, numbers, underscores, or hyphens"
            )
        if not isinstance(raw, Mapping):
            raise ExperimentalConfigError(f"providers.{provider} must be an object")
        unsupported = set(str(key) for key in raw) - _PROVIDER_CONFIG_KEYS
        if unsupported:
            raise ExperimentalConfigError(f"providers.{provider} contains unsupported keys: {', '.join(sorted(unsupported))}")
        _configured_provider_protocol(raw.get("protocol_name"))
        _configured_adapters(raw.get("adapter_names"))
        _configured_adapter_config(raw.get("adapter_config", {}))
        _configured_api_base(raw.get("api_base"))
        _configured_endpoint_paths(raw.get("endpoint_paths"), raw.get("protocol_name"))
        _configured_default_profile(
            raw.get("default_profile"), _configured_transport_profiles(_profiles_raw(raw))
        )
        auth_mode = _configured_auth_mode(raw.get("auth_mode"))
        auth_header_name = _configured_auth_header_name(raw.get("auth_header_name"))
        if auth_mode == "custom" and not auth_header_name:
            raise ExperimentalConfigError("providers.auth_header_name is required when auth_mode is custom")
        if auth_mode == "none" and not raw.get("protocol_name"):
            raise ExperimentalConfigError(
                "providers.protocol_name is required when auth_mode is none"
            )
        _configured_models(raw.get("models"))
        if "native_streaming_supported" in raw:
            as_bool(raw.get("native_streaming_supported"), name="providers.native_streaming_supported")
        if "model_quota_groups" in raw:
            _configured_quota_groups(raw.get("model_quota_groups"))
        if "model_rules" in raw:
            _configured_model_rules(raw.get("model_rules"))
        if "hooks" in raw:
            _configured_hooks(raw.get("hooks"))


def _configured_protocol(value: Any) -> Optional[str]:
    if value in (None, ""):
        return None
    name = str(value).strip().lower()
    try:
        from ..protocols import get_protocol

        protocol = get_protocol(name)
    except Exception as exc:
        raise ExperimentalConfigError(f"Unknown provider protocol_name {name!r}") from exc
    return protocol.name


def _configured_provider_protocol(value: Any) -> Optional[str]:
    """Return one protocol supported by the generative provider runtime."""

    protocol_name = _configured_protocol(value)
    if protocol_name:
        # Registry-derived allowlist (G11): every registered protocol
        # declaring a generative operation qualifies — sibling variants
        # included; the allowlist never drifts from the registry again.
        from ..protocols.registry import is_generative_protocol

        if not is_generative_protocol(str(protocol_name)):
            raise ExperimentalConfigError(
                f"protocol_name must be a supported generative protocol, got {protocol_name!r}"
            )
    return protocol_name


def _configured_api_base(value: Any) -> Optional[str]:
    if value in (None, ""):
        return None
    url = str(value).strip()
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ExperimentalConfigError("providers.api_base must be an http(s) URL")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ExperimentalConfigError(
            "providers.api_base cannot contain credentials, query parameters, or fragments"
        )
    return url.rstrip("/")


def _supported_protocol_operations(protocol: Any) -> Optional[frozenset[str]]:
    """Return the declared operations for a protocol, or None if unknown."""

    if protocol in (None, ""):
        return None
    try:
        from ..protocols import get_protocol

        return frozenset(get_protocol(str(protocol).strip().lower()).supported_operations)
    except Exception:
        return None


def _configured_endpoint_paths(value: Any, protocol: Any = None) -> dict[str, str]:
    if value in (None, {}):
        return {}
    if not isinstance(value, Mapping):
        raise ExperimentalConfigError("providers.endpoint_paths must be an object")
    supported = _supported_protocol_operations(protocol)
    result: dict[str, str] = {}
    for operation, path in value.items():
        name = str(operation).strip()
        if not isinstance(path, str):
            raise ExperimentalConfigError("providers.endpoint_paths values must be strings")
        if not name:
            raise ExperimentalConfigError("providers.endpoint_paths entries require a non-empty operation name")
        if not _PROVIDER_NAME_RE.fullmatch(name):
            raise ExperimentalConfigError("providers.endpoint_paths operation names are invalid")
        if (
            supported is not None
            and name not in supported
            and name not in _LISTING_OPERATIONS
        ):
            raise ExperimentalConfigError(
                f"providers.endpoint_paths declares unsupported operation {name!r} "
                f"for protocol {protocol!r}"
            )
        result[name] = _validate_endpoint_path(path)
    return result


def _configured_endpoint_path(value: Any) -> Optional[str]:
    """Validate one singular ``endpoint_path`` declaration (plan 2.8 fallback)."""

    if value in (None, ""):
        return None
    if not isinstance(value, str):
        raise ExperimentalConfigError("providers.endpoint_path values must be strings")
    return _validate_endpoint_path(value)


def _validate_endpoint_path(path: str) -> str:
    """Validate one endpoint path template and return its stripped form."""

    rendered = path.strip()
    if not rendered:
        raise ExperimentalConfigError("providers.endpoint_paths entries require non-empty path values")
    try:
        fields = {
            field_name
            for _, field_name, _, _ in string.Formatter().parse(rendered)
            if field_name is not None
        }
    except ValueError as exc:
        raise ExperimentalConfigError("providers.endpoint_paths contains an invalid template") from exc
    unsupported = fields - _ENDPOINT_TEMPLATE_FIELDS
    if unsupported:
        raise ExperimentalConfigError(
            f"providers.endpoint_paths contains unsupported placeholders: {', '.join(sorted(unsupported))}"
        )
    if not rendered.startswith("/") or rendered.startswith("//"):
        raise ExperimentalConfigError(
            "providers.endpoint_paths values must be absolute paths on api_base"
        )
    parsed_path = urlparse(rendered)
    if parsed_path.fragment:
        raise ExperimentalConfigError(
            "providers.endpoint_paths cannot contain fragments"
        )
    for query_key, _ in parse_qsl(parsed_path.query, keep_blank_values=True):
        if _is_secret_like_key(query_key):
            raise ExperimentalConfigError(
                "providers.endpoint_paths cannot contain secret-bearing query parameters"
            )
    return rendered


def _profiles_raw(raw: Mapping[str, Any]) -> Any:
    """Return the declared profile block (``transport_profiles`` wins)."""

    if "transport_profiles" in raw:
        return raw.get("transport_profiles")
    return raw.get("profiles")


def _configured_transport_profiles(value: Any) -> Optional[dict[str, dict[str, Any]]]:
    """Normalize D13 transport profiles from a provider JSON block.

    Plan 2.8: each profile declares ``protocol`` plus optional per-operation
    ``endpoint_paths`` (singular ``endpoint_path`` fallback) and optional
    per-profile ``auth_mode``/``auth_header_name``. Normalized entries keep
    the ``protocol`` key used by profile resolution and the provider hooks.
    """

    if value in (None, {}):
        return None
    if not isinstance(value, Mapping):
        raise ExperimentalConfigError("providers.transport_profiles must be an object")
    result: dict[str, dict[str, Any]] = {}
    for name, entry in value.items():
        profile_name = str(name).strip()
        if not profile_name or not _PROVIDER_NAME_RE.fullmatch(profile_name):
            raise ExperimentalConfigError("providers.transport_profiles profile names are invalid")
        if not isinstance(entry, Mapping):
            raise ExperimentalConfigError("providers.transport_profiles entries must be objects")
        unsupported = set(str(key) for key in entry) - {
            "protocol",
            "protocol_name",
            "endpoint_paths",
            "endpoint_path",
            "auth_mode",
            "auth_header_name",
        }
        if unsupported:
            raise ExperimentalConfigError(
                f"providers.transport_profiles.{profile_name} contains unsupported keys: "
                f"{', '.join(sorted(unsupported))}"
            )
        protocol = _configured_provider_protocol(entry.get("protocol") or entry.get("protocol_name"))
        auth_mode = _configured_auth_mode(entry.get("auth_mode")) if entry.get("auth_mode") else None
        auth_header_name = (
            _configured_auth_header_name(entry.get("auth_header_name"))
            if entry.get("auth_header_name")
            else None
        )
        if auth_mode == "custom" and not auth_header_name:
            raise ExperimentalConfigError(
                f"providers.transport_profiles.{profile_name}.auth_header_name is required when auth_mode is custom"
            )
        result[profile_name] = {
            "protocol": protocol,
            "endpoint_paths": _configured_endpoint_paths(entry.get("endpoint_paths"), protocol),
            "endpoint_path": _configured_endpoint_path(entry.get("endpoint_path")),
            "auth_mode": auth_mode,
            "auth_header_name": auth_header_name,
        }
    return result


def _configured_default_profile(
    value: Any, profiles: Optional[dict[str, dict[str, Any]]]
) -> Optional[str]:
    """Validate ``default_profile`` against the declared profile names."""

    if value in (None, ""):
        return None
    name = str(value).strip()
    if not name or not _PROVIDER_NAME_RE.fullmatch(name):
        raise ExperimentalConfigError("providers.default_profile must be a valid profile name")
    if not profiles or name not in profiles:
        raise ExperimentalConfigError(
            f"providers.default_profile {name!r} is not a declared transport profile"
        )
    return name


def _configured_auth_mode(value: Any) -> str:
    mode = str(value or "bearer").strip().lower().replace("_", "-")
    aliases = {"x-api-key": "x-api-key", "x-goog-api-key": "x-goog-api-key", "none": "none", "custom": "custom", "bearer": "bearer"}
    if mode not in aliases:
        raise ExperimentalConfigError("providers.auth_mode must be bearer, x-api-key, x-goog-api-key, custom, or none")
    return aliases[mode]


def _is_secret_like_key(value: Any) -> bool:
    """Return whether a config key names credential or authentication data.

    Bare generic parts (``token``/``key``/``auth``) match only the whole key
    so compound, legitimate keys (``auth_mode``, ``cache_key``) survive.
    """

    key_text = str(value).lower()
    if key_text in _BARE_SECRET_KEY_PARTS:
        return True
    compound_parts = tuple(
        part for part in _SECRET_KEY_PARTS if part not in _BARE_SECRET_KEY_PARTS
    )
    compact_key = re.sub(r"[^a-z0-9]+", "", key_text)
    underscored_key = re.sub(r"[^a-z0-9]+", "_", key_text)
    return any(
        part in key_text or part in compact_key or part in underscored_key
        for part in compound_parts
    )


def _configured_auth_header_name(value: Any) -> Optional[str]:
    if value in (None, ""):
        return None
    name = str(value).strip()
    if not _HTTP_TOKEN_RE.fullmatch(name):
        raise ExperimentalConfigError("providers.auth_header_name must be a valid HTTP header name")
    return name


def _configured_models(value: Any) -> tuple[str, ...]:
    if value in (None, []):
        return ()
    if not isinstance(value, (list, tuple)):
        raise ExperimentalConfigError("providers.models must be a list")
    if not all(isinstance(model, str) and model.strip() for model in value):
        raise ExperimentalConfigError("providers.models entries must be non-empty strings")
    models = tuple(model.strip() for model in value)
    return models


def _configured_adapters(value: Any) -> Optional[tuple[str, ...]]:
    if value in (None, ""):
        return None
    if isinstance(value, str):
        names = tuple(part.strip() for part in value.split(",") if part.strip())
    elif isinstance(value, (list, tuple)):
        names = tuple(str(part) for part in value)
    else:
        raise ExperimentalConfigError("providers.adapter_names must be a string or list")
    for name in names:
        _validate_adapter_name(name)
    return names


def _configured_adapter_config(value: Any) -> dict[str, dict[str, Any]]:
    if value in (None, {}):
        return {}
    if not isinstance(value, Mapping):
        raise ExperimentalConfigError("providers.adapter_config must be an object")
    result: dict[str, dict[str, Any]] = {}
    for name, config in value.items():
        adapter_name = str(name)
        _validate_adapter_name(adapter_name)
        if not isinstance(config, Mapping):
            raise ExperimentalConfigError("providers.adapter_config entries must be objects")
        result[adapter_name] = dict(config)
    return result


def _validate_adapter_name(name: str) -> None:
    try:
        from ..adapters import get_adapter

        get_adapter(name)
    except Exception as exc:
        raise ExperimentalConfigError(f"Unknown provider adapter {name!r}") from exc


def get_global_hook_names(
    *,
    config: ExperimentalConfig | None = None,
    env: Mapping[str, str] | None = None,
) -> tuple[str, ...]:
    """Return the process-wide hook names declared under ``hooks.global``.

    Global hooks are names resolved against the hook registry (hooks/registry.py)
    and applied to every request after provider class/config declarations.
    Unknown names fail at startup via ``validate_declared_names``.
    """

    source = env if env is not None else os.environ
    active = config if config is not None else load_experimental_config(env=source)
    section = active.hooks if isinstance(active.hooks, dict) else {}
    raw = section.get("global")
    if raw in (None, ""):
        return ()
    if isinstance(raw, str):
        return tuple(part.strip() for part in raw.split(",") if part.strip())
    if isinstance(raw, (list, tuple)):
        names: list[str] = []
        for entry in raw:
            name = entry.get("name") if isinstance(entry, Mapping) else entry
            if name is not None and str(name).strip():
                names.append(str(name).strip())
        return tuple(names)
    raise ExperimentalConfigError("hooks.global must be a list of hook names")


def _validate_global_hooks(value: Any) -> None:
    if value in (None, {}):
        return
    if not isinstance(value, Mapping):
        raise ExperimentalConfigError("hooks config section must be an object")
    unsupported = set(str(key) for key in value) - {"global"}
    if unsupported:
        raise ExperimentalConfigError(f"hooks contains unsupported keys: {', '.join(sorted(unsupported))}")
    raw = value.get("global")
    if raw in (None, ""):
        return
    if isinstance(raw, str):
        return
    if not isinstance(raw, (list, tuple)):
        raise ExperimentalConfigError("hooks.global must be a list of hook names")
    for entry in raw:
        name = entry.get("name") if isinstance(entry, Mapping) else entry
        if not isinstance(name, str) or not name.strip():
            raise ExperimentalConfigError("hooks.global entries must be non-empty hook names")


def _configured_hooks(value: Any) -> Optional[tuple[Any, ...]]:
    """Normalize a provider JSON ``hooks`` declaration into registry entries.

    Entries are hook names or objects (``name`` + optional ``stages`` /
    ``priority`` / ``critical``). Names are validated against the registry at
    startup by ``providers.validate_provider_hooks`` — never per request.
    """

    if value in (None, ""):
        return None
    if isinstance(value, str):
        entries: list[Any] = [part.strip() for part in value.split(",") if part.strip()]
    elif isinstance(value, (list, tuple)):
        entries = list(value)
    else:
        raise ExperimentalConfigError("providers.hooks must be a string or list")
    normalized: list[Any] = []
    for entry in entries:
        if isinstance(entry, str):
            name = entry.strip()
            if name:
                normalized.append({"name": name})
            continue
        if isinstance(entry, Mapping):
            name = entry.get("name")
            if not isinstance(name, str) or not name.strip():
                raise ExperimentalConfigError("providers.hooks entries require a non-empty name")
            item = {str(key): value for key, value in entry.items()}
            item["name"] = name.strip()
            normalized.append(item)
            continue
        raise ExperimentalConfigError("providers.hooks entries must be hook names or objects")
    return tuple(normalized)


def _configured_provider_field_cache(provider: str, model: str, value: Any) -> tuple[FieldCacheRule, ...]:
    if value in (None, {}, []):
        return ()
    if isinstance(value, list):
        section = {provider: {"*": value}}
    elif isinstance(value, Mapping):
        section = {provider: dict(value)}
    else:
        raise ExperimentalConfigError("providers.field_cache must be an object or list")
    return parse_field_cache_rules(load_config_from_mapping({"field_cache": section}), provider, model)


def _configured_quota_groups(value: Any) -> dict[str, list[str]]:
    if not isinstance(value, Mapping):
        raise ExperimentalConfigError("providers.model_quota_groups must be an object")
    result: dict[str, list[str]] = {}
    for group, models in value.items():
        if not isinstance(models, list) or not all(isinstance(model, str) for model in models):
            raise ExperimentalConfigError("providers.model_quota_groups values must be string arrays")
        result[str(group)] = list(models)
    return result


_MODEL_RULE_ROW_KEYS = frozenset(
    {
        "match",
        "strip",
        "clamp",
        "map",
        "rename",
        "strip_override",
        "effort_accept",
        "toggle",
        "allow",
        "deny",
    }
)


def _configured_model_rules(value: Any) -> tuple[dict[str, Any], ...]:
    """Validate the JSON ``model_rules`` capability table (G8).

    Shape mirrors the class declaration: an ordered list of rows, each
    with a non-empty ``match`` wildcard and the param-rule vocabulary
    inline (plus the reasoning-effort capability keys and ``allow``/
    ``deny`` face lists).
    """

    if value in (None, [], ()):
        return ()
    if not isinstance(value, (list, tuple)):
        raise ExperimentalConfigError("providers.model_rules must be a list of rule rows")
    rows: list[dict[str, Any]] = []
    for row in value:
        if not isinstance(row, Mapping):
            raise ExperimentalConfigError("providers.model_rules rows must be objects")
        unsupported = set(str(key) for key in row) - _MODEL_RULE_ROW_KEYS
        if unsupported:
            raise ExperimentalConfigError(
                f"providers.model_rules rows contain unsupported keys: {', '.join(sorted(unsupported))}"
            )
        match = row.get("match")
        if not isinstance(match, str) or not match.strip():
            raise ExperimentalConfigError("providers.model_rules rows require a non-empty 'match' wildcard")
        _validate_model_rule_row(row)
        rows.append({str(key): item for key, item in row.items()})
    return tuple(rows)


def _validate_model_rule_row(row: Mapping[str, Any]) -> None:
    strip = row.get("strip")
    if "strip" in row and (not isinstance(strip, list) or not all(isinstance(item, str) for item in strip)):
        raise ExperimentalConfigError("providers.model_rules strip must be a string array")
    strip_override = row.get("strip_override")
    if "strip_override" in row and (
        not isinstance(strip_override, list) or not all(isinstance(item, str) for item in strip_override)
    ):
        raise ExperimentalConfigError("providers.model_rules strip_override must be a string array")
    clamp = row.get("clamp")
    if "clamp" in row:
        if not isinstance(clamp, Mapping) or not clamp:
            raise ExperimentalConfigError("providers.model_rules clamp must be an object")
        for parameter, bounds in clamp.items():
            if (
                not isinstance(bounds, (list, tuple))
                or len(bounds) != 2
                or not all(isinstance(bound, (int, float)) and not isinstance(bound, bool) for bound in bounds)
            ):
                raise ExperimentalConfigError(
                    f"providers.model_rules clamp.{parameter} must be a [min, max] number pair"
                )
    maps = row.get("map")
    if "map" in row and (not isinstance(maps, Mapping) or not all(isinstance(table, Mapping) for table in maps.values())):
        raise ExperimentalConfigError("providers.model_rules map must map parameters to value tables")
    effort_accept = row.get("effort_accept")
    if "effort_accept" in row and (
        not isinstance(effort_accept, list)
        or not effort_accept
        or not all(isinstance(item, str) and item.strip() for item in effort_accept)
    ):
        raise ExperimentalConfigError("providers.model_rules effort_accept must be a non-empty string array")
    toggle = row.get("toggle")
    if "toggle" in row and not isinstance(toggle, bool):
        raise ExperimentalConfigError("providers.model_rules toggle must be a boolean")
    rename = row.get("rename")
    if "rename" in row and (
        not isinstance(rename, Mapping)
        or not all(isinstance(old, str) and isinstance(new, str) for old, new in rename.items())
    ):
        raise ExperimentalConfigError("providers.model_rules rename must map old names to new names")
    for face_key in ("allow", "deny"):
        faces = row.get(face_key)
        if face_key in row and (
            not isinstance(faces, list)
            or not faces
            or not all(isinstance(face, str) and face.strip() for face in faces)
        ):
            raise ExperimentalConfigError(
                f"providers.model_rules {face_key} must be a non-empty list of protocol names"
            )


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
            try:
                values[field_name] = as_float(raw, name=key)
            except ExperimentalConfigError:
                continue
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


def _int_setting(env: Mapping[str, str], env_key: str, data: Mapping[str, Any], json_key: str, default: int) -> int:
    if env_key in env:
        try:
            return int(env.get(env_key) or default)
        except (TypeError, ValueError):
            return default
    return as_int(data.get(json_key, default), name=env_key)


def _optional_int_setting(env: Mapping[str, str], env_key: str, data: Mapping[str, Any], json_key: str) -> Optional[int]:
    if env_key in env:
        try:
            parsed = int(env.get(env_key) or 0)
        except (TypeError, ValueError):
            return None
        return parsed if parsed > 0 else None
    return _optional_positive_int(data.get(json_key), env_key)


def _bool_setting(env: Mapping[str, str], env_key: str, data: Mapping[str, Any], json_key: str, default: bool) -> bool:
    if env_key in env:
        try:
            return as_bool(env.get(env_key), name=env_key)
        except ExperimentalConfigError:
            return default
    return as_bool(data.get(json_key, default), name=env_key)


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


def _optional_string(value: Any) -> Optional[str]:
    if value in (None, ""):
        return None
    return str(value)


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
    elif inject_data is not None:
        raise ExperimentalConfigError("field_cache.inject must be an object")
    elif data.get("target_path"):
        inject = FieldCacheInjection(target=str(data.get("target", "request")), path=str(data["target_path"]))
    scope = data.get("scope", ("provider", "model", "credential", "session"))
    if isinstance(scope, str):
        scope_values = tuple(part.strip() for part in scope.split(",") if part.strip())
    elif isinstance(scope, (list, tuple)):
        scope_values = tuple(str(part) for part in scope)
    else:
        raise ExperimentalConfigError("field_cache.scope must be a string or list")
    if {"provider", "model", "credential", "session"}.difference(scope_values):
        raise ExperimentalConfigError(
            "field_cache.scope must include provider, model, credential, and session"
        )
    try:
        return FieldCacheRule(
            name=str(data["name"]),
            source=_optional_string(data.get("source")),
            path=str(data.get("path", "")),
            field=_optional_string(data.get("field")),
            sources=tuple(str(item) for item in data["sources"]) if isinstance(data.get("sources"), list) else None,
            cache_key=_optional_string(data.get("cache_key")),
            mode=str(data.get("mode", "turn")),
            turn_count=max(1, int(data.get("turn_count", 1) or 1)),
            scope=scope_values,
            inject=inject,
            enabled=as_bool(data.get("enabled", True), name="field_cache.enabled"),
            critical=as_bool(data.get("critical", False), name="field_cache.critical"),
            placeholder=_optional_string(data.get("placeholder")),
            ttl_seconds=int(data["ttl_seconds"]) if data.get("ttl_seconds") is not None else None,
            metadata=_metadata_dict(data.get("metadata", {})),
            allow_missing_session=as_bool(data.get("allow_missing_session", False), name="field_cache.allow_missing_session"),
            max_values=int(data.get("max_values") or 1024),
            max_bytes=int(data.get("max_bytes", 4 * 1024 * 1024)),
        )
    except KeyError as exc:
        raise ExperimentalConfigError(f"Missing field-cache rule key {exc.args[0]}") from exc
    except ValueError as exc:
        raise ExperimentalConfigError(str(exc)) from exc


def _metadata_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    raise ExperimentalConfigError("field_cache.metadata must be an object")


def _env_part(value: str) -> str:
    return re.sub(r"[^A-Z0-9]+", "_", value.upper()).strip("_")
